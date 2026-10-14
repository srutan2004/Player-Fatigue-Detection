"""Streaming SoccerTrack reference indexing for audits and seeded detector labels."""

import csv
import re
import sqlite3
from contextlib import closing
from pathlib import Path

import ijson

from src.common import provenance, sha256_file, write_json


def annotation_box(record: dict):
    """Normalize shipped dict/list boxes without assuming pitch and image units agree."""
    box = record.get("bbox_image", record.get("bbox"))
    if isinstance(box, dict):
        values = [box.get(key) for key in ("x", "y", "w", "h")]
    elif isinstance(box, list) and len(box) == 4:
        values = box
    else:
        return None
    if any(x is None for x in values):
        return None
    x, y, w, h = map(float, values)
    return [x, y, x + w, y + h] if w > 0 and h > 0 else None


def build_reference_index(source: Path, output: Path, index_base: int = 1) -> dict:
    """Build a disk-backed per-frame index using explicit 1/0-based frame semantics.

    Large reference files are scanned sequentially. Frame IDs are resolved through
    image records, then an explicit six-digit ID suffix fallback. Visually verify
    this indexing against decoded frames before labels or identity links are trusted.
    """
    if index_base not in (0, 1):
        raise ValueError("index_base must be 0 or 1")
    with source.open("rb") as handle:
        first = next(ijson.parse(handle))
    flat = first[1] == "start_array"
    image_frames = {}
    if not flat:
        with source.open("rb") as handle:
            for image in ijson.items(handle, "images.item"):
                token = image.get("frame_id", image.get("frame_index"))
                if token is None:
                    filename = str(image.get("file_name", ""))
                    match = re.search(r"(\d+)(?:\.[a-zA-Z]+)?$", filename)
                    token = int(match.group(1)[-6:]) if match else int(str(image["id"])[-6:])
                image_frames[str(image.get("id", image.get("image_id")))] = int(token) - index_base
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".building.sqlite")
    connection = sqlite3.connect(temporary)
    connection.execute("DROP TABLE IF EXISTS people")
    connection.execute("""CREATE TABLE people (
        frame INTEGER, player TEXT, track TEXT, role TEXT, team TEXT, jersey TEXT,
        x1 REAL, y1 REAL, x2 REAL, y2 REAL, pitch_x REAL, pitch_y REAL)""")
    count, skipped, buffer = 0, 0, []
    try:
        with source.open("rb") as handle:
            for record in ijson.items(handle, "item" if flat else "annotations.item"):
                box = annotation_box(record)
                if box is None:
                    skipped += 1
                    continue
                attrs = record.get("attributes") or {}
                image_id = str(record.get("image_id", record.get("frame_id", "")))
                frame = image_frames.get(image_id)
                if frame is None:
                    frame = int(image_id[-6:]) - index_base
                if frame < 0:
                    raise ValueError("Negative source frame after index-base conversion")
                pitch = record.get("bbox_pitch") or {}
                pitch_x = pitch.get("x_bottom_middle") if isinstance(pitch, dict) else None
                pitch_y = pitch.get("y_bottom_middle") if isinstance(pitch, dict) else None
                player = attrs.get("player_id", record.get("player_id", record.get("track_id")))
                track = record.get("track_id", player)
                row = [
                    frame,
                    str(player) if player is not None else None,
                    str(track),
                    str(attrs.get("role", record.get("role", "unknown"))),
                    str(attrs.get("team_side", attrs.get("team", record.get("team_side", "")))),
                    str(attrs.get("jersey", attrs.get("jersey_number", ""))),
                    *box,
                    float(pitch_x) if pitch_x is not None else None,
                    float(pitch_y) if pitch_y is not None else None,
                ]
                buffer.append(row)
                count += 1
                if len(buffer) >= 10000:
                    connection.executemany(
                        "INSERT INTO people VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", buffer
                    )
                    connection.commit()
                    buffer = []
                    if count % 100000 == 0:
                        print(f"Indexed {count:,} reference boxes", flush=True)
        connection.executemany("INSERT INTO people VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", buffer)
        connection.execute("CREATE INDEX people_frame ON people(frame)")
        connection.execute("CREATE INDEX people_player ON people(player,frame)")
        connection.commit()
    finally:
        connection.close()
    if count == 0:
        raise ValueError("No image-space people boxes found in this annotation release")
    temporary.replace(output)
    report = {
        **provenance(),
        "source": str(source),
        "source_sha256": sha256_file(source),
        "index_base": index_base,
        "image_records": len(image_frames),
        "person_boxes": count,
        "non_box_records_skipped": skipped,
        "alignment_status": "requires_source_frame_overlay_verification",
    }
    write_json(output.with_suffix(".json"), report)
    return report


def reference_people(connection: sqlite3.Connection, frame: int) -> list[dict]:
    """Retrieve observed reference boxes; these labels are not fatigue ground truth."""
    connection.row_factory = sqlite3.Row
    return [dict(row) for row in connection.execute("SELECT * FROM people WHERE frame=?", (frame,))]


def build_mot_index(source: Path, output: Path, index_base: int = 0) -> dict:
    """Index the curated MOT boxes, whose released frame numbering starts at zero.

    These anatomical person rectangles are the detector-label source. GSR image
    boxes can have different geometry and must not be substituted without review.
    """
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".building.sqlite")
    count = 0
    with closing(sqlite3.connect(temporary)) as connection:
        connection.execute("DROP TABLE IF EXISTS people")
        connection.execute("""CREATE TABLE people (
            frame INTEGER, player TEXT, track TEXT, role TEXT, team TEXT, jersey TEXT,
            x1 REAL, y1 REAL, x2 REAL, y2 REAL, pitch_x REAL, pitch_y REAL)""")
        with source.open(encoding="utf-8") as handle:
            buffer = []
            for row in csv.reader(handle):
                if len(row) < 6:
                    raise ValueError("MOT row requires frame, ID, x, y, width, height")
                frame, player = int(row[0]) - index_base, row[1]
                x, y, w, h = map(float, row[2:6])
                if frame < 0 or w <= 0 or h <= 0:
                    raise ValueError("Invalid MOT frame index or box")
                buffer.append(
                    (frame, player, player, "person", "", "", x, y, x + w, y + h, None, None)
                )
                count += 1
                if len(buffer) == 10000:
                    connection.executemany(
                        "INSERT INTO people VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", buffer
                    )
                    buffer = []
            connection.executemany("INSERT INTO people VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", buffer)
        connection.execute("CREATE INDEX people_frame ON people(frame)")
        connection.execute("CREATE INDEX people_player ON people(player,frame)")
        connection.commit()
    temporary.replace(output)
    report = {
        **provenance(),
        "source": str(source),
        "source_sha256": sha256_file(source),
        "format": "mot",
        "index_base": index_base,
        "person_boxes": count,
        "alignment_status": "requires_decoded_overlay_verification",
    }
    write_json(output.with_suffix(".json"), report)
    return report
