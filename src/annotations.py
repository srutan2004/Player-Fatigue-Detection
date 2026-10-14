"""Bounded streaming schema audit; do not assume online examples match a release."""

from collections import Counter
from pathlib import Path

import ijson


def audit_annotations(path: Path, sample_limit: int = 5000) -> dict:
    """Inspect real annotation shapes without loading multi-gigabyte JSON into RAM.

    This audit does not guess frame indexing or map player IDs across halves. Those
    choices need decoded-frame overlays and physical-team identity verification.
    """
    if sample_limit < 1:
        raise ValueError("sample_limit must be positive")
    with path.open("rb") as handle:
        first = next(ijson.parse(handle), None)
    if not first or first[1] not in ("start_array", "start_map"):
        raise ValueError("Expected a JSON array or SoccerNet/COCO-style object")
    is_array = first[1] == "start_array"
    prefixes = ["item"] if is_array else ["images.item", "annotations.item"]
    report = {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "layout": "flat_array" if is_array else "object",
        "sections": {},
    }
    for prefix in prefixes:
        count, keys, roles, ids, dimensions, first_examples = (
            0,
            Counter(),
            Counter(),
            set(),
            set(),
            [],
        )
        with path.open("rb") as handle:
            for record in ijson.items(handle, prefix):
                if not isinstance(record, dict):
                    raise ValueError(f"Expected object records at {prefix}")
                count += 1
                keys.update(record.keys())
                attributes = record.get("attributes") or {}
                roles[str(attributes.get("role", record.get("role", "unknown")))] += 1
                player = attributes.get(
                    "player_id", record.get("player_id", record.get("track_id"))
                )
                if player is not None:
                    ids.add(str(player))
                if "width" in record and "height" in record:
                    dimensions.add((int(record["width"]), int(record["height"])))
                if len(first_examples) < 2:
                    first_examples.append(record)
                if count >= sample_limit:
                    break
        report["sections"][prefix] = {
            "sample_records": count,
            "sample_limit_reached": count == sample_limit,
            "field_counts": dict(keys),
            "sample_roles": dict(roles),
            "sample_player_ids": sorted(ids),
            "declared_image_dimensions": sorted(dimensions),
            "examples": first_examples,
        }
    report["alignment_status"] = "requires_visual_verification"
    report["warnings"] = [
        "This is a bounded schema sample, not a count of the complete dataset.",
        "Resolve frame indexing, actual video geometry and coordinate convention before use.",
        "Annotations are measurement references, not fatigue labels.",
    ]
    return report
