"""Check actual schemas, download boundaries and train/test leakage prevention."""

import json

import pytest

from src.annotations import audit_annotations
from src.common import load_config
from src.dataset import REPO, download_files
from src.splits import validate_source_blocks


def test_nested_release_schema_is_audited_without_inventing_coordinates(tmp_path):
    path = tmp_path / "annotations.json"
    path.write_text(
        json.dumps(
            {
                "images": [{"id": "3000001", "width": 3840, "height": 1504}],
                "annotations": [
                    {
                        "image_id": "3000001",
                        "attributes": {"player_id": "p8", "role": "player"},
                        "bbox_pitch": {"x_bottom_middle": -10, "y_bottom_middle": 3},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    report = audit_annotations(path, sample_limit=10)
    sample = report["sections"]["annotations.item"]
    assert sample["sample_player_ids"] == ["p8"]
    assert sample["sample_roles"] == {"player": 1}
    assert "x" not in sample["field_counts"]
    assert report["alignment_status"] == "requires_visual_verification"


def test_annotation_audit_is_bounded(tmp_path):
    path = tmp_path / "flat.json"
    path.write_text(json.dumps([{"player_id": i} for i in range(100)]), encoding="utf-8")
    report = audit_annotations(path, sample_limit=3)
    assert report["sections"]["item"]["sample_records"] == 3
    assert report["sections"]["item"]["sample_limit_reached"]


def test_download_budget_is_checked_before_network_or_directory_creation(tmp_path):
    path = tmp_path / "probe.json"
    path.write_text(
        json.dumps(
            {
                "repo_id": REPO,
                "status": "accessible",
                "resolved_revision": "test",
                "files": [{"path": "videos/half.mp4", "size_bytes": 2 * 1024**3}],
            }
        )
    )
    destination = tmp_path / "absent"
    with pytest.raises(ValueError, match="budget"):
        download_files(path, ["videos/half.mp4"], destination, max_gib=1)
    assert not destination.exists()


def test_source_blocks_respect_temporal_buffer_and_hold_out_whole_match():
    blocks = [
        {"match_id": "A", "half": 1, "start_seconds": 0, "end_seconds": 100, "split": "train"},
        {
            "match_id": "A",
            "half": 1,
            "start_seconds": 130,
            "end_seconds": 200,
            "split": "validation",
        },
        {"match_id": "B", "half": 1, "start_seconds": 0, "end_seconds": 100, "split": "test"},
    ]
    validate_source_blocks(blocks)
    blocks[1]["start_seconds"] = 129
    with pytest.raises(ValueError, match="buffer"):
        validate_source_blocks(blocks)
    blocks[1]["start_seconds"] = 130
    blocks[2]["match_id"] = "A"
    with pytest.raises(ValueError, match="different match"):
        validate_source_blocks(blocks)


def test_pilot_config_paths_do_not_depend_on_shell_working_directory(monkeypatch, tmp_path):
    from src.common import ROOT

    monkeypatch.chdir(tmp_path)
    config = load_config(ROOT / "configs/pilot.yaml")
    assert config["models_dir"] == ROOT / "models"
    assert config["video"]["path"].is_absolute()
