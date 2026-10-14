"""Validate source-block separation before Phase 2 image/tiling selection."""


def validate_source_blocks(blocks: list[dict], buffer_seconds: float = 30) -> None:
    """Reject time leakage between development splits and a reused test match.

    Adjacent video frames share appearance; splitting tiles instead of their source
    scenes would inflate apparent detector generalization without new evidence.
    """
    if buffer_seconds < 0:
        raise ValueError("Temporal buffer must not be negative")
    for block in blocks:
        if block["split"] not in ("train", "validation", "test"):
            raise ValueError("Unknown split")
        if block["start_seconds"] < 0 or block["end_seconds"] <= block["start_seconds"]:
            raise ValueError("Invalid source time interval")
    test_matches = {b["match_id"] for b in blocks if b["split"] == "test"}
    dev_matches = {b["match_id"] for b in blocks if b["split"] != "test"}
    if test_matches & dev_matches:
        raise ValueError("Held-out test images must come from a different match")
    for i, left in enumerate(blocks):
        for right in blocks[i + 1 :]:
            if (left["match_id"], left["half"]) != (right["match_id"], right["half"]):
                continue
            if left["split"] == right["split"]:
                continue
            before, after = sorted([left, right], key=lambda b: b["start_seconds"])
            if after["start_seconds"] - before["end_seconds"] < buffer_seconds:
                raise ValueError("Development source blocks overlap or violate the temporal buffer")
