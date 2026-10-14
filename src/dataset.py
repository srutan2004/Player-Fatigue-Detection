"""Access checks and bounded downloads for the canonical SoccerTrack v2 release."""

from pathlib import Path

from huggingface_hub import HfApi, get_hf_file_metadata, get_token, hf_hub_download, hf_hub_url
from huggingface_hub.errors import GatedRepoError, HfHubHTTPError

from src.common import provenance, sha256_file, write_json

REPO = "atomscott/soccertrack-v2"
ACCESS_URL = f"https://huggingface.co/datasets/{REPO}"


def probe_dataset(output: Path, revision: str = "v1.2", match: str = "117093") -> dict:
    """Resolve a mutable tag to a commit and test actual file access without downloading.

    The public dataset card is readable even when files are gated. Credentials are
    obtained by the Hub library from local login; tokens are never included in reports.
    """
    report = {
        **provenance(),
        "repo_id": REPO,
        "requested_revision": revision,
        "match_id": match,
        "access_url": ACCESS_URL,
        "files": [],
    }
    token = get_token()
    report["local_token_present"] = bool(token)
    try:
        info = HfApi().dataset_info(REPO, revision=revision, files_metadata=True)
        report["resolved_revision"] = info.sha
        report["gated"] = info.gated
        for item in info.siblings:
            name = item.rfilename
            if match in name and name.split("/")[0] in ("videos", "gsr", "raw", "mot"):
                report["files"].append({"path": name, "size_bytes": item.size})
        videos = [item for item in report["files"] if item["path"].endswith(".mp4")]
        if not videos:
            raise ValueError(f"No full-match videos found for match {match}")
        filename = videos[0]["path"]
        get_hf_file_metadata(
            hf_hub_url(REPO, filename, repo_type="dataset", revision=info.sha), token=token
        )
        report["status"] = "accessible"
    except GatedRepoError:
        report.update(
            status="access_required",
            action=(
                f"Accept dataset conditions at {ACCESS_URL}. "
                "Then run uv run --no-sync hf auth login locally. "
                "Do not paste tokens into chat or configuration files."
            ),
        )
    except HfHubHTTPError as exc:
        status = exc.response.status_code if exc.response is not None else None
        report.update(
            status="http_error",
            http_status=status,
            action="Check dataset revision, local authentication, and network connection.",
        )
    except Exception as exc:
        report.update(status="error", error=f"{type(exc).__name__}: {exc}")
    write_json(output, report)
    return report


def download_files(
    probe_path: Path,
    filenames: list[str],
    destination: Path,
    max_gib: float = 5,
    transport: str = "ranges",
) -> dict:
    """Download explicitly selected files at the probed commit within a byte budget.

    Full-match footage is analysis input, not a training set. This intentionally
    provides no whole-repository download or automatic second-match expansion.
    """
    import hashlib
    import json

    probe = json.loads(probe_path.read_text("utf-8"))
    if probe.get("repo_id") != REPO or probe.get("status") != "accessible":
        raise ValueError("An accessible dataset probe is required before downloading")
    if not filenames or len(set(filenames)) != len(filenames):
        raise ValueError("Select at least one file; duplicate filenames are not allowed")
    known = {item["path"]: item["size_bytes"] for item in probe["files"]}
    if any(name not in known or known[name] is None for name in filenames):
        raise ValueError("Every selected file must have a known size in the probe report")
    total = sum(known[name] for name in filenames)
    if max_gib <= 0 or total > max_gib * 1024**3:
        raise ValueError(f"Selection is {total / 1024**3:.2f} GiB, exceeding the download budget")
    destination = destination.resolve()
    for name in filenames:
        if not (destination / name).resolve().is_relative_to(destination):
            raise ValueError("Unsafe dataset filename")
    import shutil

    if shutil.disk_usage(destination if destination.exists() else destination.parent).free < total:
        raise ValueError("Insufficient free disk space for the selected files")
    destination.mkdir(parents=True, exist_ok=True)
    report = {
        **provenance(),
        "repo_id": REPO,
        "revision": probe["resolved_revision"],
        "selected_size_bytes": total,
        "files": [],
    }
    selection_id = hashlib.sha256("\n".join(sorted(filenames)).encode()).hexdigest()[:10]
    manifest_path = destination / "manifests" / f"{probe['match_id']}_{selection_id}.json"
    for name in filenames:
        if transport == "ranges" and known[name] > 32 * 1024**2:
            from src.download import fetch_ranges

            metadata = get_hf_file_metadata(
                hf_hub_url(REPO, name, repo_type="dataset", revision=probe["resolved_revision"]),
                token=get_token(),
            )
            etag = metadata.etag or ""
            checksum = etag if len(etag) == 64 else None
            local = fetch_ranges(metadata.location, destination / name, known[name], checksum)
        else:
            local = Path(
                hf_hub_download(
                    REPO,
                    name,
                    repo_type="dataset",
                    revision=probe["resolved_revision"],
                    local_dir=destination,
                )
            )
        report["files"].append(
            {
                "remote_path": name,
                "local_path": str(local),
                "size_bytes": local.stat().st_size,
                "sha256": sha256_file(local),
            }
        )
        write_json(manifest_path, report)
    return report
