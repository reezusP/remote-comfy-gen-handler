"""Download handler for model files on RunPod serverless workers.

Handles two download sources:
- CivitAI: Uses download_with_aria.py with a model version ID
- Direct URL: Uses aria2c to download from any URL (HuggingFace, etc.)

Files are downloaded to /runpod-volume/ComfyUI/models/<dest>/.

SHA256 verification + content-addressable dedup:
Each entry may include an optional `sha256` field. When present:
- If a file already exists at the destination and its hash matches, the
  download is skipped and the result includes `cached: true`.
- Otherwise the file is downloaded and its hash verified post-download. On
  mismatch the corrupt file is deleted and the job fails.

`destination_path` may be used as a synonym for `dest` + `filename`. It is a
relative path under MODELS_BASE — e.g. `"loras/sub/m.safetensors"` resolves to
`/runpod-volume/ComfyUI/models/loras/sub/m.safetensors`.
"""

import hashlib
import json
import os
import re
import subprocess
import time

import runpod

MODELS_BASE = "/runpod-volume/ComfyUI/models"
CIVITAI_SCRIPT = "/tools/civitai-downloader/download_with_aria.py"


def _sha256_file(path: str) -> str:
    """Compute SHA256 of a file, reading in 64 KiB chunks."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(64 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _split_destination_path(destination_path: str) -> tuple[str, str]:
    """Split a `destination_path` into (dest_subdir, filename).

    `destination_path` is relative to MODELS_BASE (e.g. "loras/sub/m.safetensors").
    Leading slashes and ".." segments are stripped to keep writes confined.
    """
    cleaned = destination_path.lstrip("/").replace("\\", "/")
    parts = [p for p in cleaned.split("/") if p and p != ".."]
    if not parts:
        raise RuntimeError(f"destination_path is empty after normalization: {destination_path!r}")
    filename = parts[-1]
    dest = "/".join(parts[:-1]) if len(parts) > 1 else ""
    return dest, filename


def _send_progress(job: dict, message: str, percent: float = 0) -> None:
    """Send a progress update to RunPod."""
    try:
        runpod.serverless.progress_update(job, {
            "stage": "download",
            "percent": round(percent, 1),
            "message": message,
        })
    except Exception:
        pass


def _download_civitai(version_id: str, dest_dir: str) -> dict:
    """Download a model from CivitAI using download_with_aria.py.

    Args:
        version_id: CivitAI model version ID.
        dest_dir: Absolute path to destination directory.

    Returns:
        Dict with filename, path, size_mb.
    """
    os.makedirs(dest_dir, exist_ok=True)

    # List files before download to detect the new file
    before = set(os.listdir(dest_dir)) if os.path.isdir(dest_dir) else set()

    result = subprocess.run(
        ["python3", CIVITAI_SCRIPT, "-m", str(version_id), "-o", dest_dir],
        capture_output=True,
        text=True,
        timeout=600,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"CivitAI download failed (exit {result.returncode}): "
            f"{result.stderr.strip() or result.stdout.strip()}"
        )

    # Find newly downloaded file(s)
    after = set(os.listdir(dest_dir)) if os.path.isdir(dest_dir) else set()
    new_files = after - before

    if not new_files:
        raise RuntimeError(
            f"CivitAI download produced no new files. "
            f"stdout: {result.stdout.strip()}"
        )

    # Return info about the first new file (usually there's only one)
    filename = sorted(new_files)[0]
    filepath = os.path.join(dest_dir, filename)
    size_mb = round(os.path.getsize(filepath) / (1024 * 1024), 1)

    return {
        "filename": filename,
        "path": filepath,
        "size_mb": size_mb,
    }


def _parse_aria2c_progress(line: str) -> tuple[float, str] | None:
    """Parse aria2c progress from a summary line.

    aria2c prints lines like:
      [#abc123 1.2GiB/3.5GiB(34%) CN:8 DL:52MiB]
      [#abc123 45MiB/3.5GiB(1%) CN:1 DL:12MiB]

    Returns (percent, speed_str) or None if not a progress line.
    """
    m = re.search(r'\((\d+)%\)', line)
    if not m:
        return None
    pct = int(m.group(1))
    speed = ""
    s = re.search(r'DL:([^\s\]]+)', line)
    if s:
        speed = s.group(1)
    return (pct, speed)


def _download_url(
    url: str,
    dest_dir: str,
    filename: str | None = None,
    job: dict | None = None,
    item_index: int = 0,
    total_items: int = 1,
) -> dict:
    """Download a file from a direct URL using aria2c with progress streaming.

    Args:
        url: Direct download URL.
        dest_dir: Absolute path to destination directory.
        filename: Output filename. If None, derived from URL.
        job: RunPod job dict for progress updates.
        item_index: Current download index (0-based) for progress calculation.
        total_items: Total number of downloads in this batch.

    Returns:
        Dict with filename, path, size_mb.
    """
    os.makedirs(dest_dir, exist_ok=True)

    if not filename:
        filename = url.rstrip("/").rsplit("/", 1)[-1]
        # Strip query params from filename
        if "?" in filename:
            filename = filename.split("?")[0]

    # Stream aria2c output to capture real-time progress
    proc = subprocess.Popen(
        [
            "aria2c", "-d", dest_dir, "-o", filename,
            "--allow-overwrite=true",
            "--summary-interval=3",
            "--console-log-level=notice",
            url,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    output_lines = []
    last_progress_time = 0
    try:
        for line in proc.stdout:
            output_lines.append(line)
            parsed = _parse_aria2c_progress(line)
            if parsed and job:
                dl_pct, speed = parsed
                now = time.time()
                # Throttle progress updates to every 3 seconds
                if now - last_progress_time >= 3:
                    last_progress_time = now
                    # Map download progress into the overall batch progress
                    base_pct = (item_index / total_items) * 100
                    item_pct = (dl_pct / 100) * (100 / total_items)
                    overall_pct = base_pct + item_pct
                    speed_str = f" ({speed}/s)" if speed else ""
                    _send_progress(
                        job,
                        f"Downloading {item_index+1}/{total_items}: "
                        f"{filename} {dl_pct}%{speed_str}",
                        percent=overall_pct,
                    )
    except Exception:
        pass

    proc.wait(timeout=600)

    if proc.returncode != 0:
        full_output = "".join(output_lines).strip()
        raise RuntimeError(
            f"aria2c download failed (exit {proc.returncode}): {full_output}"
        )

    filepath = os.path.join(dest_dir, filename)
    if not os.path.isfile(filepath):
        raise RuntimeError(f"Download completed but file not found: {filepath}")

    size_mb = round(os.path.getsize(filepath) / (1024 * 1024), 1)

    return {
        "filename": filename,
        "path": filepath,
        "size_mb": size_mb,
    }


def _resolve_target(dl: dict) -> tuple[str, str]:
    """Resolve (dest_subdir, filename) from a download entry.

    Supports two shapes:
    - `dest` + (optional) `filename` (ComfyGen native — filename may be derived
      from the URL at download time when None)
    - `destination_path` (BlockFlow preset manifest synonym)
    """
    if "destination_path" in dl and dl["destination_path"]:
        return _split_destination_path(dl["destination_path"])
    dest = dl.get("dest", "checkpoints")
    return dest, dl.get("filename")


def handle(job: dict) -> dict:
    """Handle a download command job.

    Expected input:
    {
        "command": "download",
        "downloads": [
            {"source": "civitai", "version_id": "12345", "dest": "loras"},
            {"source": "url", "url": "https://...", "dest": "checkpoints",
             "filename": "model.safetensors", "sha256": "<optional hex>"},
            {"source": "url", "url": "https://...",
             "destination_path": "loras/sub/m.safetensors", "sha256": "<hex>"}
        ]
    }

    `sha256` (optional, per entry): if present, the post-download hash is
    verified. A mismatch fails the job and removes the corrupt file. If a file
    already exists at the destination with the matching hash, aria2c is not
    invoked and the entry is reported with `cached: true`.

    `destination_path` (optional, per entry): synonym for `dest` + `filename`,
    interpreted relative to MODELS_BASE. Used by BlockFlow's preset manifest.

    Returns:
    {
        "ok": true,
        "files": [
            {"filename": "...", "dest": "loras", "path": "...",
             "size_mb": 123.4, "bytes": 129500000,
             "sha256": "<hex>",     # present iff caller supplied sha256
             "cached": false}       # true if served from existing file
        ]
    }
    """
    start_time = time.time()
    job_input = job["input"]
    job_id = job.get("id", "unknown")
    downloads = job_input.get("downloads", [])

    if not downloads:
        raise RuntimeError("No downloads specified. Provide a 'downloads' array.")

    # Set CivitAI token if provided in the job payload
    civitai_token = job_input.get("civitai_token", "")
    if civitai_token:
        os.environ["CIVITAI_TOKEN"] = civitai_token

    print(f"[job {job_id[:8]}] Download command: {len(downloads)} file(s)")
    results = []

    for i, dl in enumerate(downloads):
        source = dl.get("source", "")
        dest, override_filename = _resolve_target(dl)
        dest_dir = os.path.join(MODELS_BASE, dest)
        expected_sha = dl.get("sha256")

        pct = (i / len(downloads)) * 100
        _send_progress(job, f"Downloading {i+1}/{len(downloads)}", percent=pct)

        if source == "civitai":
            version_id = dl.get("version_id")
            if not version_id:
                raise RuntimeError(f"Download {i+1}: 'version_id' required for civitai source")
            print(f"[job {job_id[:8]}] CivitAI download: version {version_id} -> {dest}")
            info = _download_civitai(str(version_id), dest_dir)
            cached = False

        elif source == "url":
            url = dl.get("url")
            if not url:
                raise RuntimeError(f"Download {i+1}: 'url' required for url source")
            filename = override_filename
            if not filename:
                filename = url.rstrip("/").rsplit("/", 1)[-1]
                if "?" in filename:
                    filename = filename.split("?")[0]
            target_path = os.path.join(dest_dir, filename)

            # Content-addressable dedup: if a file already exists at the target
            # with the expected hash, skip aria2c entirely.
            cached = False
            if expected_sha and os.path.isfile(target_path):
                existing_sha = _sha256_file(target_path)
                if existing_sha == expected_sha:
                    cached = True
                    size_mb = round(os.path.getsize(target_path) / (1024 * 1024), 1)
                    info = {"filename": filename, "path": target_path, "size_mb": size_mb}
                    print(f"[job {job_id[:8]}] Cached: {filename} (sha256 match)")

            if not cached:
                print(f"[job {job_id[:8]}] URL download: {url} -> {dest}/{filename}")
                info = _download_url(
                    url, dest_dir, filename,
                    job=job, item_index=i, total_items=len(downloads),
                )

        else:
            raise RuntimeError(f"Download {i+1}: unknown source '{source}'. Use 'civitai' or 'url'.")

        # Post-download sha256 verification (skip if we just confirmed via cache).
        if expected_sha and not cached:
            actual_sha = _sha256_file(info["path"])
            if actual_sha != expected_sha:
                try:
                    os.unlink(info["path"])
                except OSError:
                    pass
                raise RuntimeError(
                    f"Download {i+1}: sha256 mismatch for {info['filename']}: "
                    f"expected {expected_sha}, got {actual_sha}. Corrupt file removed."
                )
            info["sha256"] = actual_sha
        elif expected_sha and cached:
            info["sha256"] = expected_sha

        info["dest"] = dest
        info["cached"] = cached
        info["bytes"] = os.path.getsize(info["path"])
        results.append(info)
        print(f"[job {job_id[:8]}] Downloaded: {info['filename']} ({info['size_mb']} MB, cached={cached})")

    elapsed = int(time.time() - start_time)
    _send_progress(job, f"Done — {len(results)} file(s) in {elapsed}s", percent=100)
    print(f"[job {job_id[:8]}] Download complete: {len(results)} file(s) in {elapsed}s")

    return {"ok": True, "files": results}


def _cli_main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for the CPU installer pod (`python -m download_handler`).

    Reads a job dict (same shape as the worker dispatch input — `{"input": {...}}`)
    from --job FILE or stdin, runs handle(), prints the result as JSON to stdout,
    and returns 0 iff result["ok"] is truthy. Lets exceptions propagate so the
    pod's exit code (non-zero) signals failure to the installer poller.
    """
    import argparse
    import sys

    p = argparse.ArgumentParser(
        description="Download handler CLI mode — used by the CPU installer pod."
    )
    p.add_argument("--job", help="Path to job JSON file (omit to read stdin).")
    args = p.parse_args(argv)

    if args.job:
        with open(args.job) as f:
            job = json.load(f)
    else:
        job = json.load(sys.stdin)

    result = handle(job)
    print(json.dumps(result))
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    import sys
    sys.exit(_cli_main())
