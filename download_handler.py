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
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, Future
from typing import Callable

import runpod

MODELS_BASE = "/runpod-volume/ComfyUI/models"
CIVITAI_SCRIPT = "/tools/civitai-downloader/download_with_aria.py"
CIVITAI_API_BASE = "https://civitai.com/api/v1"


def _civitai_version_metadata(version_id: str, token: str | None = None) -> dict | None:
    """Look up a CivitAI model version's primary file metadata.

    Hits `GET /api/v1/model-versions/{version_id}` and returns
    `{"filename": str, "sha256": str}` for the primary file, or None if the
    call fails, the version isn't published, or no SHA256 hash is reported.

    Used to skip the download subprocess when a file with the expected hash
    already lives on the network volume — same content-addressable dedup the
    URL download path already does, just sourced from the CivitAI API instead
    of caller-supplied metadata. Schema confirmed via Context7 against
    https://developer.civitai.com (per-file `hashes.SHA256`).
    """
    url = f"{CIVITAI_API_BASE}/model-versions/{version_id}"
    headers = {"User-Agent": "comfy-gen-handler/0.2"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read())
    except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError, TimeoutError) as exc:
        print(f"[civitai] version-metadata lookup failed for {version_id}: "
              f"{type(exc).__name__}: {exc}", flush=True)
        return None

    files = data.get("files") or []
    if not files:
        return None
    # Prefer the explicitly-primary file; otherwise the first one. This is
    # also what download_with_aria.py picks.
    primary = next((f for f in files if f.get("primary")), files[0])
    sha = (primary.get("hashes") or {}).get("SHA256")
    name = primary.get("name")
    if not sha or not name:
        return None
    return {"filename": name, "sha256": sha.lower()}


def _find_file_by_sha(dest_dir: str, expected_sha: str, hint_name: str | None = None) -> str | None:
    """Return the path of a file under `dest_dir` whose SHA256 matches.

    Hash budget: at most ONE file. When `hint_name` is given, that's the only
    file we check. With no hint and no expected size info, scanning every file
    in dest_dir and hashing each is catastrophic — on a populated checkpoints/
    on a network volume that's literally minutes of wall time per call (bead
    cwt). Better to declare a miss and let the subprocess decide what to do.

    The rare "same bytes, different filename" edge case is sacrificed for
    predictable latency.
    """
    if not os.path.isdir(dest_dir):
        return None
    if not hint_name:
        return None
    candidate = os.path.join(dest_dir, hint_name)
    if not os.path.isfile(candidate):
        return None
    try:
        if _sha256_file(candidate) == expected_sha.lower():
            return candidate
    except OSError:
        pass
    return None


def _sha256_file(path: str) -> str:
    """Compute SHA256 of a file, reading in 64 KiB chunks."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(64 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_file_with_heartbeat(
    path: str, job_tag: str, label: str, heartbeat_sec: float = 15.0,
) -> str:
    """Like _sha256_file but emits a heartbeat every heartbeat_sec seconds so
    multi-GB hashes over a network volume don't go dark. Used for the
    post-download verify path which can take minutes on large files (bead 8r7)."""
    h = hashlib.sha256()
    started = time.time()
    last_beat = started
    bytes_so_far = 0
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(64 * 1024), b""):
            h.update(chunk)
            bytes_so_far += len(chunk)
            now = time.time()
            if now - last_beat >= heartbeat_sec:
                last_beat = now
                mb = bytes_so_far / (1024 * 1024)
                elapsed = now - started
                rate = mb / elapsed if elapsed > 0 else 0
                print(f"[job {job_tag}] still hashing {label} — "
                      f"{mb:.0f}MB at {rate:.0f}MB/s ({elapsed:.0f}s in)", flush=True)
    return h.hexdigest()


# Background pool for post-download sha256 verification. Single thread —
# disk-bound, parallelism doesn't help and we don't want concurrent giant reads
# competing on the network volume. Lazy-init so import-time stays cheap.
_VERIFY_POOL: ThreadPoolExecutor | None = None
_pending_verifications: list[tuple[int, dict, Future, str]] = []


def _verify_pool() -> ThreadPoolExecutor:
    global _VERIFY_POOL
    if _VERIFY_POOL is None:
        _VERIFY_POOL = ThreadPoolExecutor(max_workers=1, thread_name_prefix="civitai-verify")
    return _VERIFY_POOL


def _async_verify_sha256(path: str, expected: str, *, job_tag: str, label: str) -> str:
    """Compute sha256, raise on mismatch. Designed to run in _verify_pool."""
    actual = _sha256_file_with_heartbeat(path, job_tag, label)
    if actual != expected:
        try:
            os.unlink(path)
        except OSError:
            pass
        raise RuntimeError(
            f"sha256 mismatch for {label}: expected {expected}, got {actual}. "
            f"Corrupt file removed."
        )
    return actual


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


def _download_civitai(
    version_id: str,
    dest_dir: str,
    timeout_sec: int = 600,
    job: dict | None = None,
    item_index: int = 0,
    total_items: int = 1,
    progress_callback: Callable[[dict], None] | None = None,
    expected_sha: str | None = None,
) -> dict:
    """Download a model from CivitAI using download_with_aria.py.

    Streams the subprocess's stdout/stderr line-by-line so multi-GB downloads
    aren't silent for their full duration. aria2c-style progress lines emitted
    by the underlying script are parsed (via `_parse_aria2c_progress`) and
    surfaced as `download_progress` events through `progress_callback`.

    Args:
        version_id: CivitAI model version ID.
        dest_dir: Absolute path to destination directory.
        timeout_sec: Subprocess timeout. Orchestrator-controlled per job
            (BlockFlow computes ~`300 + size_gb * 60`); 600 is a safe minimum.
        job: RunPod job dict — used for progress_update messages and to tag
            stdout lines with the job id.
        item_index: 0-based index of this download in the batch (for overall %).
        total_items: Total number of downloads in this batch.
        progress_callback: Optional event sink (SSE forwarder in install-preset).

    Returns:
        Dict with filename, path, size_mb.
    """
    job_tag = (job.get("id", "")[:8] if job else "") or "civitai"
    print(f"[job {job_tag}] civitai: entering _download_civitai for version {version_id}", flush=True)
    os.makedirs(dest_dir, exist_ok=True)

    # --- Content-addressable dedup ------------------------------------------
    # Mirror the URL download path's `cached: true` skip when a file matching
    # the expected SHA256 already exists. Source of truth for the expected
    # hash, in priority:
    #   1. `expected_sha` arg (caller — e.g. blockflow-presets explicit hash)
    #   2. CivitAI API's per-file hashes.SHA256 (one extra ~100ms GET to avoid
    #      a 5-30min subprocess download). Schema confirmed via Context7.
    # Either way: if a file in dest_dir hashes to the expected SHA, return it
    # with cached=True and skip the subprocess entirely.
    dedup_target_sha = (expected_sha or "").lower() or None
    api_filename: str | None = None
    if not dedup_target_sha:
        meta = _civitai_version_metadata(
            version_id, token=os.environ.get("CIVITAI_TOKEN") or None,
        )
        if meta:
            dedup_target_sha = meta["sha256"]
            api_filename = meta["filename"]
            print(f"[job {job_tag}] civitai: api reports {api_filename} "
                  f"sha256={dedup_target_sha[:12]}…", flush=True)

    if dedup_target_sha:
        cached_hit = _find_file_by_sha(dest_dir, dedup_target_sha, hint_name=api_filename)
        if cached_hit:
            size_mb = round(os.path.getsize(cached_hit) / (1024 * 1024), 1)
            print(f"[job {job_tag}] civitai: cached hit — sha256 match for "
                  f"{os.path.basename(cached_hit)}; skipping download.", flush=True)
            if job:
                _send_progress(
                    job,
                    f"Cached {item_index+1}/{total_items}: "
                    f"{os.path.basename(cached_hit)} (sha256 match)",
                    percent=((item_index + 1) / total_items) * 100,
                )
            if progress_callback:
                progress_callback({
                    "type": "download_done",
                    "file_index": item_index,
                    "file": os.path.basename(cached_hit),
                    "cached": True,
                    "bytes": os.path.getsize(cached_hit),
                    "sha256": dedup_target_sha,
                })
            return {
                "filename": os.path.basename(cached_hit),
                "path": cached_hit,
                "size_mb": size_mb,
                "cached": True,
                "sha256": dedup_target_sha,
            }

    # ------------------------------------------------------------------------
    # No cached hit — proceed with the subprocess download.
    # List files before download as a fallback for detecting what landed.
    # Primary path: parse the CivitAI script's "Model ready at: <path>" line.
    # Diff is unreliable when a prior attempt left files in the dest (resume
    # case): aria2c writes the same filename in place and `after - before` is
    # empty even though the download succeeded.
    before = set(os.listdir(dest_dir)) if os.path.isdir(dest_dir) else set()
    model_ready_path: str | None = None

    proc = subprocess.Popen(
        ["python3", "-u", CIVITAI_SCRIPT, "-m", str(version_id), "-o", dest_dir],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,  # line-buffered
    )

    output_lines: list[str] = []
    last_progress_time = 0.0
    last_heartbeat_time = time.time()
    HEARTBEAT_SEC = 15  # log "still running" every 15s even on silent output

    try:
        assert proc.stdout is not None
        for raw in proc.stdout:
            line = raw.rstrip()
            if not line:
                continue
            output_lines.append(line)
            # The script emits "✅ Model ready at: /abs/path/to/file" on success.
            # Capture the path so we don't need to guess from a directory diff.
            if "Model ready at:" in line and model_ready_path is None:
                _, _, path_part = line.partition("Model ready at:")
                candidate = path_part.strip()
                if candidate and os.path.isfile(candidate):
                    model_ready_path = candidate
            # Surface every line to the worker log so RunPod's log viewer shows
            # live progress. The script's own format (aria2c summary lines,
            # status messages from CivitAI_Downloader) is the most useful thing
            # we can show — anything more structured would risk drift.
            print(f"[job {job_tag}] civitai: {line}", flush=True)

            now = time.time()
            parsed = _parse_aria2c_progress(line)
            if parsed and (now - last_progress_time) >= 3:
                dl_pct, speed = parsed
                last_progress_time = now
                last_heartbeat_time = now
                base_pct = (item_index / total_items) * 100
                item_pct = (dl_pct / 100) * (100 / total_items)
                overall_pct = base_pct + item_pct
                speed_str = f" ({speed}/s)" if speed else ""
                # Prefer the real filename (from CivitAI API) over the abstract
                # civitai/<vid> token in progress messages.
                display_name = api_filename or f"civitai/{version_id}"
                if job:
                    _send_progress(
                        job,
                        f"Downloading {item_index+1}/{total_items}: "
                        f"{display_name} {dl_pct}%{speed_str}",
                        percent=overall_pct,
                    )
                if progress_callback:
                    progress_callback({
                        "type": "download_progress",
                        "file_index": item_index,
                        "file": display_name,
                        "percent": dl_pct,
                        "speed": speed or "",
                    })
            elif (now - last_heartbeat_time) >= HEARTBEAT_SEC:
                last_heartbeat_time = now
                heartbeat_name = api_filename or f"version {version_id}"
                print(f"[job {job_tag}] civitai: ... still downloading {heartbeat_name}", flush=True)
    except Exception as exc:  # noqa: BLE001 — surface and re-raise via returncode below
        print(f"[job {job_tag}] civitai: stream error: {type(exc).__name__}: {exc}", flush=True)

    proc.wait(timeout=timeout_sec)

    if proc.returncode != 0:
        tail = "\n".join(output_lines[-20:]).strip()
        raise RuntimeError(
            f"CivitAI download failed (exit {proc.returncode}): {tail}"
        )

    # Resolve the resulting file.
    # Priority 1: the script told us the path via "Model ready at:".
    # Priority 2: directory diff (works for clean dests).
    # Priority 3: if both fail, treat any newly-mtime'd .safetensors/.gguf/.bin
    #             as a fallback — last-ditch, but covers resume cases where
    #             aria2c wrote the same filename twice.
    if model_ready_path:
        filepath = model_ready_path
        filename = os.path.basename(filepath)
    else:
        after = set(os.listdir(dest_dir)) if os.path.isdir(dest_dir) else set()
        # Ignore aria2's partial-state files when picking the result.
        new_files = {f for f in (after - before) if not f.endswith(".aria2")}
        if new_files:
            filename = sorted(new_files)[0]
            filepath = os.path.join(dest_dir, filename)
        else:
            tail = "\n".join(output_lines[-20:]).strip()
            raise RuntimeError(
                f"CivitAI download produced no new files and no 'Model ready at:' "
                f"line was emitted. tail: {tail}"
            )

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
    progress_callback: Callable[[dict], None] | None = None,
    timeout_sec: int = 600,
    expected_sha: str | None = None,
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

    # Build aria2c command. When expected_sha is supplied we ask aria2c to
    # verify in-flight via --checksum — aria2c already streams the bytes for
    # writing, so adding a hash to the same pipe is essentially free. On
    # mismatch aria2c exits non-zero and we delete the corrupt file, identical
    # outcome to the post-download verify but without the second pass over the
    # entire file (saves 30-180s on multi-GB downloads over network volume).
    aria_cmd = [
        "aria2c", "-d", dest_dir, "-o", filename,
        "--allow-overwrite=true",
        "--summary-interval=3",
        "--console-log-level=notice",
    ]
    if expected_sha:
        aria_cmd.append(f"--checksum=sha-256={expected_sha.lower()}")
    aria_cmd.append(url)

    # Stream aria2c output to capture real-time progress
    proc = subprocess.Popen(
        aria_cmd,
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
                    if progress_callback:
                        progress_callback({
                            "type": "download_progress",
                            "file_index": item_index,
                            "file": filename,
                            "percent": dl_pct,
                            "speed": speed or "",
                        })
    except Exception:
        pass

    proc.wait(timeout=timeout_sec)

    filepath = os.path.join(dest_dir, filename)
    if proc.returncode != 0:
        full_output = "".join(output_lines).strip()
        # aria2c exit 32 = checksum mismatch. Surface as a sha256 mismatch
        # error and remove the corrupt file so retries start clean.
        if proc.returncode == 32 and expected_sha:
            try:
                os.unlink(filepath)
            except OSError:
                pass
            raise RuntimeError(
                f"sha256 mismatch for {filename} (expected {expected_sha}): "
                f"aria2c --checksum verification failed. Corrupt file removed."
            )
        raise RuntimeError(
            f"aria2c download failed (exit {proc.returncode}): {full_output}"
        )

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

    Defensive: if `dest` looks like a full file path (contains `/` and the
    last segment has an extension) AND no explicit `filename` was provided,
    interpret it as `destination_path` and split. Catches a foot-gun seen
    in callers that conflate the two shapes — without this, dest_dir resolves
    to an existing FILE and `os.makedirs(dest_dir, exist_ok=True)` raises
    FileExistsError instead of dedup'ing against the cached file.
    """
    if "destination_path" in dl and dl["destination_path"]:
        return _split_destination_path(dl["destination_path"])
    dest = dl.get("dest", "checkpoints")
    filename = dl.get("filename")
    if not filename and "/" in dest and "." in dest.rsplit("/", 1)[1]:
        return _split_destination_path(dest)
    return dest, filename


def handle(job: dict, progress_callback: Callable[[dict], None] | None = None) -> dict:
    """Handle a download command job.

    `progress_callback`, when supplied, receives structured events instead of
    (and in addition to) the runpod harness's progress_update path — used by
    the installer pod's aiohttp server to bridge into an SSE stream. Event
    shapes: {"type": "download_start"|"download_done"|"download_progress",
    "file_index": int, ...}. When None (default), the legacy harness path is
    used; existing callers behave exactly as before.

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
    # Each call gets a fresh verification queue. Module-global keeps the pool
    # warm across calls but the per-call list must reset (test isolation +
    # robust to mid-job exceptions in earlier handle() invocations).
    _pending_verifications.clear()

    if not downloads:
        raise RuntimeError("No downloads specified. Provide a 'downloads' array.")

    # Set CivitAI token if provided in the job payload
    civitai_token = job_input.get("civitai_token", "")
    if civitai_token:
        os.environ["CIVITAI_TOKEN"] = civitai_token

    # Per-job subprocess timeout. Orchestrator passes `timeout_sec` based on
    # the preset's disk_size_estimate_gb so large downloads aren't capped by
    # an internal 10-minute hardcode. Falls back to 600s for callers
    # (and legacy BlockFlow builds) that don't pass it.
    raw_timeout = job_input.get("timeout_sec")
    subprocess_timeout = max(int(raw_timeout) if raw_timeout else 600, 600)

    print(f"[job {job_id[:8]}] Download command: {len(downloads)} file(s)")
    results = []

    for i, dl in enumerate(downloads):
        source = dl.get("source", "")
        # `huggingface` is a schema alias from blockflow-presets — functionally
        # an aria2c URL fetch, identical to source=`url`. Normalize at entry so
        # the rest of the dispatch (announce, dedup, error msg) stays one path.
        if source == "huggingface":
            source = "url"
        dest, override_filename = _resolve_target(dl)
        dest_dir = os.path.join(MODELS_BASE, dest)
        expected_sha = dl.get("sha256")

        pct = (i / len(downloads)) * 100
        _send_progress(job, f"Downloading {i+1}/{len(downloads)}", percent=pct)
        if progress_callback:
            # `file` resolved best-effort here so the SSE consumer sees the
            # final filename even when civitai derives it post-download.
            announced_name = override_filename
            if source == "url" and not announced_name:
                announced_name = (dl.get("url") or "").rstrip("/").rsplit("/", 1)[-1].split("?")[0]
            progress_callback({
                "type": "download_start",
                "file_index": i,
                "file": announced_name or "",
            })

        if source == "civitai":
            version_id = dl.get("version_id")
            if not version_id:
                raise RuntimeError(f"Download {i+1}: 'version_id' required for civitai source")
            print(f"[job {job_id[:8]}] CivitAI download: version {version_id} -> {dest}")
            info = _download_civitai(
                str(version_id), dest_dir,
                timeout_sec=subprocess_timeout,
                job=job, item_index=i, total_items=len(downloads),
                progress_callback=progress_callback,
                expected_sha=expected_sha,
            )
            cached = bool(info.pop("cached", False))
            # When the dedup path served the file, record the API-reported sha
            # on the result so the post-loop sha-verify branch sees it.
            if cached and "sha256" in info:
                expected_sha = info["sha256"]

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
                    progress_callback=progress_callback,
                    timeout_sec=subprocess_timeout,
                    expected_sha=expected_sha,
                )

        else:
            raise RuntimeError(
                f"Download {i+1}: unknown source '{dl.get('source','')}'. "
                f"Use 'civitai', 'url', or 'huggingface' (alias for 'url').")

        # Post-download sha256 verification — three paths:
        #   (a) cached hit: dedup already proved the sha; record it, no work.
        #   (b) URL/HF download with expected_sha: aria2c verified in-flight
        #       via --checksum=sha-256=..., so a non-zero exit would have
        #       blown up above. No second pass needed.
        #   (c) CivitAI download with expected_sha: the wrapped script doesn't
        #       expose a checksum kwarg, so the file needs a post-hash. We
        #       submit it to a background pool and continue dispatching the
        #       NEXT download immediately. Results awaited at the end of the
        #       loop (bead 8r7 follow-up: async verify).
        if expected_sha and cached:
            info["sha256"] = expected_sha
        elif expected_sha and source == "url":
            # aria2c --checksum already verified. Trust it.
            info["sha256"] = expected_sha.lower()
            print(f"[job {job_id[:8]}] sha256 verified in-flight (aria2c --checksum) for {info['filename']}", flush=True)
        elif expected_sha and source == "civitai":
            size_mb = round(os.path.getsize(info["path"]) / (1024 * 1024), 1)
            print(f"[job {job_id[:8]}] Verifying sha256 of {info['filename']} ({size_mb} MB) in background...", flush=True)
            fut = _verify_pool().submit(
                _async_verify_sha256,
                info["path"], expected_sha.lower(),
                job_tag=job_id[:8], label=info["filename"],
            )
            _pending_verifications.append((i, info, fut, expected_sha.lower()))
            # info["sha256"] is filled in after the future resolves at the
            # end of the loop. For now, mark it pending so callers can see.
            info["sha256_pending"] = True

        info["dest"] = dest
        info["cached"] = cached
        info["bytes"] = os.path.getsize(info["path"])
        results.append(info)
        print(f"[job {job_id[:8]}] Downloaded: {info['filename']} ({info['size_mb']} MB, cached={cached})")
        if progress_callback:
            progress_callback({
                "type": "download_done",
                "file_index": i,
                "file": info["filename"],
                "cached": cached,
                "bytes": info["bytes"],
                "sha256": info.get("sha256"),
            })

    # Drain async sha256 verifications. The CivitAI path submits these to a
    # background pool so the next download can start immediately; we settle the
    # results here so the whole batch succeeds-or-fails atomically.
    if _pending_verifications:
        print(f"[job {job_id[:8]}] Awaiting {len(_pending_verifications)} background sha256 verification(s)...", flush=True)
        verify_wait_started = time.time()
        for idx, info, fut, expected in _pending_verifications:
            actual = fut.result()  # raises on mismatch — propagated to caller
            info["sha256"] = actual
            info.pop("sha256_pending", None)
        verify_wait_elapsed = int(time.time() - verify_wait_started)
        print(f"[job {job_id[:8]}] All sha256 verifications cleared in {verify_wait_elapsed}s", flush=True)
        _pending_verifications.clear()

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
