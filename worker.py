"""RunPod serverless handler for ComfyUI workflows.

Accepts a workflow JSON + input file URLs, executes on local ComfyUI,
uploads outputs (S3 or tmpfiles.org), and returns public URLs.
"""

import hashlib
import json
import os
import shutil
import tempfile
import threading
import time

import struct
import subprocess

import runpod

import comfy_client
import node_installer
import storage


# --- Model hash helpers ---

MODEL_DIRS = [
    "/ComfyUI/models",
    "/runpod-volume/ComfyUI/models",
]


def _resolve_model_path(filename: str) -> str | None:
    """Find a .safetensors file on the volume by walking model dirs."""
    for base in MODEL_DIRS:
        if not os.path.isdir(base):
            continue
        for root, _dirs, files in os.walk(base):
            if filename in files:
                return os.path.join(root, filename)
    return None


def _model_type_from_path(path: str) -> str:
    """Derive model type from the immediate parent folder of the file.

    e.g. /runpod-volume/ComfyUI/models/checkpoints/Wan2.2-I2V-A14B/model.safetensors -> 'Wan2.2-I2V-A14B'
    e.g. /runpod-volume/ComfyUI/models/loras/high/user_lora.safetensors -> 'high'
    """
    return os.path.basename(os.path.dirname(path)) or "unknown"


def _sha256_file(path: str) -> str:
    """Compute SHA256 hash of a file, reading in chunks."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1 << 20)  # 1MB chunks
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _extract_seed(workflow: dict) -> int | None:
    """Extract seed from KSampler node in the workflow."""
    for node in workflow.values():
        if not isinstance(node, dict):
            continue
        if node.get("class_type") in ("KSampler", "KSamplerAdvanced"):
            seed = node.get("inputs", {}).get("seed")
            if isinstance(seed, (int, float)):
                return int(seed)
    return None


def _read_dimensions(path: str) -> dict | None:
    """Read width/height from an image or video file.

    PNG/JPEG: parsed from file headers (no deps).
    Video: parsed via ffprobe.
    """
    VIDEO_EXTS = (".mp4", ".webm", ".avi", ".mov", ".mkv", ".gif")
    if path.lower().endswith(VIDEO_EXTS):
        return _read_video_dimensions(path)
    try:
        with open(path, "rb") as f:
            header = f.read(32)
            # PNG: width/height at bytes 16-24
            if header[:8] == b"\x89PNG\r\n\x1a\n":
                w, h = struct.unpack(">II", header[16:24])
                return {"width": w, "height": h}
            # JPEG: scan for SOF0/SOF2 marker
            f.seek(0)
            data = f.read()
            for marker in (b"\xff\xc0", b"\xff\xc2"):
                idx = data.find(marker)
                if idx != -1:
                    h, w = struct.unpack(">HH", data[idx + 5 : idx + 9])
                    return {"width": w, "height": h}
    except Exception:
        pass
    return None


def _read_video_dimensions(path: str) -> dict | None:
    """Read width/height from a video file via ffprobe."""
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=width,height",
                "-of", "json",
                path,
            ],
            capture_output=True, timeout=10,
        )
        info = json.loads(result.stdout)
        stream = info.get("streams", [{}])[0]
        w, h = stream.get("width"), stream.get("height")
        if w and h:
            return {"width": w, "height": h}
    except Exception:
        pass
    return None


def _strip_metadata(path: str) -> None:
    """Strip all metadata from an output file in-place.

    PNG: removes tEXt, iTXt, zTXt chunks (ComfyUI workflow/prompt data).
    JPEG: removes EXIF/APP markers and COM chunks.
    Video: uses ffmpeg -map_metadata -1 to strip container metadata.
    """
    VIDEO_EXTS = (".mp4", ".webm", ".avi", ".mov", ".mkv")

    if path.lower().endswith(".png"):
        _strip_png_metadata(path)
    elif path.lower().endswith((".jpg", ".jpeg")):
        _strip_jpeg_metadata(path)
    elif path.lower().endswith(VIDEO_EXTS):
        _strip_video_metadata(path)


def _strip_png_metadata(path: str) -> None:
    """Rewrite PNG keeping only image data chunks, dropping text metadata."""
    STRIP_TYPES = {b"tEXt", b"iTXt", b"zTXt"}
    with open(path, "rb") as f:
        sig = f.read(8)
        if sig != b"\x89PNG\r\n\x1a\n":
            return
        chunks = []
        while True:
            raw = f.read(8)
            if len(raw) < 8:
                break
            length = struct.unpack(">I", raw[:4])[0]
            chunk_type = raw[4:8]
            data = f.read(length)
            crc = f.read(4)
            if chunk_type not in STRIP_TYPES:
                chunks.append(raw + data + crc)

    with open(path, "wb") as f:
        f.write(sig)
        for chunk in chunks:
            f.write(chunk)


def _strip_jpeg_metadata(path: str) -> None:
    """Rewrite JPEG keeping only image data, dropping EXIF/APP/COM markers."""
    with open(path, "rb") as f:
        data = f.read()
    if data[:2] != b"\xff\xd8":
        return

    out = bytearray(b"\xff\xd8")
    i = 2
    while i < len(data) - 1:
        if data[i] != 0xFF:
            # Raw image data — copy rest
            out.extend(data[i:])
            break
        marker = data[i + 1]
        # SOS (start of scan) — copy everything from here onward
        if marker == 0xDA:
            out.extend(data[i:])
            break
        # APP0-APP15 (0xE0-0xEF), COM (0xFE) — skip
        if (0xE0 <= marker <= 0xEF) or marker == 0xFE:
            seg_len = struct.unpack(">H", data[i + 2 : i + 4])[0]
            i += 2 + seg_len
            continue
        # Other markers — keep
        if marker in (0xC0, 0xC2, 0xC4, 0xDB, 0xDD):
            seg_len = struct.unpack(">H", data[i + 2 : i + 4])[0]
            out.extend(data[i : i + 2 + seg_len])
            i += 2 + seg_len
        else:
            # Unknown marker with length
            if i + 3 < len(data):
                seg_len = struct.unpack(">H", data[i + 2 : i + 4])[0]
                out.extend(data[i : i + 2 + seg_len])
                i += 2 + seg_len
            else:
                out.extend(data[i:])
                break

    with open(path, "wb") as f:
        f.write(out)


def _strip_video_metadata(path: str) -> None:
    """Strip container metadata from video using ffmpeg."""
    tmp = path + ".stripped"
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", path, "-map_metadata", "-1", "-c", "copy", tmp],
            capture_output=True, timeout=30,
        )
        if os.path.isfile(tmp) and os.path.getsize(tmp) > 0:
            os.replace(tmp, path)
    except Exception:
        pass
    finally:
        if os.path.isfile(tmp):
            os.unlink(tmp)


def _extract_model_refs(workflow: dict) -> list[dict]:
    """Extract model/lora/vae/clip references from the workflow."""
    refs = []
    LOADER_FIELDS = {
        "CheckpointLoaderSimple": [("ckpt_name", {})],
        "UNETLoader": [("unet_name", {})],
        "CLIPLoader": [("clip_name", {})],
        "VAELoader": [("vae_name", {})],
        "LoraLoader": [("lora_name", {"strength_model": "strength_model", "strength_clip": "strength_clip"})],
        "LoraLoaderModelOnly": [("lora_name", {"strength_model": "strength_model"})],
    }

    for _node_id, node in workflow.items():
        if not isinstance(node, dict):
            continue
        class_type = node.get("class_type", "")
        if class_type not in LOADER_FIELDS:
            continue
        inputs = node.get("inputs", {})
        for field_name, extra_fields in LOADER_FIELDS[class_type]:
            filename = inputs.get(field_name, "")
            if not filename or not isinstance(filename, str):
                continue
            ref = {"filename": filename}
            for out_key, in_key in extra_fields.items():
                val = inputs.get(in_key)
                if val is not None:
                    ref[out_key] = val
            refs.append(ref)

    return refs


def _compute_model_hashes(workflow: dict) -> dict:
    """Compute SHA256 hashes for all model files referenced in the workflow.

    Runs synchronously but designed to be called from a background thread.
    """
    refs = _extract_model_refs(workflow)
    result = {}

    for ref in refs:
        filename = ref["filename"]
        if filename in result:
            continue
        path = _resolve_model_path(filename)
        if not path:
            result[filename] = {"error": "not_found"}
            continue

        try:
            sha = _sha256_file(path)
            entry = {
                "sha256": sha,
                "type": _model_type_from_path(path),
            }
            # Include strength for LoRAs (use strength_model value, 1 decimal)
            if "strength_model" in ref:
                entry["strength"] = round(float(ref["strength_model"]), 1)
            result[filename] = entry
        except Exception as e:
            result[filename] = {"error": str(e)}

    return result


def _check_models_exist(workflow: dict) -> list[str]:
    """Check that all model files referenced in the workflow exist on disk.

    Returns list of missing filenames. Empty list means all models found.
    """
    refs = _extract_model_refs(workflow)
    missing = []
    for ref in refs:
        filename = ref["filename"]
        if filename in missing:
            continue
        if _resolve_model_path(filename) is None:
            missing.append(filename)
    return missing


# --- Progress helper ---

def _send_progress(job: dict, stage: str, message: str, percent: float = 0, **extra) -> None:
    """Send a progress update to RunPod. Silently ignores failures."""
    try:
        data = {
            "stage": stage,
            "percent": round(percent, 1),
            "message": message,
        }
        data.update(extra)
        runpod.serverless.progress_update(job, data)
    except Exception:
        pass


def handler(job: dict) -> dict:
    """Process a ComfyUI workflow job.

    Expected input:
    {
        "workflow": { ... },             # ComfyUI API-format workflow
        "file_inputs": {                 # Optional: files to download
            "<node_id>": {
                "field": "image",        # Input field name on the node
                "url": "https://...",    # Pre-signed S3 URL
                "filename": "ref.png"    # Filename for ComfyUI input
            }
        },
        "overrides": {                   # Optional: parameter overrides
            "<node_id>": {"seed": 123, "denoise": 0.5}
        },
        "timeout": 600                   # Optional: max seconds
    }

    Returns:
    {
        "images": [{"url": "...", "size_bytes": ...}],
        "videos": [{"url": "...", "size_bytes": ...}],
        "elapsed_seconds": 123,
        "prompt_id": "abc-123",
        "model_hashes": {"file.safetensors": {"sha256": "...", "type": "checkpoints"}}
    }
    """
    job_input = job["input"]

    # Dispatch non-workflow commands to separate handlers
    command = job_input.get("command")
    if command == "download":
        import download_handler
        return download_handler.handle(job)
    if command == "list_models":
        import list_handler
        return list_handler.handle(job)
    if command == "query_info":
        import info_handler
        return info_handler.handle(job)

    start_time = time.time()
    job_id = job.get("id", "unknown")
    print(f"@@JOB_START {job_id}", flush=True)

    workflow = job_input["workflow"]
    file_inputs = job_input.get("file_inputs", {})
    overrides = job_input.get("overrides", {})
    timeout = job_input.get("timeout", 600)

    # Create temp dirs for this job
    tmp_dir = tempfile.mkdtemp(prefix=f"comfy-job-{job_id[:8]}-")
    input_dir = os.path.join(tmp_dir, "input")
    output_dir = os.path.join(tmp_dir, "output")
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    try:
        _send_progress(job, "init", "Preparing job", percent=0)

        # --- Start model hashing in background ---
        hash_result = {}
        hash_thread = threading.Thread(
            target=lambda: hash_result.update(_compute_model_hashes(workflow)),
            daemon=True,
        )
        hash_thread.start()

        # --- Step 1: Download input files and patch workflow ---
        _send_progress(job, "download_inputs", "Downloading input files", percent=5)
        for node_id, file_info in file_inputs.items():
            url = file_info["url"]
            filename = file_info["filename"]
            field = file_info["field"]
            local_path = os.path.join(input_dir, filename)

            print(f"[job {job_id[:8]}] Downloading input: {filename}")
            storage.download(url, local_path)

            # Upload to ComfyUI's input directory
            comfy_client.upload_input_file(local_path, filename)

            # Patch workflow to reference the uploaded filename
            if node_id in workflow:
                workflow[node_id]["inputs"][field] = filename

        # --- Step 2: Apply parameter overrides ---
        for node_id, params in overrides.items():
            if node_id in workflow:
                workflow[node_id]["inputs"].update(params)

        # --- Step 2b: Check all referenced models exist on disk ---
        missing_models = _check_models_exist(workflow)
        if missing_models:
            names = ", ".join(missing_models)
            raise RuntimeError(
                f"Missing {len(missing_models)} model(s) on network volume: {names}. "
                f"Download them with: comfy-gen download"
            )

        # --- Step 3: Pre-flight check for missing custom nodes ---
        # Uses filesystem scan (no ComfyUI needed) to detect missing nodes,
        # installs them, and restarts ComfyUI if anything was added.
        _send_progress(job, "node_check", "Checking custom nodes", percent=10)

        def _node_progress(msg: str) -> None:
            _send_progress(job, "node_check", msg, percent=11)

        installed = node_installer.ensure_nodes(workflow, progress_fn=_node_progress)
        if installed:
            print(f"[job {job_id[:8]}] Installed custom nodes: {installed}")

        print(f"[job {job_id[:8]}] Queuing prompt...")
        _send_progress(job, "queue", "Queuing prompt", percent=15)
        try:
            prompt_id, client_id = comfy_client.queue_prompt(workflow)
        except RuntimeError as e:
            missing_node = node_installer.parse_missing_node_from_error(str(e))
            if missing_node:
                print(f"[job {job_id[:8]}] ComfyUI rejected: missing {missing_node}. Attempting install...")
                _send_progress(job, "node_check", f"Installing missing node: {missing_node}", percent=11)
                node_installer.ensure_nodes(workflow, max_retries=1, progress_fn=_node_progress)
                prompt_id, client_id = comfy_client.queue_prompt(workflow)
            else:
                raise
        print(f"[job {job_id[:8]}] Prompt ID: {prompt_id}")

        # --- Step 4: Wait for completion with progress ---
        def _node_class(node_id: str) -> str:
            """Look up class_type for a node ID from the workflow."""
            node = workflow.get(node_id, {})
            return node.get("class_type", "") if isinstance(node, dict) else ""

        SAMPLER_TYPES = {"KSampler", "KSamplerAdvanced", "SamplerCustom", "SamplerCustomAdvanced"}

        def on_progress(data):
            stage = data.get("stage", "executing")
            pct_raw = data.get("percent", 0)
            msg = data.get("message", "")
            node_id = data.get("node", "")
            class_type = _node_class(node_id) if node_id else ""
            completed = data.get("completed_nodes", 0)
            total = data.get("total_nodes", 0)

            # For inference steps, only show if it's a sampler node
            if stage == "inference" and class_type not in SAMPLER_TYPES:
                stage = "processing"
                msg = class_type or msg

            elif stage == "inference":
                # Sampler inference: "KSampler Step 1/20"
                step = data.get("step", "")
                total_steps = data.get("total_steps", "")
                step_info = f"Step {step}/{total_steps}" if step and total_steps else msg
                msg = f"{class_type} {step_info}" if class_type else step_info

            elif stage == "executing" and class_type and node_id:
                msg = class_type

            # Map ComfyUI progress (0-100) into our 20-90 range
            pct = 20 + pct_raw * 0.7

            # Pass node progress as structured fields (not baked into message)
            extra = {
                k: v for k, v in data.items()
                if k not in ("stage", "percent", "message", "completed_nodes", "total_nodes")
            }
            if completed > 0 and total > 0:
                extra["completed_nodes"] = completed
                extra["total_nodes"] = total
            if class_type:
                extra["class_type"] = class_type

            _send_progress(job, stage, msg, percent=pct, **extra)
            prefix = f"({completed}/{total}) " if total > 0 and completed > 0 else ""
            print(f"[job {job_id[:8]}] {stage}: {prefix}{msg}", flush=True)

        history = comfy_client.poll_completion(
            prompt_id,
            client_id=client_id,
            timeout=timeout,
            progress_callback=on_progress,
            workflow=workflow,
        )

        # --- Step 5: Collect outputs ---
        _send_progress(job, "collecting", "Collecting outputs", percent=90)
        print(f"[job {job_id[:8]}] History outputs: {json.dumps({k: list(v.keys()) for k, v in history.get('outputs', {}).items()})}")
        results = comfy_client.collect_outputs(history, output_dir)
        print(f"[job {job_id[:8]}] Collected: {len(results['images'])} images, {len(results['videos'])} videos")

        # --- Check for empty outputs / partial execution ---
        if not results["images"] and not results["videos"]:
            # Dump full history for debugging
            status = history.get("status", {})
            status_messages = status.get("messages", [])
            print(f"[job {job_id[:8]}] WARNING: No outputs collected!", flush=True)
            print(f"[job {job_id[:8]}] History status: {json.dumps(status, indent=2)}", flush=True)
            print(f"[job {job_id[:8]}] Full history outputs: {json.dumps(history.get('outputs', {}), indent=2)}", flush=True)

            # Check for execution errors in status messages
            errors = [
                msg for msg in status_messages
                if isinstance(msg, list) and len(msg) >= 1 and msg[0] == "execution_error"
            ]
            if errors:
                err_detail = errors[0][1] if len(errors[0]) > 1 else {}
                node_id = err_detail.get("node_id", "?")
                exc_msg = err_detail.get("exception_message", "unknown")
                exc_type = err_detail.get("exception_type", "")
                traceback_lines = err_detail.get("traceback", [])
                traceback_str = "\n".join(traceback_lines) if traceback_lines else ""
                print(f"[job {job_id[:8]}] Execution error in node {node_id} ({exc_type}): {exc_msg}", flush=True)
                if traceback_str:
                    print(f"[job {job_id[:8]}] Traceback:\n{traceback_str}", flush=True)
                raise RuntimeError(
                    f"Execution error in node {node_id} ({exc_type}): {exc_msg}"
                )

            # No explicit error but no outputs — partial execution or text-only output
            raise RuntimeError(
                f"Workflow produced no image/video outputs. "
                f"History status: {json.dumps(status, indent=2)}"
            )

        # --- Step 6: Strip metadata and upload outputs ---
        _send_progress(job, "upload", "Uploading outputs", percent=92)
        output_images = []
        output_videos = []

        for img in results["images"]:
            _strip_metadata(img["path"])
            img["size_bytes"] = os.path.getsize(img["path"])
            url = storage.upload(img["path"])
            output_images.append({"url": url, "size_bytes": img["size_bytes"]})
            print(f"[job {job_id[:8]}] Uploaded image: {img['filename']} ({img['size_bytes']:,} bytes)")

        for vid in results["videos"]:
            _strip_metadata(vid["path"])
            vid["size_bytes"] = os.path.getsize(vid["path"])
            url = storage.upload(vid["path"])
            output_videos.append({"url": url, "size_bytes": vid["size_bytes"]})
            print(f"[job {job_id[:8]}] Uploaded video: {vid['filename']} ({vid['size_bytes']:,} bytes)")

        # --- Wait for model hashes (should be done by now) ---
        hash_thread.join(timeout=30)

        elapsed = int(time.time() - start_time)

        print(f"@@JOB_END {job_id}", flush=True)

        # --- Build output in standard convention ---
        # Primary output: prefer video, fall back to image
        primary = None
        primary_path = None
        if output_videos:
            primary = output_videos[0]
            primary_path = results["videos"][0]["path"]
        elif output_images:
            primary = output_images[0]
            primary_path = results["images"][0]["path"]

        output = {"url": primary["url"] if primary else None}

        seed = _extract_seed(workflow)
        if seed is not None:
            output["seed"] = seed

        if primary_path:
            dims = _read_dimensions(primary_path)
            if dims:
                output["resolution"] = dims

        if hash_result:
            output["model_hashes"] = hash_result

        return {"ok": True, "output": output}

    except Exception as e:
        elapsed = int(time.time() - start_time)
        print(f"@@JOB_END {job_id}", flush=True)
        error_msg = str(e)

        # Parse ComfyUI-specific errors into clean messages
        if "missing_node_type" in error_msg:
            # Extract the readable part from ComfyUI's error JSON
            try:
                # Find the JSON payload in the error string
                json_start = error_msg.find('{"error"')
                if json_start != -1:
                    comfy_err = json.loads(error_msg[json_start:])
                    node_title = comfy_err["error"]["extra_info"].get("node_title", "")
                    class_type = comfy_err["error"]["extra_info"].get("class_type", "")
                    error_msg = (
                        f"Missing custom node: {node_title or class_type}. "
                        f"The auto-installer could not find or install this node."
                    )
            except (json.JSONDecodeError, KeyError):
                pass

        print(f"[job {job_id[:8]}] FAILED after {elapsed}s: {error_msg}", flush=True)
        return {"ok": False, "error": error_msg, "elapsed_seconds": elapsed}

    finally:
        # --- Cleanup temp files ---
        shutil.rmtree(tmp_dir, ignore_errors=True)


# Load models BEFORE starting the handler — ComfyUI is already running
# (started by start.sh), so this is just registering with RunPod.
runpod.serverless.start({"handler": handler})
