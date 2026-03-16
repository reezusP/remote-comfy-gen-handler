"""Client for talking to ComfyUI running on localhost."""

import json
import os
import threading
import time
import urllib.request
import urllib.parse
import uuid

COMFY_HOST = os.environ.get("COMFY_HOST", "127.0.0.1:8188")
COMFY_URL = f"http://{COMFY_HOST}"


def queue_prompt(workflow: dict) -> tuple[str, str]:
    """Submit a workflow to ComfyUI and return (prompt_id, client_id)."""
    client_id = str(uuid.uuid4())
    payload = json.dumps({"prompt": workflow, "client_id": client_id}).encode()
    req = urllib.request.Request(
        f"{COMFY_URL}/prompt",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    try:
        resp = json.loads(urllib.request.urlopen(req).read())
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")
        # Try to parse ComfyUI's structured error response
        try:
            err_data = json.loads(body)
            raise RuntimeError(json.dumps(err_data)) from e
        except (json.JSONDecodeError, ValueError):
            raise RuntimeError(f"ComfyUI /prompt returned {e.code}: {body[:500]}") from e
    if "error" in resp:
        raise RuntimeError(json.dumps(resp))
    return resp["prompt_id"], client_id


def poll_completion(
    prompt_id: str,
    client_id: str = "",
    timeout: int = 600,
    interval: int = 3,
    progress_callback=None,
    workflow: dict | None = None,
) -> dict:
    """Poll ComfyUI for completion, with optional websocket progress.

    Args:
        prompt_id: The prompt ID to monitor.
        client_id: Client ID used when queuing (enables websocket progress).
        timeout: Max seconds to wait.
        interval: Fallback poll interval if websocket unavailable.
        progress_callback: Optional callable(data_dict) for progress updates.
        workflow: The workflow dict (used to determine total node count and class types).

    Returns the history entry for the prompt.
    """
    # Try websocket-based monitoring first (gives real-time progress)
    if client_id:
        try:
            return _ws_poll_completion(prompt_id, client_id, timeout, progress_callback, workflow)
        except Exception as e:
            print(f"[comfy_client] Websocket poll failed ({e}), falling back to HTTP polling")

    # Fallback: HTTP polling (no per-step progress)
    elapsed = 0
    while elapsed < timeout:
        time.sleep(interval)
        elapsed += interval

        history = _get_history(prompt_id)
        if history is None:
            continue

        status = history.get("status", {})

        for msg in status.get("messages", []):
            if msg[0] == "execution_error":
                err = msg[1] if len(msg) > 1 else {}
                raise RuntimeError(
                    f"Execution error in node {err.get('node_id', '?')}: "
                    f"{err.get('exception_message', 'unknown')}"
                )

        if status.get("completed", False) or status.get("status_str") == "success":
            return history

    raise TimeoutError(f"Workflow timed out after {timeout}s")


def _ws_poll_completion(
    prompt_id: str,
    client_id: str,
    timeout: int,
    progress_callback=None,
    workflow: dict | None = None,
) -> dict:
    """Monitor ComfyUI execution via websocket for real-time progress."""
    import websocket

    ws_url = f"ws://{COMFY_HOST}/ws?clientId={client_id}"
    ws = websocket.create_connection(ws_url, timeout=timeout)

    # Count total workflow nodes (only dict entries with class_type)
    all_node_ids = set()
    if workflow:
        for nid, node in workflow.items():
            if isinstance(node, dict) and "class_type" in node:
                all_node_ids.add(nid)

    try:
        deadline = time.time() + timeout
        current_node = ""
        cached_node_ids: set[str] = set()
        nodes_to_execute = len(all_node_ids)  # Updated after execution_cached
        completed_nodes = 0

        while time.time() < deadline:
            remaining = deadline - time.time()
            if remaining <= 0:
                break
            ws.settimeout(min(remaining, 30))

            try:
                msg = ws.recv()
            except websocket.WebSocketTimeoutException:
                continue

            if isinstance(msg, bytes):
                continue

            try:
                data = json.loads(msg)
            except (json.JSONDecodeError, TypeError):
                continue

            msg_type = data.get("type", "")

            if msg_type == "execution_start":
                if progress_callback:
                    progress_callback({
                        "stage": "executing",
                        "percent": 0,
                        "message": "Execution started",
                    })

            elif msg_type == "execution_cached":
                cached = data.get("data", {}).get("nodes", [])
                cached_node_ids.update(cached)
                nodes_to_execute = len(all_node_ids - cached_node_ids)
                if cached and progress_callback:
                    progress_callback({
                        "stage": "executing",
                        "message": f"{len(cached)} nodes cached",
                    })

            elif msg_type == "executing":
                node_id = data.get("data", {}).get("node")
                if node_id is None:
                    # Execution finished
                    break
                current_node = node_id
                completed_nodes += 1
                if nodes_to_execute > 0 and progress_callback:
                    pct = round(completed_nodes / nodes_to_execute * 100, 1)
                    progress_callback({
                        "stage": "executing",
                        "percent": pct,
                        "node": current_node,
                        "completed_nodes": completed_nodes,
                        "total_nodes": nodes_to_execute,
                        "message": f"Node {completed_nodes}/{nodes_to_execute}",
                    })

            elif msg_type == "progress":
                # Per-step progress within a node (e.g., KSampler steps)
                d = data.get("data", {})
                value = d.get("value", 0)
                max_val = d.get("max", 1)
                if progress_callback:
                    node_pct = round(value / max(max_val, 1) * 100, 1)
                    progress_callback({
                        "stage": "inference",
                        "step": value,
                        "total_steps": max_val,
                        "percent": node_pct,
                        "node": current_node,
                        "completed_nodes": completed_nodes,
                        "total_nodes": nodes_to_execute,
                        "message": f"Step {value}/{max_val}",
                    })

            elif msg_type == "execution_error":
                d = data.get("data", {})
                node_id = d.get("node_id", "?")
                exc_msg = d.get("exception_message", "unknown")
                exc_type = d.get("exception_type", "")
                traceback_lines = d.get("traceback", [])
                traceback_str = "\n".join(traceback_lines) if traceback_lines else ""
                print(f"[comfy_client] Execution error in node {node_id} ({exc_type}): {exc_msg}", flush=True)
                if traceback_str:
                    print(f"[comfy_client] Traceback:\n{traceback_str}", flush=True)
                raise RuntimeError(
                    f"Execution error in node {node_id} ({exc_type}): {exc_msg}"
                )

            elif msg_type == "execution_interrupted":
                d = data.get("data", {})
                print(f"[comfy_client] Execution interrupted: {json.dumps(d)}", flush=True)

            elif msg_type == "status":
                queue = data.get("data", {}).get("status", {}).get("exec_info", {})
                remaining_q = queue.get("queue_remaining", 0)
                if remaining_q == 0 and completed_nodes > 0:
                    break

    finally:
        ws.close()

    # Log if execution completed with fewer nodes than expected
    if nodes_to_execute > 0 and completed_nodes < nodes_to_execute:
        print(
            f"[comfy_client] WARNING: Partial execution — "
            f"{completed_nodes}/{nodes_to_execute} nodes completed",
            flush=True,
        )

    # Fetch final history
    for _ in range(10):
        history = _get_history(prompt_id)
        if history and history.get("status", {}).get("status_str") == "success":
            return history
        time.sleep(1)

    history = _get_history(prompt_id)
    if history:
        return history
    raise TimeoutError(f"Workflow completed but history not found for {prompt_id}")


def collect_outputs(history: dict, output_dir: str) -> dict:
    """Download all output files from a completed workflow.

    Returns {"images": [...], "videos": [...]} with local file paths.
    """
    outputs = history.get("outputs", {})
    images = []
    videos = []

    VIDEO_EXTS = (".mp4", ".webm", ".avi", ".mov", ".mkv", ".gif")
    OUTPUT_KEYS = ("images", "gifs", "videos")

    os.makedirs(output_dir, exist_ok=True)

    for node_id, node_output in outputs.items():
        for key in OUTPUT_KEYS:
            if key not in node_output:
                continue
            for file_info in node_output[key]:
                ftype = file_info.get("type", "output")
                if ftype == "temp":
                    continue

                fname = file_info["filename"]
                subfolder = file_info.get("subfolder", "")
                local_path = os.path.join(output_dir, fname)
                size = _download_output(fname, subfolder, ftype, local_path)

                entry = {"path": local_path, "size_bytes": size, "filename": fname}
                if any(fname.lower().endswith(ext) for ext in VIDEO_EXTS):
                    videos.append(entry)
                else:
                    images.append(entry)

    return {"images": images, "videos": videos}


def upload_input_file(local_path: str, filename: str) -> None:
    """Upload a file to ComfyUI's input directory via the API."""
    with open(local_path, "rb") as f:
        file_data = f.read()

    # ComfyUI upload endpoint
    boundary = uuid.uuid4().hex
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="image"; filename="{filename}"\r\n'
        f"Content-Type: application/octet-stream\r\n\r\n"
    ).encode() + file_data + f"\r\n--{boundary}--\r\n".encode()

    req = urllib.request.Request(
        f"{COMFY_URL}/upload/image",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    urllib.request.urlopen(req)


def _get_history(prompt_id: str) -> dict | None:
    try:
        with urllib.request.urlopen(f"{COMFY_URL}/history/{prompt_id}") as r:
            data = json.loads(r.read())
        return data.get(prompt_id)
    except Exception:
        return None


def _download_output(filename: str, subfolder: str, folder_type: str, save_path: str) -> int:
    query = urllib.parse.urlencode({
        "filename": filename,
        "subfolder": subfolder,
        "type": folder_type,
    })
    with urllib.request.urlopen(f"{COMFY_URL}/view?{query}") as r:
        content = r.read()
        with open(save_path, "wb") as f:
            f.write(content)
        return len(content)
