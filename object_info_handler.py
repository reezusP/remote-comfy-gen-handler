"""Worker-level `object_info` command — surface ComfyUI's /object_info to callers.

Used by the smoke gate (and any client that needs to introspect node classes)
to pre-validate a workflow against the actually-installed node set BEFORE
submitting. Wraps info_handler._get_object_info, optionally filtered.

Returns the raw ComfyUI shape per class:
    {"ok": true, "classes": {"<ClassName>": {"input": {...}, "output": [...], ...}}}

Optional input `class_types: [...]` narrows the response to just those classes
(unknown names are silently dropped). Omit to get every installed class.
"""

from __future__ import annotations

import urllib.error

import info_handler


def handle(job: dict) -> dict:
    """Handle an object_info command.

    Expected input:
    {
        "command": "object_info",
        "class_types": ["KSampler", "VAEDecode", ...]   # optional
    }

    Returns:
    {
        "ok": true,
        "classes": {
            "<ClassName>": {
                "input": {"required": {...}, "optional": {...}},
                "output": [...],
                "output_name": [...],
                ...
            }
        }
    }

    On upstream ComfyUI error, returns {"ok": false, "error": <msg>}.
    """
    try:
        all_classes = info_handler._get_object_info()
    except (urllib.error.URLError, OSError) as e:
        return {"ok": False, "error": str(e)}

    requested = job["input"].get("class_types")
    if requested is None:
        classes = all_classes
    else:
        classes = {name: all_classes[name] for name in requested if name in all_classes}
    return {"ok": True, "classes": classes}
