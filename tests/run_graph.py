"""Execute the ProductLock graph on a running ComfyUI and report the result.

This is the real end-to-end check: the nodes run inside ComfyUI's executor with
ComfyUI's own tensors, not in a bare Python harness.

Usage:
    python tests/run_graph.py [host:port]
"""

import json
import sys
import time
import urllib.error
import urllib.request

HOST = sys.argv[1] if len(sys.argv) > 1 else "127.0.0.1:8207"
BASE = f"http://{HOST}"

PROMPT = {
    "1": {"class_type": "LoadImage", "inputs": {"image": "auric_packshot.png"}},
    "2": {"class_type": "LoadVideo", "inputs": {"file": "auric_generated.mp4"}},
    "3": {"class_type": "GetVideoComponents", "inputs": {"video": ["2", 0]}},
    "4": {"class_type": "ProductLockTrack", "inputs": {
        "reference": ["1", 0], "frames": ["3", 0], "detector": "SIFT",
        "max_features": 4000, "ratio_threshold": 0.75, "ransac_threshold": 3.0,
        "min_inliers": 12, "reject_implausible": True}},
    "5": {"class_type": "ProductLockComposite", "inputs": {
        "reference": ["1", 0], "frames": ["3", 0], "track": ["4", 0],
        "shrink": 1, "feather": 3, "grade_match": 0.6, "grade_blur": 51,
        "on_low_confidence": "passthrough"}},
    "6": {"class_type": "ProductLockQC", "inputs": {
        "reference": ["1", 0], "before": ["3", 0], "after": ["5", 0],
        "track": ["4", 0], "label": True}},
    "7": {"class_type": "CreateVideo", "inputs": {"images": ["5", 0], "fps": ["3", 2]}},
    "8": {"class_type": "SaveVideo", "inputs": {
        "video": ["7", 0], "filename_prefix": "ProductLock/locked",
        # DynamicCombo wire value is the bare option key, not {"key": ...}
        "format": "auto", "codec": "auto"}},
    "9": {"class_type": "SaveImage", "inputs": {
        "images": ["6", 0], "filename_prefix": "ProductLock/comparison"}},
    "10": {"class_type": "SaveImage", "inputs": {
        "images": ["6", 1], "filename_prefix": "ProductLock/drift"}},
}


def post(path, payload):
    req = urllib.request.Request(
        BASE + path, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")
        print(f"HTTP {e.code} from {path}:")
        try:
            print(json.dumps(json.loads(body), indent=2)[:4000])
        except json.JSONDecodeError:
            print(body[:4000])
        raise SystemExit(1)


def get(path):
    with urllib.request.urlopen(BASE + path, timeout=60) as r:
        return json.loads(r.read())


def main():
    result = post("/prompt", {"prompt": PROMPT, "client_id": "productlock-verify"})
    prompt_id = result["prompt_id"]
    print(f"queued {prompt_id}")

    for _ in range(600):
        history = get(f"/history/{prompt_id}")
        if prompt_id in history:
            entry = history[prompt_id]
            status = entry.get("status", {})
            print(f"status: {status.get('status_str')} completed={status.get('completed')}")

            if status.get("status_str") == "error":
                for msg in status.get("messages", []):
                    if msg[0] in ("execution_error", "execution_interrupted"):
                        print(json.dumps(msg[1], indent=2)[:4000])
                raise SystemExit(1)

            for node_id, out in sorted(entry.get("outputs", {}).items(), key=lambda kv: int(kv[0])):
                cls = PROMPT[node_id]["class_type"]
                for key in ("images", "video"):
                    for item in out.get(key, []):
                        name = item.get("filename") if isinstance(item, dict) else item
                        print(f"  node {node_id:>2} {cls:<20} -> {key}: {name}")
            return
        time.sleep(1)

    raise SystemExit("timed out waiting for execution")


if __name__ == "__main__":
    main()
