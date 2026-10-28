"""Write demo assets into ComfyUI's input folder.

Produces a packshot PNG and an mp4 standing in for a generative clip, so the
workflow can be exercised end to end without spending API credits. The mp4 also
picks up real codec artefacts, which is a fairer test than clean arrays.

Usage:
    python tests/make_demo_assets.py <comfyui_input_dir>
"""

import os
import sys

import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import test_tracking as tt  # noqa: E402

FPS = 24
SEVERITY = 0.75


def main():
    if len(sys.argv) < 2:
        raise SystemExit("usage: make_demo_assets.py <comfyui_input_dir>")
    out_dir = sys.argv[1]
    os.makedirs(out_dir, exist_ok=True)

    tt.N_FRAMES = 48
    ref = tt.make_packshot()
    frames, _ = tt.build_sequence(ref, SEVERITY)

    packshot_path = os.path.join(out_dir, "auric_packshot.png")
    cv2.imwrite(packshot_path, cv2.cvtColor(ref, cv2.COLOR_RGB2BGR))

    video_path = os.path.join(out_dir, "auric_generated.mp4")
    writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"mp4v"),
                             FPS, (tt.FRAME_W, tt.FRAME_H))
    if not writer.isOpened():
        raise SystemExit("VideoWriter failed to open - no mp4 encoder available")
    for f in frames:
        writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    writer.release()

    print(f"packshot: {packshot_path}  {ref.shape[1]}x{ref.shape[0]}")
    print(f"clip:     {video_path}  {len(frames)} frames @ {FPS}fps")


if __name__ == "__main__":
    main()
