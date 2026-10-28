"""Measure how accurately ProductLockTrack recovers a KNOWN homography.

Builds a synthetic packshot, moves it through a known camera path over a
background, then degrades each frame the way a video model degrades a product
(soft focus, colour drift, noise, and a local sinusoidal warp that stands in for
"melting"). Tracking never sees the ground truth.

Reported error is mean corner reprojection error in pixels: transform the
reference's four corners by the true homography and by the recovered one, and
measure how far apart they land. Sub-pixel means the composite lands exactly.

Run:
    .venv\\Scripts\\python.exe tests\\test_tracking.py
"""

import os
import sys

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from productlock import ProductLockComposite, ProductLockQC, ProductLockTrack  # noqa: E402
from productlock.conv import numpy_to_image, numpy_to_mask  # noqa: E402

RNG = np.random.default_rng(7)
FRAME_W, FRAME_H = 960, 540
N_FRAMES = 24


def make_packshot(w=300, h=400):
    """A product with enough structure to be trackable: label, text, barcode."""
    img = np.full((h, w, 3), 250, np.uint8)
    cv2.rectangle(img, (0, 0), (w - 1, h - 1), (40, 40, 40), 3)
    cv2.rectangle(img, (18, 24), (w - 18, 150), (196, 74, 44), -1)
    cv2.putText(img, "AURIC", (34, 100), cv2.FONT_HERSHEY_DUPLEX, 1.5, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(img, "COLD BREW", (34, 132), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 235, 220), 1, cv2.LINE_AA)
    cv2.circle(img, (w // 2, 210), 42, (30, 90, 60), 3)
    cv2.putText(img, "100%", (w // 2 - 34, 218), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (30, 90, 60), 2, cv2.LINE_AA)
    # barcode - dense corners, the kind of detail a diffusion model destroys first
    x = 30
    for width in RNG.integers(2, 7, size=34):
        cv2.rectangle(img, (x, 300), (x + int(width), 360), (20, 20, 20), -1)
        x += int(width) + 4
        if x > w - 40:
            break
    cv2.putText(img, "5 012345 678900", (34, 384), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (20, 20, 20), 1, cv2.LINE_AA)
    return img


def make_background():
    """Soft studio-ish gradient with texture, so the scene has its own features."""
    yy, xx = np.mgrid[0:FRAME_H, 0:FRAME_W].astype(np.float32)
    base = 90 + 70 * (1.0 - yy / FRAME_H) + 30 * np.sin(xx / 180.0)
    bg = np.stack([base * 0.85, base * 0.92, base], axis=2)
    bg += RNG.normal(0, 4, bg.shape)
    return np.clip(bg, 0, 255).astype(np.uint8)


def ground_truth_homography(ref_shape, t):
    """Known camera path: the product drifts, rotates, scales and tips in perspective."""
    h, w = ref_shape[:2]
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])

    angle = np.deg2rad(-12 + 26 * t)
    scale = 0.85 + 0.25 * t
    cx, cy = FRAME_W * (0.36 + 0.16 * t), FRAME_H * (0.52 - 0.06 * t)
    tilt = 0.10 * t  # perspective foreshortening on one edge

    corners = []
    for i, (sx, sy) in enumerate(src):
        px, py = (sx - w / 2) * scale, (sy - h / 2) * scale
        rx = px * np.cos(angle) - py * np.sin(angle)
        ry = px * np.sin(angle) + py * np.cos(angle)
        if i in (1, 2):  # right-hand edge recedes
            rx *= 1.0 - tilt
            ry *= 1.0 - tilt * 0.5
        corners.append([cx + rx, cy + ry])
    return cv2.getPerspectiveTransform(src, np.float32(corners))


def degrade(frame, severity):
    """Stand-in for generative drift: soft focus, colour shift, noise, local melt."""
    if severity <= 0:
        return frame
    out = frame.astype(np.float32)

    k = int(severity * 6) * 2 + 1
    if k > 1:
        out = cv2.GaussianBlur(out, (k, k), 0)

    gain = 1.0 + np.float32([0.10, -0.04, -0.09]) * severity
    out = out * gain + 9.0 * severity

    # local sinusoidal warp - the "melting" that breaks a rigid paste
    amp = 3.5 * severity
    yy, xx = np.mgrid[0:out.shape[0], 0:out.shape[1]].astype(np.float32)
    map_x = xx + amp * np.sin(yy / 22.0)
    map_y = yy + amp * np.cos(xx / 26.0)
    out = cv2.remap(out, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    out += RNG.normal(0, 4.0 * severity, out.shape)
    return np.clip(out, 0, 255).astype(np.uint8)


def build_sequence(ref, severity):
    """Returns (degraded frames, ground-truth homographies)."""
    bg = make_background()
    alpha = np.full(ref.shape[:2], 255, np.uint8)
    frames, truths = [], []
    for i in range(N_FRAMES):
        t = i / (N_FRAMES - 1)
        H = ground_truth_homography(ref.shape, t)
        warped = cv2.warpPerspective(ref, H, (FRAME_W, FRAME_H))
        a = (cv2.warpPerspective(alpha, H, (FRAME_W, FRAME_H)).astype(np.float32) / 255.0)[..., None]
        clean = (bg.astype(np.float32) * (1 - a) + warped.astype(np.float32) * a).astype(np.uint8)
        frames.append(degrade(clean, severity))
        truths.append(H)
    return frames, truths


def corner_error(ref_shape, H_true, H_est):
    h, w = ref_shape[:2]
    corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
    a = cv2.perspectiveTransform(corners, H_true).reshape(-1, 2)
    b = cv2.perspectiveTransform(corners, H_est).reshape(-1, 2)
    return float(np.linalg.norm(a - b, axis=1).mean())


def main():
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(out_dir, exist_ok=True)

    ref = make_packshot()
    ref_tensor = numpy_to_image([ref])
    mask_tensor = numpy_to_mask([np.full(ref.shape[:2], 255, np.uint8)])
    tracker = ProductLockTrack()

    header = (f"{'severity':>8} | {'inliers gate only':^30} | {'+ geometry check':^30}\n"
              f"{'':>8} | {'locked':>7} {'median':>7} {'max px':>8}    | "
              f"{'locked':>7} {'median':>7} {'max px':>8}    ")
    print(header)
    print("-" * len(header.split("\n")[1]))

    results = []
    for severity in [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]:
        frames, truths = build_sequence(ref, severity)
        row, keep = f"{severity:>8.2f} |", None
        for reject in (False, True):
            track, summary = tracker.track(
                reference=ref_tensor, frames=numpy_to_image(frames), detector="SIFT",
                max_features=4000, ratio_threshold=0.75, ransac_threshold=3.0,
                min_inliers=12, reject_implausible=reject, reference_mask=mask_tensor,
            )
            errors = [
                corner_error(ref.shape, truths[i], H)
                for i, H in enumerate(track["homographies"]) if H is not None
            ]
            locked = len(errors)
            if errors:
                row += (f" {locked:>3}/{N_FRAMES:<3} {np.median(errors):>7.2f} "
                        f"{max(errors):>8.2f}    |")
            else:
                row += f" {locked:>3}/{N_FRAMES:<3} {'--':>7} {'--':>8}    |"
            if reject:
                keep = track
        print(row.rstrip("|"))
        results.append((severity, frames, keep))

    # Full pipeline render at a realistic severity, for eyeballing.
    severity, frames, track = results[3]
    compositor, qc = ProductLockComposite(), ProductLockQC()
    locked_frames, matte, comp_summary = compositor.composite(
        reference=ref_tensor, frames=numpy_to_image(frames), track=track,
        shrink=1, feather=3, grade_match=0.6, grade_blur=51,
        on_low_confidence="passthrough", reference_mask=mask_tensor,
    )
    comparison, plot, report = qc.qc(
        reference=ref_tensor, before=numpy_to_image(frames), after=locked_frames,
        track=track, label=True, reference_mask=mask_tensor,
    )

    print(f"\nseverity {severity} pipeline\n  {comp_summary}\n  " + report.replace("\n", "\n  "))

    def save(name, tensor, idx=0):
        arr = (np.clip(tensor[idx].numpy(), 0, 1) * 255).astype(np.uint8)
        path = os.path.join(out_dir, name)
        cv2.imwrite(path, cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))
        return path

    for i in (0, N_FRAMES // 2, N_FRAMES - 1):
        save(f"comparison_{i:02d}.png", comparison, i)
    save("drift_plot.png", plot)
    cv2.imwrite(os.path.join(out_dir, "packshot.png"), cv2.cvtColor(ref, cv2.COLOR_RGB2BGR))
    print(f"\nWrote comparison frames, drift plot and packshot to {out_dir}")


if __name__ == "__main__":
    main()
