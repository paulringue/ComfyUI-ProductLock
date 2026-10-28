"""Quality control: show the fix, and measure the problem it fixed.

The honest metric here is *product deviation on the generated frames* - warp
each generated frame back into reference space and compare it to the real
packshot inside the product matte. That number quantifies how far the model
drifted, which is the thing being corrected.

Deliberately not measured: fidelity of the composited output. That would be
circular - the output contains the reference pixels by construction, so any
such score would be near-perfect and would prove nothing.
"""

import cv2
import numpy as np

from .conv import image_to_numpy, mask_to_numpy, numpy_to_image, resize_to


def _deviation(frame, ref, H, ref_mask):
    """Mean abs difference (0-255) between the frame's product and the real one."""
    h, w = ref.shape[:2]
    try:
        H_inv = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        return None
    back = cv2.warpPerspective(frame, H_inv, (w, h), flags=cv2.INTER_LINEAR)
    valid = ref_mask > 127
    if not valid.any():
        return None
    diff = np.abs(back.astype(np.float32) - ref.astype(np.float32)).mean(axis=2)
    return float(diff[valid].mean())


def _plot(values, flags, width, height, title):
    """Small dependency-free line plot drawn straight into a numpy array."""
    img = np.full((height, width, 3), 24, np.uint8)
    if not values:
        return img

    pad = 34
    plot_w, plot_h = width - 2 * pad, height - 2 * pad
    finite = [v for v in values if v is not None]
    vmax = max(finite) if finite else 1.0
    vmax = max(vmax, 1e-6)

    # axes
    cv2.rectangle(img, (pad, pad), (pad + plot_w, pad + plot_h), (70, 70, 70), 1)

    n = len(values)
    step = plot_w / max(n - 1, 1)
    points = []
    for i, v in enumerate(values):
        if v is None:
            points.append(None)
            continue
        x = int(pad + i * step)
        y = int(pad + plot_h - (v / vmax) * plot_h)
        points.append((x, y))

    for i in range(1, n):
        if points[i - 1] and points[i]:
            cv2.line(img, points[i - 1], points[i], (120, 200, 255), 2, cv2.LINE_AA)

    # mark frames where tracking failed
    for i, failed in enumerate(flags):
        if failed:
            x = int(pad + i * step)
            cv2.line(img, (x, pad), (x, pad + plot_h), (60, 80, 200), 1)

    cv2.putText(img, title, (pad, pad - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA)
    cv2.putText(img, f"{vmax:.1f}", (4, pad + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160, 160, 160), 1, cv2.LINE_AA)
    cv2.putText(img, "0", (4, pad + plot_h + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160, 160, 160), 1, cv2.LINE_AA)
    return img


class ProductLockQC:
    """Before/after strip, a drift plot, and a numeric report."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "reference": ("IMAGE",),
                "before": ("IMAGE",),
                "after": ("IMAGE",),
                "track": ("PRODUCT_TRACK",),
                "label": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "reference_mask": ("MASK",),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "STRING")
    RETURN_NAMES = ("comparison", "drift_plot", "report")
    FUNCTION = "qc"
    CATEGORY = "ProductLock"

    def qc(self, reference, before, after, track, label, reference_mask=None):
        ref = image_to_numpy(reference)[0]
        before_np = image_to_numpy(before)
        after_np = image_to_numpy(after)
        homographies = track["homographies"]

        if reference_mask is not None:
            ref_mask = resize_to(mask_to_numpy(reference_mask)[0], ref.shape[:2], nearest=True)
        else:
            ref_mask = np.full(ref.shape[:2], 255, np.uint8)

        comparisons, deviations, failed = [], [], []
        for i, (b, a) in enumerate(zip(before_np, after_np)):
            H = homographies[i] if i < len(homographies) else None
            failed.append(H is None)
            deviations.append(_deviation(b, ref, H, ref_mask) if H is not None else None)

            a_fit = resize_to(a, b.shape[:2])
            pair = np.concatenate([b, a_fit], axis=1)
            if label:
                cv2.putText(pair, "generated", (12, 28), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(pair, "product-locked", (b.shape[1] + 12, 28), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (255, 255, 255), 2, cv2.LINE_AA)
                if H is None:
                    cv2.putText(pair, "TRACK LOST", (12, 58), cv2.FONT_HERSHEY_SIMPLEX,
                                0.7, (80, 80, 255), 2, cv2.LINE_AA)
            comparisons.append(pair)

        plot = _plot(deviations, failed, 720, 260,
                     "product deviation of generated frames (mean abs error, 0-255)")

        measured = [d for d in deviations if d is not None]
        n = len(before_np)
        if measured:
            report = (
                f"Frames: {n}. Tracked: {n - sum(failed)}. Lost: {sum(failed)}.\n"
                f"Product deviation on generated frames (mean abs error 0-255): "
                f"min {min(measured):.1f}, median {np.median(measured):.1f}, max {max(measured):.1f}.\n"
                f"Deviation is measured on the INPUT frames - it quantifies the drift being "
                f"corrected, not the quality of the output."
            )
        else:
            report = f"Frames: {n}. Tracking failed on every frame - nothing to measure."

        return (numpy_to_image(comparisons), numpy_to_image([plot]), report)
