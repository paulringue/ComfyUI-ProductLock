"""Warp the real product back over the generated frames.

Two things stop this looking like a sticker:

1. The alpha comes from the *warped reference silhouette*, eroded then feathered,
   so the seam sits just inside the product edge rather than on it.
2. A low-frequency grade match. We divide a heavily blurred copy of the frame by
   a heavily blurred copy of the warped reference and use that ratio as a gain.
   Low frequencies carry scene lighting; high frequencies carry logo detail. So
   the product picks up the shot's light without any of its own detail moving.
"""

import cv2
import numpy as np

from .conv import image_to_numpy, mask_to_numpy, numpy_to_image, numpy_to_mask, resize_to

LOW_CONFIDENCE_POLICY = ["passthrough", "hold_last_good"]


def _odd(n):
    """OpenCV kernel sizes must be odd and positive."""
    n = int(n)
    return max(1, n | 1)


def _low_frequency_grade(src, dst, blur, strength, max_gain=2.0):
    """Adapt src's lighting towards dst without touching src's detail."""
    k = _odd(blur)
    src_lf = cv2.GaussianBlur(src, (k, k), 0)
    dst_lf = cv2.GaussianBlur(dst, (k, k), 0)
    # +1 keeps the ratio finite in the blacks; clamp keeps a dark frame from
    # crushing the product to nothing or a blown one from clipping it.
    ratio = np.clip((dst_lf + 1.0) / (src_lf + 1.0), 1.0 / max_gain, max_gain)
    return src * (1.0 + (ratio - 1.0) * strength)


class ProductLockComposite:
    """Composite the true product asset over each frame using the tracked homography."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "reference": ("IMAGE",),
                "frames": ("IMAGE",),
                "track": ("PRODUCT_TRACK",),
                "shrink": ("INT", {"default": 1, "min": 0, "max": 64,
                                   "tooltip": "Erode the matte before feathering so the seam sits inside the product edge."}),
                "feather": ("INT", {"default": 3, "min": 0, "max": 64}),
                "grade_match": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 1.0, "step": 0.05,
                                          "tooltip": "0 = paste the packshot flat. 1 = fully adopt the shot's lighting."}),
                "grade_blur": ("INT", {"default": 51, "min": 3, "max": 301,
                                       "tooltip": "Large values keep the grade purely low-frequency."}),
                "on_low_confidence": (LOW_CONFIDENCE_POLICY, {"default": "passthrough"}),
            },
            "optional": {
                "reference_mask": ("MASK",),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "STRING")
    RETURN_NAMES = ("frames", "product_matte", "summary")
    FUNCTION = "composite"
    CATEGORY = "ProductLock"

    def composite(self, reference, frames, track, shrink, feather, grade_match,
                  grade_blur, on_low_confidence, reference_mask=None):
        ref = image_to_numpy(reference)[0]
        frames_np = image_to_numpy(frames)
        homographies = track["homographies"]

        if len(homographies) != len(frames_np):
            raise ValueError(
                f"ProductLockComposite: track covers {len(homographies)} frames "
                f"but {len(frames_np)} frames were supplied. Both inputs must come "
                f"from the same frame batch."
            )

        if reference_mask is not None:
            alpha_ref = resize_to(mask_to_numpy(reference_mask)[0], ref.shape[:2], nearest=True)
        else:
            alpha_ref = np.full(ref.shape[:2], 255, np.uint8)

        out_frames, out_mattes = [], []
        last_good, composited, skipped = None, 0, 0

        for frame, H in zip(frames_np, homographies):
            h, w = frame.shape[:2]

            if H is None and on_low_confidence == "hold_last_good":
                H = last_good
            elif H is not None:
                last_good = H

            if H is None:
                out_frames.append(frame)
                out_mattes.append(np.zeros((h, w), np.uint8))
                skipped += 1
                continue

            warped = cv2.warpPerspective(ref, H, (w, h), flags=cv2.INTER_LINEAR)
            matte = cv2.warpPerspective(alpha_ref, H, (w, h), flags=cv2.INTER_LINEAR)

            if shrink > 0:
                kernel = np.ones((_odd(shrink * 2 + 1),) * 2, np.uint8)
                matte = cv2.erode(matte, kernel)
            if feather > 0:
                matte = cv2.GaussianBlur(matte, (_odd(feather * 2 + 1),) * 2, 0)

            alpha = (matte.astype(np.float32) / 255.0)[..., None]
            src = warped.astype(np.float32)
            if grade_match > 0.0:
                src = _low_frequency_grade(src, frame.astype(np.float32), grade_blur, grade_match)

            blended = frame.astype(np.float32) * (1.0 - alpha) + src * alpha
            out_frames.append(np.clip(blended, 0, 255).astype(np.uint8))
            out_mattes.append(matte)
            composited += 1

        summary = (
            f"Composited {composited}/{len(frames_np)} frames; "
            f"{skipped} left untouched (policy: {on_low_confidence})."
        )
        return (numpy_to_image(out_frames), numpy_to_mask(out_mattes), summary)
