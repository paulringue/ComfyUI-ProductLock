"""Conversions between ComfyUI tensors and OpenCV arrays.

ComfyUI conventions:
    IMAGE -> torch.Tensor [B, H, W, C], float32, 0..1, RGB
    MASK  -> torch.Tensor [B, H, W],    float32, 0..1
OpenCV here works in uint8 RGB (not BGR) so nothing needs re-ordering on the
way in or out; only the greyscale conversions care about channel order.
"""

import numpy as np
import torch


def image_to_numpy(image):
    """IMAGE tensor -> list of uint8 RGB arrays, one per batch item."""
    arr = np.clip(image.detach().cpu().numpy(), 0.0, 1.0)
    return [(a * 255.0 + 0.5).astype(np.uint8) for a in arr]


def numpy_to_image(frames):
    """List of uint8 RGB arrays -> IMAGE tensor."""
    stack = np.stack([f.astype(np.float32) / 255.0 for f in frames], axis=0)
    return torch.from_numpy(stack)


def mask_to_numpy(mask):
    """MASK tensor -> list of uint8 single-channel arrays."""
    arr = np.clip(mask.detach().cpu().numpy(), 0.0, 1.0)
    return [(a * 255.0 + 0.5).astype(np.uint8) for a in arr]


def numpy_to_mask(masks):
    """List of uint8 single-channel arrays -> MASK tensor."""
    stack = np.stack([m.astype(np.float32) / 255.0 for m in masks], axis=0)
    return torch.from_numpy(stack)


def resize_to(arr, shape_hw, nearest=False):
    """Resize only when needed; keeps the common path allocation-free."""
    import cv2

    if arr.shape[:2] == tuple(shape_hw):
        return arr
    interp = cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR
    return cv2.resize(arr, (shape_hw[1], shape_hw[0]), interpolation=interp)
