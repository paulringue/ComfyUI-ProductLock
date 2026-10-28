"""ComfyUI-ProductLock - keep the client's product intact through generative video."""

from .productlock import ProductLockComposite, ProductLockQC, ProductLockTrack

NODE_CLASS_MAPPINGS = {
    "ProductLockTrack": ProductLockTrack,
    "ProductLockComposite": ProductLockComposite,
    "ProductLockQC": ProductLockQC,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ProductLockTrack": "Product Lock - Track",
    "ProductLockComposite": "Product Lock - Composite",
    "ProductLockQC": "Product Lock - QC",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
