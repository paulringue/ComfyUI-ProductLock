"""Per-frame planar tracking of a reference product against generated frames.

The whole pack rests on one estimate: a homography mapping reference-image
coordinates to frame coordinates. Feature matching gives us that, and the
inlier count gives us a usable confidence signal for free -- when a video model
dissolves the product, the features stop matching and the inlier count falls off
a cliff. That is the signal we flag on, rather than compositing garbage.
"""

import cv2
import numpy as np

from .conv import image_to_numpy, mask_to_numpy, resize_to

DETECTORS = ["SIFT", "ORB"]


def _make_detector(name, max_features):
    """Returns (detector, norm). ORB is binary so it needs Hamming distance."""
    if name == "ORB":
        return cv2.ORB_create(nfeatures=max_features), cv2.NORM_HAMMING
    return cv2.SIFT_create(nfeatures=max_features), cv2.NORM_L2


def _ratio_test(knn_matches, ratio_threshold):
    """Lowe's ratio test - keeps matches that are clearly better than runner-up."""
    good = []
    for pair in knn_matches:
        if len(pair) != 2:
            continue
        best, second = pair
        if best.distance < ratio_threshold * second.distance:
            good.append(best)
    return good


# A healthy inlier count is not enough on its own. RANSAC will happily agree on a
# degenerate homography when the product has melted - measured on the synthetic
# suite, that produced 300px+ corner errors on frames reporting 12+ inliers. So the
# recovered quad gets a geometric sanity check before we trust it.
MIN_AREA_FRACTION = 0.0005   # product smaller than this is off-screen or collapsed
MAX_AREA_FRACTION = 4.0      # ...or blown up far beyond the frame
MIN_EDGE_PIXELS = 8.0
MAX_OPPOSITE_EDGE_RATIO = 8.0


def _quad(H, ref_shape):
    h, w = ref_shape
    corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(corners, H).reshape(-1, 2)


def _signed_area(quad):
    x, y = quad[:, 0], quad[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _is_plausible(H, ref_shape, frame_shape):
    """Reject homographies that are numerically fine but geometrically nonsense."""
    if H is None or not np.isfinite(H).all():
        return False, "non-finite"

    quad = _quad(H, ref_shape)
    if not np.isfinite(quad).all():
        return False, "non-finite corners"

    # Convex, and wound the same way as the source - a sign flip means the
    # product got mirrored, which no real camera move produces.
    edges = np.roll(quad, -1, axis=0) - quad
    cross = np.cross(edges, np.roll(edges, -1, axis=0))
    if not (np.all(cross > 0) or np.all(cross < 0)):
        return False, "non-convex"
    if _signed_area(quad) * _signed_area(np.float32(
            [[0, 0], [ref_shape[1], 0], [ref_shape[1], ref_shape[0]], [0, ref_shape[0]]])) < 0:
        return False, "mirrored"

    frame_area = float(frame_shape[0] * frame_shape[1])
    area = abs(_signed_area(quad))
    if area < MIN_AREA_FRACTION * frame_area:
        return False, "collapsed"
    if area > MAX_AREA_FRACTION * frame_area:
        return False, "exploded"

    lengths = np.linalg.norm(edges, axis=1)
    if lengths.min() < MIN_EDGE_PIXELS:
        return False, "degenerate edge"
    for a, b in ((0, 2), (1, 3)):
        lo, hi = sorted((lengths[a], lengths[b]))
        if lo > 0 and hi / lo > MAX_OPPOSITE_EDGE_RATIO:
            return False, "extreme perspective"

    return True, "ok"


class ProductLockTrack:
    """Estimate where the reference product sits in every frame."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "reference": ("IMAGE",),
                "frames": ("IMAGE",),
                "detector": (DETECTORS, {"default": "SIFT"}),
                "max_features": ("INT", {"default": 4000, "min": 100, "max": 20000}),
                "ratio_threshold": ("FLOAT", {"default": 0.75, "min": 0.50, "max": 0.95, "step": 0.01}),
                "ransac_threshold": ("FLOAT", {"default": 3.0, "min": 0.5, "max": 20.0, "step": 0.5}),
                "min_inliers": ("INT", {"default": 12, "min": 4, "max": 500}),
                "reject_implausible": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Discard homographies whose recovered quad is non-convex, "
                               "mirrored, collapsed or wildly out of scale. RANSAC can report "
                               "healthy inliers on a nonsense transform.",
                }),
            },
            "optional": {
                "reference_mask": ("MASK", {
                    "tooltip": "Restricts feature detection to the product itself. "
                               "Without it, background features in the packshot can "
                               "drag the homography off the product.",
                }),
            },
        }

    RETURN_TYPES = ("PRODUCT_TRACK", "STRING")
    RETURN_NAMES = ("track", "summary")
    FUNCTION = "track"
    CATEGORY = "ProductLock"

    def track(self, reference, frames, detector, max_features, ratio_threshold,
              ransac_threshold, min_inliers, reject_implausible=True, reference_mask=None):
        ref = image_to_numpy(reference)[0]
        ref_gray = cv2.cvtColor(ref, cv2.COLOR_RGB2GRAY)

        ref_mask = None
        if reference_mask is not None:
            m = resize_to(mask_to_numpy(reference_mask)[0], ref_gray.shape[:2], nearest=True)
            ref_mask = ((m > 127).astype(np.uint8)) * 255

        det, norm = _make_detector(detector, max_features)
        kp_ref, desc_ref = det.detectAndCompute(ref_gray, ref_mask)
        if desc_ref is None or len(kp_ref) < 4:
            raise ValueError(
                "ProductLockTrack: fewer than 4 features on the reference image. "
                "A flat or featureless packshot (plain bottle, solid-colour box) "
                "cannot be tracked this way - see the README limitations."
            )

        matcher = cv2.BFMatcher(norm)
        homographies, inliers, matches, reasons = [], [], [], []

        for frame in image_to_numpy(frames):
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            kp, desc = det.detectAndCompute(gray, None)

            H, n_inliers, n_matches, reason = None, 0, 0, "no features"
            if desc is not None and len(kp) >= 4:
                good = _ratio_test(matcher.knnMatch(desc_ref, desc, k=2), ratio_threshold)
                n_matches = len(good)
                if n_matches < 4:
                    reason = "too few matches"
                else:
                    src = np.float32([kp_ref[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
                    dst = np.float32([kp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
                    H_est, inlier_mask = cv2.findHomography(src, dst, cv2.RANSAC, ransac_threshold)
                    if H_est is None:
                        reason = "no homography"
                    else:
                        n_inliers = int(inlier_mask.sum())
                        if n_inliers < min_inliers:
                            reason = "low inliers"
                        elif reject_implausible:
                            ok, why = _is_plausible(H_est, ref_gray.shape[:2], gray.shape[:2])
                            reason = "ok" if ok else why
                            H = H_est if ok else None
                        else:
                            H, reason = H_est, "ok"

            homographies.append(H)
            inliers.append(n_inliers)
            matches.append(n_matches)
            reasons.append(reason)

        track = {
            "homographies": homographies,
            "inliers": inliers,
            "matches": matches,
            "reasons": reasons,
            "min_inliers": min_inliers,
            "ref_shape": tuple(ref_gray.shape[:2]),
            "detector": detector,
        }

        locked = sum(1 for H in homographies if H is not None)
        total = len(homographies)
        rejected = {}
        for r in reasons:
            if r != "ok":
                rejected[r] = rejected.get(r, 0) + 1
        detail = ", ".join(f"{k}: {v}" for k, v in sorted(rejected.items())) or "none rejected"
        summary = (
            f"{detector}: locked {locked}/{total} frames "
            f"({100.0 * locked / max(total, 1):.1f}%). "
            f"Inliers min/median/max = {min(inliers)}/{int(np.median(inliers))}/{max(inliers)}, "
            f"threshold {min_inliers}. Rejected -- {detail}."
        )
        return (track, summary)
