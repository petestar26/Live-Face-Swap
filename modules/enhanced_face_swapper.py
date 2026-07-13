"""
enhanced_face_swapper.py — high-quality post-processing helpers for face swaps.

This is a *corrected, dependency-light* rework of the commonly-shared
"EnhancedFaceSwapper" snippet. It fixes the bugs in that snippet and adapts it to
this repo, which already runs on Apple Silicon (CoreML) and ships GPEN enhancers:

  * device is auto-detected (cuda -> mps -> cpu) instead of hardcoded to 'cuda';
  * GFPGAN is loaded lazily and uses the correct API (BGR in, 3-tuple out) — it is
    OPTIONAL and absence never crashes the module;
  * CodeFormer's `from codeformer import CodeFormer` is NOT a real pip API, so it is
    not imported by default;
  * the sharpen step uses a proper unsharp mask (blend), not a kernel scaled past
    unity gain (which only brightened the image);
  * temporal smoothing is motion-guarded: averaging a *moving* face produces ghosting,
    so the history is reset when the face shifts, giving anti-flicker without smearing;
  * color/lighting matching reuses this repo's already-validated LAB transfer.

Recommended path for the in-app swap is the built-in `face_color_match` toggle (wired
into modules/processors/frame/face_swapper.swap_face). This module is for scripts /
custom pipelines that want the same quality primitives in one place.
"""

from collections import deque
from typing import Callable, Optional, Tuple

import cv2
import numpy as np

# Reuse the repo's robust LAB mean/std color transfer (handles bad input, div-by-zero).
from modules.processors.frame.face_swapper import apply_color_transfer


def detect_device() -> str:
    """Return the best available torch device string: 'cuda', 'mps', or 'cpu'."""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


class EnhancedFaceSwapper:
    def __init__(
        self,
        device: Optional[str] = None,
        history_length: int = 4,
        sharpen_strength: float = 0.6,
        restore_strength: float = 0.7,
        use_gfpgan: bool = False,
        gfpgan_model_path: str = "models/GFPGANv1.4.pth",
        motion_threshold: float = 0.06,
        restorer: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ):
        """
        device            'cuda'|'mps'|'cpu'|None(auto)
        history_length    frames kept for temporal anti-flicker
        sharpen_strength  unsharp-mask amount, 0..~1.5 (0 disables)
        restore_strength  blend weight of the restored face, 0..1
        use_gfpgan        lazily load GFPGAN for face restoration (optional)
        motion_threshold  fraction of face size; if the face moves more than this
                          between frames the temporal history is reset (no ghosting)
        restorer          optional custom callable(face_bgr)->face_bgr; takes priority
                          over GFPGAN. Use to plug in this repo's GPEN enhancer.
        """
        self.device = device or detect_device()
        self.history: deque = deque(maxlen=max(1, history_length))
        self.sharpen_strength = float(sharpen_strength)
        self.restore_strength = float(np.clip(restore_strength, 0.0, 1.0))
        self.motion_threshold = float(motion_threshold)
        self._last_center: Optional[Tuple[float, float]] = None

        self.restorer = restorer
        self.gfpgan = None
        if use_gfpgan and restorer is None:
            self._init_gfpgan(gfpgan_model_path)

    # ------------------------------------------------------------------ restoration
    def _init_gfpgan(self, model_path: str) -> None:
        """Lazily construct a GFPGANer. Never raises — absence just disables restore."""
        try:
            from gfpgan import GFPGANer
            self.gfpgan = GFPGANer(model_path=model_path, upscale=1, device=self.device)
        except Exception as e:  # pragma: no cover - depends on optional install
            print(f"[EnhancedFaceSwapper] GFPGAN unavailable, restoration disabled: {e}")
            self.gfpgan = None

    def enhance_face(self, face_bgr: np.ndarray) -> np.ndarray:
        """Restore a (BGR uint8) face crop. Returns input unchanged if no restorer."""
        if self.restorer is not None:
            try:
                return self.restorer(face_bgr)
            except Exception as e:
                print(f"[EnhancedFaceSwapper] custom restorer failed: {e}")
                return face_bgr
        if self.gfpgan is not None:
            try:
                # GFPGANer.enhance takes BGR and returns (cropped_faces, restored_faces, restored_img)
                _, _, restored = self.gfpgan.enhance(
                    face_bgr, has_aligned=False, only_center_face=True, paste_back=True
                )
                if restored is not None:
                    if restored.shape[:2] != face_bgr.shape[:2]:
                        restored = cv2.resize(restored, (face_bgr.shape[1], face_bgr.shape[0]))
                    if self.restore_strength >= 1.0:
                        return restored
                    return cv2.addWeighted(
                        face_bgr, 1.0 - self.restore_strength, restored, self.restore_strength, 0
                    )
            except Exception as e:
                print(f"[EnhancedFaceSwapper] GFPGAN enhance failed: {e}")
        return face_bgr

    # ------------------------------------------------------------------ color match
    @staticmethod
    def match_color(source_bgr: np.ndarray, target_bgr: np.ndarray) -> np.ndarray:
        """Match `source` color/lighting to `target` (LAB mean/std transfer)."""
        return apply_color_transfer(source_bgr, target_bgr)

    # ------------------------------------------------------------------ sharpen
    def sharpen(self, img_bgr: np.ndarray) -> np.ndarray:
        """Unsharp mask. Amount 0 -> unchanged; preserves overall brightness."""
        if self.sharpen_strength <= 0.0:
            return img_bgr
        blurred = cv2.GaussianBlur(img_bgr, (0, 0), sigmaX=1.5)
        # out = img + amount*(img - blurred), via addWeighted (unity gain preserved)
        return cv2.addWeighted(img_bgr, 1.0 + self.sharpen_strength, blurred, -self.sharpen_strength, 0)

    # ------------------------------------------------------------------ temporal
    def _temporal_smooth(self, face_bgr: np.ndarray, center: Tuple[float, float],
                         face_size: float) -> np.ndarray:
        """Average recent face crops for anti-flicker, but reset on large motion.

        Averaging a moving face smears it (ghosting). We therefore drop the history
        whenever the face center moves more than `motion_threshold * face_size`, and
        only average while the face is roughly still.
        """
        if self._last_center is not None and face_size > 0:
            moved = np.hypot(center[0] - self._last_center[0], center[1] - self._last_center[1])
            if moved > self.motion_threshold * face_size:
                self.history.clear()
        self._last_center = center

        cur = face_bgr.astype(np.float32)
        # keep history homogeneous in size
        if self.history and self.history[-1].shape != cur.shape:
            self.history.clear()
        self.history.append(cur)
        if len(self.history) == 1:
            return face_bgr
        avg = np.mean(self.history, axis=0)
        return np.clip(avg, 0, 255).astype(np.uint8)

    # ------------------------------------------------------------------ merge
    def merge(
        self,
        swapped_face: np.ndarray,
        original_frame: np.ndarray,
        mask: np.ndarray,
        target_box: Tuple[int, int, int, int],
        do_color_match: bool = True,
        do_restore: bool = True,
        do_sharpen: bool = True,
        do_temporal: bool = True,
    ) -> np.ndarray:
        """Composite `swapped_face` into `original_frame` at `target_box` (x1,y1,x2,y2).

        `mask` is a single-channel (HxW, uint8 or float) face mask in swapped_face space.
        Returns a new frame (original is not modified).
        """
        x1, y1, x2, y2 = (int(v) for v in target_box)
        H, W = original_frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(W, x2), min(H, y2)
        if x2 <= x1 or y2 <= y1:
            return original_frame
        bw, bh = x2 - x1, y2 - y1

        face = cv2.resize(swapped_face, (bw, bh))
        roi = original_frame[y1:y2, x1:x2]

        if do_temporal:
            center = ((x1 + x2) * 0.5, (y1 + y2) * 0.5)
            face = self._temporal_smooth(face, center, float(max(bw, bh)))

        if do_color_match:
            face = self.match_color(face, roi)
        if do_restore:
            face = self.enhance_face(face)
        if do_sharpen:
            face = self.sharpen(face)

        # feathered alpha blend
        m = cv2.resize(mask, (bw, bh)).astype(np.float32)
        if m.max() > 1.0:
            m /= 255.0
        m = cv2.GaussianBlur(m, (0, 0), sigmaX=max(1.0, min(bw, bh) / 40.0))
        m = m[..., None]

        blended = face.astype(np.float32) * m + roi.astype(np.float32) * (1.0 - m)
        result = original_frame.copy()
        result[y1:y2, x1:x2] = np.clip(blended, 0, 255).astype(np.uint8)
        return result
