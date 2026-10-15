"""Geometry safeguards shared by swap and enhancement compositing."""
import cv2
import numpy as np


def crop_support(shape):
    """Fade to zero before the generated crop ends."""
    h, w = shape[:2]
    y, x = np.mgrid[:h, :w]
    distance = np.minimum.reduce((x, y, w - 1 - x, h - 1 - y)).astype(np.float32)
    t = np.clip(distance / max(2.0, min(h, w) * 0.06), 0, 1)
    return t * t * (3 - 2 * t)


def face_outline(face):
    """Use the 106-point contour and five-point pose, not 68-point indices."""
    lm = np.asarray(face.landmark_2d_106, dtype=np.float32)
    kps = np.asarray(face.kps, dtype=np.float32)
    if lm.shape != (106, 2) or kps.shape != (5, 2):
        return None
    if not np.isfinite(lm).all() or not np.isfinite(kps).all():
        return None
    eyes = kps[:2].mean(axis=0)
    up = eyes - kps[3:5].mean(axis=0)
    eye_axis = kps[1] - kps[0]
    if np.linalg.norm(eye_axis) < 2 or np.linalg.norm(up) < 2:
        return None
    top = eyes + up * 0.65
    forehead = np.array([top - eye_axis * 0.65, top, top + eye_axis * 0.65])
    points = np.concatenate((lm[:33], forehead))
    box = np.asarray(face.bbox, dtype=np.float32)
    span = box[2:] - box[:2]
    if not np.isfinite(box).all() or np.any(span <= 0):
        return None
    if np.any(points < box[:2] - span * 0.25) or np.any(points > box[2:] + span * 0.25):
        return None
    return points
