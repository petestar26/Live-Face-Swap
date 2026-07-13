import cv2
import numpy as np
import time
from typing import Optional, Tuple, Callable
import platform
import threading

# Only import Windows-specific library if on Windows
if platform.system() == "Windows":
    from pygrabber.dshow_graph import FilterGraph


class VideoCapturer:
    def __init__(self, device_index: int):
        self.device_index = device_index
        self.frame_callback = None
        self._current_frame = None
        self._frame_ready = threading.Event()
        self.is_running = False
        self.cap = None
        # Actual values reported by the camera after configuration
        self.actual_width: int = 0
        self.actual_height: int = 0
        self.actual_fps: float = 0.0

        # Initialize Windows-specific components if on Windows
        if platform.system() == "Windows":
            self.graph = FilterGraph()
            # Verify device exists
            devices = self.graph.get_input_devices()
            if self.device_index >= len(devices):
                raise ValueError(
                    f"Invalid device index {device_index}. Available devices: {len(devices)}"
                )

    def start(self, width: int = 960, height: int = 540, fps: int = 60) -> bool:
        """Initialize and start video capture"""
        try:
            if platform.system() == "Windows":
                # device_index comes from pygrabber.FilterGraph (DirectShow
                # enumeration), so open with DSHOW first to preserve mapping.
                # MSMF and DirectShow enumerate cameras in different orders, so
                # opening MSMF with a DSHOW index silently selects the wrong
                # camera. MSMF/ANY remain as fallbacks for cameras DSHOW can't
                # open.
                #
                # Pass codec + resolution + fps as construction params (OpenCV
                # 4.6+). DSHOW locks the pixel format at open time and ignores
                # later cap.set(CAP_PROP_FOURCC, ...) — without this, DSHOW
                # falls back to uncompressed YUYV at 1080p, which is USB-
                # bandwidth-limited to ~5 fps. Setting MJPG at construction
                # negotiates compressed frames from the first read.
                mjpg = cv2.VideoWriter_fourcc(*'MJPG')
                params_mjpg = [
                    cv2.CAP_PROP_FOURCC, mjpg,
                    cv2.CAP_PROP_FRAME_WIDTH, width,
                    cv2.CAP_PROP_FRAME_HEIGHT, height,
                    cv2.CAP_PROP_FPS, fps,
                ]
                # Try DSHOW+MJPG first (best for real USB webcams). Then fall back
                # to PLAIN opens (no forced format): virtual cameras such as
                # "DroidCam Source 3" and "OBS Virtual Camera" REJECT the forced
                # MJPG pixel format and only open via plain DSHOW / MSMF / ANY.
                # Without these fallbacks the app dies with "Failed to start
                # camera" on those cameras.
                attempts = [
                    (cv2.CAP_DSHOW, params_mjpg),
                    (cv2.CAP_DSHOW, None),
                    (cv2.CAP_MSMF, None),
                    (cv2.CAP_ANY, None),
                ]
                self.cap = None
                for backend, params in attempts:
                    cap = None
                    try:
                        cap = (
                            cv2.VideoCapture(self.device_index, backend, params)
                            if params is not None
                            else cv2.VideoCapture(self.device_index, backend)
                        )
                        if cap.isOpened():
                            # Some virtual cameras "open" but never deliver a
                            # frame with the wrong API — verify a real frame
                            # arrives before committing to this backend.
                            for _ in range(8):
                                ret, frame = cap.read()
                                if ret and frame is not None and getattr(frame, "size", 0) > 0:
                                    self.cap = cap
                                    break
                            if self.cap is not None:
                                break
                        cap.release()
                    except Exception:
                        if cap is not None:
                            try:
                                cap.release()
                            except Exception:
                                pass
                        continue
            else:
                # Unix-like systems (Linux/Mac) capture method
                self.cap = cv2.VideoCapture(self.device_index)

            if not self.cap or not self.cap.isOpened():
                raise RuntimeError("Failed to open camera")

            # Keep the driver buffer at 1 so read() always returns the freshest
            # frame instead of a queued older one — cuts the lag between real
            # speech and the swapped video (better lip-sync). Best-effort: some
            # backends ignore it, which is harmless.
            try:
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass

            # Belt-and-braces: also set via cap.set() for backends that honor
            # post-open changes (MSMF, V4L2). DSHOW ignores these, but the
            # construction params above already handled it.
            if platform.system() != "Windows":
                self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                self.cap.set(cv2.CAP_PROP_FPS, fps)

            # Read back resolution (usually reliable)
            self.actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # CAP_PROP_FPS is unreliable on DirectShow — often reports 30
            # even when the camera delivers 60.  Measure empirically by
            # timing a burst of frames.
            reported_fps = self.cap.get(cv2.CAP_PROP_FPS)
            self.actual_fps = self._measure_fps(warmup=10, sample=30,
                                                fallback=reported_fps or fps)

            print(f"[VideoCapturer] {self.actual_width}x{self.actual_height} "
                  f"@ {self.actual_fps:.1f}fps (reported={reported_fps:.0f})",
                  flush=True)

            self.is_running = True
            return True

        except Exception as e:
            print(f"Failed to start capture: {str(e)}")
            if self.cap:
                self.cap.release()
            return False

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read a frame from the camera"""
        if not self.is_running or self.cap is None:
            return False, None

        ret, frame = self.cap.read()
        if ret:
            self._current_frame = frame
            if self.frame_callback:
                self.frame_callback(frame)
            return True, frame
        return False, None

    def release(self) -> None:
        """Stop capture and release resources"""
        if self.is_running and self.cap is not None:
            self.cap.release()
            self.is_running = False
            self.cap = None

    def _measure_fps(self, warmup: int = 10, sample: int = 30,
                     fallback: float = 30.0) -> float:
        """Read warmup+sample frames and return measured FPS.

        This is more reliable than CAP_PROP_FPS which often lies on
        DirectShow.  Takes ~0.5-1s at startup but gives a ground-truth
        number for adaptive polling/detection intervals.
        """
        try:
            for _ in range(warmup):
                self.cap.read()
            t0 = time.perf_counter()
            for _ in range(sample):
                ret, _ = self.cap.read()
                if not ret:
                    return fallback
            elapsed = time.perf_counter() - t0
            if elapsed <= 0:
                return fallback
            return sample / elapsed
        except Exception:
            return fallback

    def set_frame_callback(self, callback: Callable[[np.ndarray], None]) -> None:
        """Set callback for frame processing"""
        self.frame_callback = callback
