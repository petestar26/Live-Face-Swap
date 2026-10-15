import cv2
import numpy as np
import time
from typing import Optional, Tuple, Callable
import platform
import threading

# Only import Windows-specific library if on Windows
if platform.system() == "Windows":
    from pygrabber.dshow_graph import FilterGraph


def _index_for_backend(device_name: str, backend: int) -> Optional[int]:
    """Index of `device_name` in a SPECIFIC backend's enumeration, or None.

    DirectShow and Media Foundation number cameras independently: DSHOW may
    list 'DroidCam Source 3' at 2 while MSMF puts something else there. The UI
    picks a camera by its DSHOW index, so falling back to MSMF with that same
    number silently opens a DIFFERENT camera — which is how selecting DroidCam
    ended up showing a frozen frame from a dead device instead of the phone.
    Re-resolving the name per backend keeps the chosen camera the chosen camera.
    """
    if not device_name:
        return None
    try:
        from cv2_enumerate_cameras import enumerate_cameras
        for info in enumerate_cameras(backend):
            if info.name == device_name:
                return info.index
    except Exception:
        pass
    return None


class VideoCapturer:
    def __init__(self, device_index: int):
        self.device_index = device_index
        self.device_name: str = ""
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
            # Remember the NAME so each backend can be given its own correct index.
            self.device_name = devices[self.device_index]

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
                # Virtual cameras (DroidCam, OBS) are Media-Foundation devices:
                # DirectShow either refuses them outright or, worse, half-opens
                # them and leaves the driver wedged so the MSMF attempt that
                # follows only ever returns one stale frame — which is exactly
                # how picking DroidCam produced a frozen picture. They also gain
                # nothing from MJPG negotiation (no USB bandwidth limit), so put
                # MSMF first for them and keep the USB-webcam order otherwise.
                _vname = (self.device_name or "").lower()
                _is_virtual = any(k in _vname for k in ("droidcam", "obs", "virtual"))
                if _is_virtual:
                    attempts = [
                        (cv2.CAP_MSMF, None),
                        (cv2.CAP_DSHOW, None),
                        (cv2.CAP_ANY, None),
                    ]
                self.cap = None
                for backend, params in attempts:
                    cap = None
                    # Translate the DSHOW-based selection into THIS backend's own
                    # index, so a fallback can't land on a different camera.
                    idx = _index_for_backend(self.device_name, backend)
                    if idx is None:
                        if backend not in (cv2.CAP_DSHOW, cv2.CAP_ANY):
                            # Name not present in this backend at all (e.g. OBS
                            # Virtual Camera is DirectShow-only) — skip rather
                            # than open whatever happens to sit at that number.
                            continue
                        idx = self.device_index
                    try:
                        cap = (
                            cv2.VideoCapture(idx, backend, params)
                            if params is not None
                            else cv2.VideoCapture(idx, backend)
                        )
                        if cap.isOpened():
                            # Some virtual cameras "open" but never deliver a
                            # frame with the wrong API — verify a real frame
                            # arrives before committing to this backend.
                            # Pace these probe reads. A virtual camera returns
                            # nothing for the first few reads while it spins up;
                            # hammering read() back-to-back through that window
                            # wedges DroidCam's driver, which then repeats one
                            # frame forever (the "video gets stuck" symptom).
                            # Measured: 8 unpaced reads => frozen; the same reads
                            # spaced out => full motion. Costs at most ~0.4s once,
                            # and only when the first read isn't ready yet.
                            if _is_virtual:
                                # DroidCam/OBS virtual cameras wedge if they are
                                # read too soon after opening: a single early
                                # read makes the driver repeat that one frame
                                # forever (the frozen-preview symptom). Measured
                                # directly - open+read => motion 0.00, open with
                                # no probe => motion 110+. They also never open
                                # on the "wrong" API the way USB webcams do, so
                                # isOpened() is trustworthy here. Let the first
                                # read happen later, in the normal capture loop.
                                self.cap = cap
                            else:
                                for _probe in range(8):
                                    ret, frame = cap.read()
                                    if ret and frame is not None and getattr(frame, "size", 0) > 0:
                                        self.cap = cap
                                        break
                                    time.sleep(0.06)
                            if self.cap is not None:
                                import os as _os
                                if _os.environ.get("KP_CAM_DEBUG"):
                                    _bn = {cv2.CAP_DSHOW: "DSHOW", cv2.CAP_MSMF: "MSMF",
                                           cv2.CAP_ANY: "ANY"}.get(backend, str(backend))
                                    print(f"[VideoCapturer] opened '{self.device_name}' "
                                          f"via {_bn} at index {idx}"
                                          f"{' +MJPG' if params else ''}")
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
            #
            # EXCEPT for virtual cameras (DroidCam / OBS): that burst reads 40
            # frames back-to-back, which wedges their driver — afterwards it
            # keeps handing back the SAME frame forever, so the preview looks
            # frozen even though the device is streaming fine. They also report
            # their frame rate honestly (it is software-defined, not negotiated
            # over USB), so trust the reported value and skip the burst.
            reported_fps = self.cap.get(cv2.CAP_PROP_FPS)
            _lname = (self.device_name or "").lower()
            if any(k in _lname for k in ("droidcam", "obs", "virtual")):
                self.actual_fps = float(reported_fps or fps)
            else:
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
