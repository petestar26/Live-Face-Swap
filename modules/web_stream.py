"""
web_stream.py — view the live face swap on your phone (or any browser) over Wi-Fi.

Phones can't run the swap model themselves (no CUDA, heavy weights), so the desktop does
all the work and *streams the result*. This is a tiny, dependency-free MJPEG-over-HTTP
server (Python stdlib + OpenCV, both already required) that any device on the same network
can open in a browser:

    http://<desktop-lan-ip>:8765/

Usage from the app (already wired behind the "Stream to Phone" toggle):

    import modules.web_stream as web_stream
    web_stream.push(frame_bgr)     # call each processed frame; lazily starts the server
    web_stream.stop()              # optional; daemon thread also dies with the process

Or standalone:

    srv = WebStreamServer(port=8765); srv.start(); print(srv.url())
    srv.update_frame(frame_bgr); ...; srv.stop()
"""

import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional

import cv2
import numpy as np

_BOUNDARY = "frame"


def get_lan_ip() -> str:
    """Best-effort LAN IP of this machine (the address a phone should connect to)."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))  # no packets actually sent; picks the routed iface
        return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"
    finally:
        s.close()


class _FrameBuffer:
    """Holds the most recent JPEG-encoded frame and wakes streaming clients on update."""

    def __init__(self):
        self._cond = threading.Condition()
        self._jpeg: Optional[bytes] = None
        self._seq = 0

    def update(self, jpeg: bytes) -> None:
        with self._cond:
            self._jpeg = jpeg
            self._seq += 1
            self._cond.notify_all()

    def latest(self) -> Optional[bytes]:
        with self._cond:
            return self._jpeg

    def wait_for(self, last_seq: int, timeout: float = 5.0):
        """Block until a frame newer than last_seq arrives; return (jpeg, seq) or (None, seq)."""
        with self._cond:
            if self._seq == last_seq:
                self._cond.wait(timeout)
            return self._jpeg, self._seq


_PAGE = (
    "<!doctype html><html><head><meta charset='utf-8'>"
    "<meta name='viewport' content='width=device-width,initial-scale=1,"
    "maximum-scale=1,user-scalable=no'>"
    "<title>Live Face Swap — by Peter</title>"
    "<style>html,body{margin:0;height:100%;background:#000;overflow:hidden}"
    "img{position:absolute;inset:0;width:100%;height:100%;object-fit:contain}"
    "</style></head><body>"
    "<img src='/stream.mjpg' alt='live' onclick=\"if(document.fullscreenElement)"
    "document.exitFullscreen();else document.documentElement.requestFullscreen();\">"
    "</body></html>"
).encode("utf-8")


def _make_handler(buffer: _FrameBuffer):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):  # silence per-request console spam
            pass

        def _no_store(self):
            self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(_PAGE)))
                self._no_store()
                self.end_headers()
                self.wfile.write(_PAGE)
                return

            if self.path.startswith("/snapshot"):
                jpeg = buffer.latest()
                if jpeg is None:
                    self.send_error(503, "no frame yet")
                    return
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(jpeg)))
                self._no_store()
                self.end_headers()
                self.wfile.write(jpeg)
                return

            if self.path.startswith("/stream.mjpg"):
                self.send_response(200)
                self.send_header(
                    "Content-Type", f"multipart/x-mixed-replace; boundary={_BOUNDARY}"
                )
                self._no_store()
                self.end_headers()
                seq = 0
                try:
                    while True:
                        jpeg, seq = buffer.wait_for(seq, timeout=5.0)
                        if jpeg is None:
                            continue
                        self.wfile.write(b"--" + _BOUNDARY.encode() + b"\r\n")
                        self.wfile.write(b"Content-Type: image/jpeg\r\n")
                        self.wfile.write(
                            b"Content-Length: " + str(len(jpeg)).encode() + b"\r\n\r\n"
                        )
                        self.wfile.write(jpeg)
                        self.wfile.write(b"\r\n")
                except (BrokenPipeError, ConnectionResetError, OSError):
                    return  # client (phone) closed the tab — normal
                return

            self.send_error(404)

    return Handler


class WebStreamServer:
    def __init__(self, host: str = "0.0.0.0", port: int = 8765, jpeg_quality: int = 80):
        self.host = host
        self.port = int(port)
        self.jpeg_quality = int(np.clip(jpeg_quality, 10, 100))
        self._buffer = _FrameBuffer()
        self._httpd: Optional[ThreadingHTTPServer] = None
        self._thread: Optional[threading.Thread] = None

    def start(self) -> "WebStreamServer":
        if self._httpd is not None:
            return self
        self._httpd = ThreadingHTTPServer((self.host, self.port), _make_handler(self._buffer))
        self._httpd.daemon_threads = True
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()
        return self

    def update_frame(self, frame_bgr: np.ndarray) -> None:
        if frame_bgr is None or self._httpd is None:
            return
        ok, buf = cv2.imencode(
            ".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality]
        )
        if ok:
            self._buffer.update(buf.tobytes())

    def url(self) -> str:
        return f"http://{get_lan_ip()}:{self.port}/"

    def stop(self) -> None:
        if self._httpd is not None:
            try:
                self._httpd.shutdown()
                self._httpd.server_close()
            except Exception:
                pass
            self._httpd = None


# ----------------------------------------------------------------- module singleton
_server: Optional[WebStreamServer] = None
_lock = threading.Lock()
_announced = False


def push(frame_bgr: np.ndarray, port: int = 8765, jpeg_quality: int = 80, on_url=None) -> None:
    """Send a frame to the phone stream. Lazily starts the server on first call.

    Fully defensive: any failure is swallowed so the live loop is never disrupted.
    Announces the connect URL exactly once (to the console, and via on_url if given —
    e.g. the app's status bar, so a GUI user doesn't have to watch the terminal).
    """
    global _server, _announced
    try:
        if _server is None:
            with _lock:
                if _server is None:
                    _server = WebStreamServer(port=port, jpeg_quality=jpeg_quality).start()
                    _announced = False
        if not _announced:
            _announced = True
            url = _server.url()
            print(f"[web_stream] Phone stream live — open on your phone: {url}")
            if on_url is not None:
                try:
                    on_url(url)
                except Exception:
                    pass
        _server.update_frame(frame_bgr)
    except Exception as e:  # never let streaming break the swap
        print(f"[web_stream] disabled after error: {e}")
        stop()


def get_url() -> Optional[str]:
    """Connect URL while the stream is running, else None."""
    return _server.url() if _server is not None else None


def stop() -> None:
    global _server, _announced
    with _lock:
        if _server is not None:
            _server.stop()
            _server = None
        _announced = False


def is_running() -> bool:
    return _server is not None
