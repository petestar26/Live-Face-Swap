"""
web_stream.py — view the live face swap on your phone (or any browser) over Wi-Fi.

Phones can't run the swap model themselves (no CUDA, heavy weights), so the desktop does
all the work and *streams the result*. This is a tiny, dependency-free MJPEG-over-HTTP
server (Python stdlib + OpenCV, both already required) that any device on the same network
can open in a browser:

    http://<desktop-lan-ip>:8765/?k=<access-key>

ACCESS CONTROL (host-it-yourself model)
---------------------------------------
The stream is protected by a secret access key so only people you send the exact
link to can watch — the code and models never leave your machine; the viewer only
sees the output. Behaviour:

    * By default a random key is generated each launch and printed with the URL.
    * Set env  KP_STREAM_TOKEN=mysecret  to use a fixed key (stable link across restarts).
    * Set env  KP_STREAM_NOAUTH=1        to disable the key (old open behaviour, LAN only).

Usage from the app (already wired behind the "Stream to Phone" toggle):

    import modules.web_stream as web_stream
    web_stream.push(frame_bgr)     # call each processed frame; lazily starts the server
    web_stream.stop()              # optional; daemon thread also dies with the process

Or standalone:

    srv = WebStreamServer(port=8765); srv.start(); print(srv.url())
    srv.update_frame(frame_bgr); ...; srv.stop()
"""

import os
import secrets
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional
from urllib.parse import urlparse, parse_qs

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


def _resolve_token(token: Optional[str]) -> str:
    """Decide the access key: explicit arg > env > random. '' means no auth."""
    if os.environ.get("KP_STREAM_NOAUTH", "").strip() in ("1", "true", "yes", "on"):
        return ""
    if token:
        return token
    env = os.environ.get("KP_STREAM_TOKEN", "").strip()
    if env:
        return env
    return secrets.token_urlsafe(9)  # ~12-char unguessable key, fresh each launch


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


def _build_page(token: str) -> bytes:
    """Landing page; embeds the access key into the stream URL when auth is on."""
    q = f"?k={token}" if token else ""
    return (
        "<!doctype html><html><head><meta charset='utf-8'>"
        # viewport-fit=cover lets the page reach under a notch / rounded corners.
        "<meta name='viewport' content='width=device-width,initial-scale=1,"
        "maximum-scale=1,user-scalable=no,viewport-fit=cover'>"
        "<meta name='mobile-web-app-capable' content='yes'>"
        "<meta name='apple-mobile-web-app-status-bar-style' content='black-translucent'>"
        "<title>Live Face Swap — by Peter</title>"
        "<style>"
        "*{margin:0;padding:0;box-sizing:border-box}"
        # position:fixed + inset:0 pins the page to the VISIBLE viewport. height:100%
        # alone is measured against the layout viewport, which on phones is taller
        # than the screen while the address bar is showing — that is what made the
        # page scroll. overscroll-behavior/touch-action kill the rubber-band drag.
        "html,body{position:fixed;inset:0;width:100%;height:100%;background:#000;"
        "overflow:hidden;overscroll-behavior:none;touch-action:none}"
        "#v{position:fixed;inset:0;width:100%;height:100%;object-fit:contain;"
        "display:block}"
        # 'fill' = object-fit:cover -> video covers the whole screen (crops overflow).
        "body.fill #v{object-fit:cover}"
        "#fs{position:fixed;top:calc(env(safe-area-inset-top,0px) + 10px);"
        "right:calc(env(safe-area-inset-right,0px) + 10px);width:40px;height:40px;"
        "border:0;border-radius:50%;background:rgba(0,0,0,.45);color:#fff;"
        "font-size:19px;line-height:40px;text-align:center;cursor:pointer;"
        "opacity:.55;z-index:2}"
        "</style></head><body class='fill'>"
        f"<img id='v' src='/stream.mjpg{q}' alt='live'>"
        "<button id='fs' title='Fullscreen'>&#9974;</button>"
        "<script>"
        "var b=document.body,v=document.getElementById('v'),f=document.getElementById('fs');"
        # Tap the picture: switch between filling the screen and showing everything.
        "v.addEventListener('click',function(){b.classList.toggle('fill');});"
        # Button: real browser fullscreen (hides the address bar entirely).
        "f.addEventListener('click',function(e){e.stopPropagation();"
        "var d=document.documentElement;"
        "if(document.fullscreenElement||document.webkitFullscreenElement){"
        "(document.exitFullscreen||document.webkitExitFullscreen).call(document);}"
        "else{(d.requestFullscreen||d.webkitRequestFullscreen||"
        "d.webkitRequestFullScreen).call(d);}});"
        "</script>"
        "</body></html>"
    ).encode("utf-8")


_DENIED = (
    b"<!doctype html><html><head><meta charset='utf-8'><title>Access denied</title>"
    b"<style>body{font-family:sans-serif;background:#111;color:#ddd;text-align:center;"
    b"padding-top:20%}</style></head><body><h2>Access denied</h2>"
    b"<p>You need the private link with its access key.</p></body></html>"
)


def _make_handler(buffer: _FrameBuffer, token: str):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):  # silence per-request console spam
            pass

        def _no_store(self):
            self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")

        def _authorized(self) -> bool:
            if not token:
                return True
            supplied = (parse_qs(urlparse(self.path).query).get("k") or [""])[0]
            return bool(supplied) and secrets.compare_digest(supplied, token)

        def _deny(self):
            self.send_response(403)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(_DENIED)))
            self._no_store()
            self.end_headers()
            self.wfile.write(_DENIED)

        def do_GET(self):
            path = urlparse(self.path).path

            if not self._authorized():
                self._deny()
                return

            if path in ("/", "/index.html"):
                page = _build_page(token)
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(page)))
                self._no_store()
                self.end_headers()
                self.wfile.write(page)
                return

            if path.startswith("/snapshot"):
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

            if path.startswith("/stream.mjpg"):
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
    def __init__(self, host: str = "0.0.0.0", port: int = 8765, jpeg_quality: int = 80,
                 token: Optional[str] = None):
        self.host = host
        self.port = int(port)
        self.jpeg_quality = int(np.clip(jpeg_quality, 10, 100))
        self.token = _resolve_token(token)
        self._buffer = _FrameBuffer()
        self._httpd: Optional[ThreadingHTTPServer] = None
        self._thread: Optional[threading.Thread] = None

    def start(self) -> "WebStreamServer":
        if self._httpd is not None:
            return self
        self._httpd = ThreadingHTTPServer(
            (self.host, self.port), _make_handler(self._buffer, self.token)
        )
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
        base = f"http://{get_lan_ip()}:{self.port}/"
        return base + (f"?k={self.token}" if self.token else "")

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
            print(f"[web_stream] Private stream live — share ONLY this link: {url}")
            if _server.token:
                print("[web_stream] (the ?k=... part is the access key; without it viewers are denied)")
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
    """Connect URL (with access key) while the stream is running, else None."""
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
