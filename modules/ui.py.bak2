"""PySide6 UI for Deep-Live-Cam.

Public API kept stable for the rest of the codebase:
    init(start, destroy, lang) -> _Window
        Returned object has .mainloop() that core.py calls.
    update_status(text)
        Thread-safe; routed through Qt signal when called off-UI.
    check_and_ignore_nsfw(target, destroy=None) -> bool
"""

from __future__ import annotations

import os
import platform
import queue
import sys
import tempfile
import threading
import time
import webbrowser
from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np
import requests
from PIL import Image, ImageOps
from PySide6.QtCore import (
    QObject,
    QThread,
    QTimer,
    Qt,
    Signal,
)
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QVBoxLayout,
    QWidget,
)

import modules.globals
import modules.metadata
from modules.capturer import get_video_frame, get_video_frame_total
from modules.face_analyser import (
    add_blank_map,
    detect_many_faces_fast,
    detect_one_face_fast,
    ensure_landmarks,
    get_one_face,
    get_unique_faces_from_target_image,
    get_unique_faces_from_target_video,
    has_valid_map,
    simplify_maps,
)
from modules.gettext import LanguageManager
from modules.gpu_processing import gpu_cvt_color, gpu_flip, gpu_resize
from modules.processors.frame.core import get_frame_processors_modules
from modules.utilities import (
    has_image_extension,
    is_image,
    is_video,
)
from modules import imread_unicode
from modules.video_capture import VideoCapturer

if platform.system() == "Windows":
    from pygrabber.dshow_graph import FilterGraph

import json


# ─── constants ────────────────────────────────────────────────────────────

ROOT_HEIGHT = 820
ROOT_WIDTH = 640

PREVIEW_MAX_HEIGHT = 700
PREVIEW_MAX_WIDTH = 1200
PREVIEW_DEFAULT_WIDTH = 640
PREVIEW_DEFAULT_HEIGHT = 360

POPUP_WIDTH = 750
POPUP_HEIGHT = 810
POPUP_SCROLL_WIDTH = 720
POPUP_SCROLL_HEIGHT = 700

POPUP_LIVE_WIDTH = 900
POPUP_LIVE_HEIGHT = 820
POPUP_LIVE_SCROLL_WIDTH = 870
POPUP_LIVE_SCROLL_HEIGHT = 700

MAPPER_PREVIEW_SIZE = 100
SOURCE_TARGET_PREVIEW_SIZE = 200


# ─── modern dark stylesheet ───────────────────────────────────────────────

QSS = """
QMainWindow, QDialog {
    background-color: #16181d;
    color: #e8eaed;
}
QWidget {
    color: #e8eaed;
    font-family: "Segoe UI", "SF Pro Display", "Helvetica Neue", Arial, sans-serif;
    font-size: 11pt;
}

QGroupBox {
    background-color: #20232a;
    border: 1px solid #2c2f37;
    border-radius: 12px;
    margin-top: 16px;
    padding: 20px 14px 12px 14px;
    font-weight: 600;
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 12px;
    padding: 2px 10px;
    color: #8fb6ff;
    font-size: 10pt;
    letter-spacing: 1px;
    text-transform: uppercase;
}

QPushButton {
    background-color: #2d6cdf;
    color: white;
    border: none;
    border-radius: 9px;
    padding: 9px 16px;
    font-weight: 600;
}
QPushButton:hover  { background-color: #3f80f5; }
QPushButton:pressed{ background-color: #1d57c2; }
QPushButton:disabled { background-color: #383b42; color: #7a7d85; }
QPushButton#secondary {
    background-color: #2b2f37;
    border: 1px solid #3a3f49;
}
QPushButton#secondary:hover { background-color: #353a44; border-color: #4a5160; }
QPushButton#danger { background-color: #c2412d; }
QPushButton#danger:hover  { background-color: #d8523c; }
QPushButton#success {
    background-color: #1f9d57;
    font-size: 12pt;
    padding: 9px 22px;
}
QPushButton#success:hover  { background-color: #25b365; }
QPushButton#success:pressed{ background-color: #178045; }
QPushButton#success:disabled { background-color: #383b42; color: #7a7d85; }

QComboBox {
    background-color: #23262d;
    border: 1px solid #383c45;
    border-radius: 8px;
    padding: 7px 10px;
    min-height: 24px;
}
QComboBox:hover { border-color: #2d6cdf; }
QComboBox::drop-down { border: none; width: 22px; }
QComboBox QAbstractItemView {
    background-color: #23262d;
    selection-background-color: #2d6cdf;
    border: 1px solid #383c45;
    border-radius: 6px;
    padding: 4px;
    outline: none;
}

QCheckBox {
    spacing: 8px;
    padding: 4px 0;
}
QCheckBox::indicator {
    width: 38px; height: 20px;
    border-radius: 10px;
    background-color: #383c45;
}
QCheckBox::indicator:checked {
    background-color: #2d6cdf;
}

QSlider::groove:horizontal {
    height: 6px;
    background: #383c45;
    border-radius: 3px;
}
QSlider::handle:horizontal {
    background: #ffffff;
    width: 16px; height: 16px;
    margin: -6px 0;
    border-radius: 8px;
    border: 1px solid #cccccc;
}
QSlider::sub-page:horizontal {
    background: #2d6cdf;
    border-radius: 3px;
}

QLabel#imageDrop {
    background-color: #23262d;
    border: 2px dashed #3a3f49;
    border-radius: 12px;
    color: #7a7d85;
}
QLabel#sourceThumb {
    border: 2px solid #2d6cdf;
    border-radius: 6px;
}
QLabel#sourceThumb:hover {
    border-color: #e25141;
}
QLabel#hintLabel {
    color: #8a8f99;
    font-size: 9pt;
}
QLabel#statusLabel {
    color: #c2c5cc;
    font-size: 10pt;
    font-style: italic;
}
QLabel#linkLabel {
    color: #6ea8ff;
    text-decoration: underline;
}

QScrollArea { border: none; background: transparent; }

QFrame#card {
    background-color: #20232a;
    border-radius: 12px;
}
"""


# ─── module-level state ───────────────────────────────────────────────────

_APP: Optional[QApplication] = None
_MAIN: Optional["MainWindow"] = None
_PREVIEW: Optional["PreviewWindow"] = None
_WEBCAM_PREVIEW: Optional["WebcamPreviewWindow"] = None
_MAPPER: Optional["MapperDialog"] = None
_LIVE_MAPPER: Optional["LiveMapperDialog"] = None
_LANG: Optional[LanguageManager] = None
_BRIDGE: Optional["_UIBridge"] = None


def _(text: str) -> str:
    """Translate via LanguageManager; falls back to identity."""
    if _LANG is None:
        return text
    return _LANG._(text)


# Preserve original cwd state for file dialogs.
_RECENT_SOURCE_DIR: Optional[str] = None
_RECENT_TARGET_DIR: Optional[str] = None
_RECENT_OUTPUT_DIR: Optional[str] = None


# ─── image utilities ─────────────────────────────────────────────────────


def fit_image_to_size(image, width: int, height: int):
    """BGR ndarray → BGR ndarray scaled to fit within (width, height)."""
    if width is None and height is None or width <= 0 or height <= 0:
        return image
    h, w = image.shape[:2]
    ratio_w = width / w
    ratio_h = height / h
    ratio = min(ratio_w, ratio_h)
    new_size = (max(1, int(w * ratio)), max(1, int(h * ratio)))
    return gpu_resize(image, dsize=new_size)


def _bgr_to_qpixmap(bgr: np.ndarray) -> QPixmap:
    """Zero-copy BGR ndarray → QPixmap."""
    h, w = bgr.shape[:2]
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    qimg = QImage(rgb.data, w, h, w * 3, QImage.Format.Format_RGB888).copy()
    return QPixmap.fromImage(qimg)


def _pil_to_qpixmap(image: Image.Image) -> QPixmap:
    """PIL.Image → QPixmap."""
    image = image.convert("RGBA")
    data = image.tobytes("raw", "RGBA")
    qimg = QImage(data, image.width, image.height, QImage.Format.Format_RGBA8888)
    return QPixmap.fromImage(qimg.copy())


def render_image_preview(image_path: str, size: Tuple[int, int]) -> QPixmap:
    image = Image.open(image_path)
    if size:
        image = ImageOps.fit(image, size, Image.LANCZOS)
    return _pil_to_qpixmap(image)


def render_video_preview(
    video_path: str, size: Tuple[int, int], frame_number: int = 0
) -> Optional[QPixmap]:
    capture = cv2.VideoCapture(video_path)
    try:
        if frame_number:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        has_frame, frame = capture.read()
        if not has_frame:
            return None
        image = Image.fromarray(gpu_cvt_color(frame, cv2.COLOR_BGR2RGB))
        if size:
            image = ImageOps.fit(image, size, Image.LANCZOS)
        return _pil_to_qpixmap(image)
    finally:
        capture.release()


# ─── persistence ─────────────────────────────────────────────────────────


def save_switch_states():
    state = {
        "keep_fps": modules.globals.keep_fps,
        "keep_audio": modules.globals.keep_audio,
        "keep_frames": modules.globals.keep_frames,
        "many_faces": modules.globals.many_faces,
        "map_faces": modules.globals.map_faces,
        "poisson_blend": modules.globals.poisson_blend,
        "adaptive_face_mask": modules.globals.adaptive_face_mask,
        "color_correction": modules.globals.color_correction,
        "face_color_match": modules.globals.face_color_match,
        "source_skin_tone": getattr(modules.globals, "source_skin_tone", False),
        "web_stream": modules.globals.web_stream,
        "nsfw_filter": modules.globals.nsfw_filter,
        "live_mirror": modules.globals.live_mirror,
        "live_resizable": modules.globals.live_resizable,
        "fp_ui": modules.globals.fp_ui,
        "show_fps": modules.globals.show_fps,
        "mouth_mask": modules.globals.mouth_mask,
        "show_mouth_mask_box": modules.globals.show_mouth_mask_box,
        "mouth_mask_size": modules.globals.mouth_mask_size,
        "teeth_effect": getattr(modules.globals, "teeth_effect", "None"),
        "teeth_image_path": getattr(modules.globals, "teeth_image_path", ""),
    }
    try:
        with open("switch_states.json", "w") as f:
            json.dump(state, f)
    except OSError:
        pass


def load_switch_states():
    try:
        with open("switch_states.json", "r") as f:
            state = json.load(f)
        modules.globals.keep_fps = state.get("keep_fps", True)
        modules.globals.keep_audio = state.get("keep_audio", True)
        modules.globals.keep_frames = state.get("keep_frames", False)
        modules.globals.many_faces = state.get("many_faces", False)
        modules.globals.map_faces = state.get("map_faces", False)
        modules.globals.poisson_blend = state.get("poisson_blend", False)
        modules.globals.adaptive_face_mask = state.get("adaptive_face_mask", True)
        modules.globals.color_correction = state.get("color_correction", False)
        modules.globals.face_color_match = state.get("face_color_match", False)
        modules.globals.source_skin_tone = state.get("source_skin_tone", False)
        modules.globals.web_stream = state.get("web_stream", False)
        modules.globals.nsfw_filter = state.get("nsfw_filter", False)
        modules.globals.live_mirror = state.get("live_mirror", False)
        modules.globals.live_resizable = state.get("live_resizable", False)
        modules.globals.fp_ui = state.get("fp_ui", {"face_enhancer": False})
        modules.globals.show_fps = state.get("show_fps", False)
        modules.globals.teeth_effect = state.get("teeth_effect", "None")
        modules.globals.teeth_image_path = state.get("teeth_image_path", "")
        # Mouth mask always starts disabled (slider at 0) on launch,
        # regardless of the persisted value — enable it explicitly each session.
        modules.globals.mouth_mask_size = 0.0
        modules.globals.mouth_mask = False
        modules.globals.show_mouth_mask_box = False
    except FileNotFoundError:
        pass
    except (OSError, json.JSONDecodeError):
        pass


# ─── thread-safe status bridge ───────────────────────────────────────────


class _UIBridge(QObject):
    """Single QObject that owns cross-thread signals."""

    statusChanged = Signal(str)


def _emit_status(text: str) -> None:
    if _BRIDGE is None:
        print(text)
        return
    _BRIDGE.statusChanged.emit(text)


# ─── public API ──────────────────────────────────────────────────────────


def update_status(text: str) -> None:
    """Thread-safe status update — uses signal if called off-UI thread."""
    _emit_status(_(text))
    if _APP is not None and QThread.currentThread() is _APP.thread():
        # On UI thread — flush events so the user sees the update during
        # long synchronous start() runs.
        _APP.processEvents()


def check_and_ignore_nsfw(target, destroy: Optional[Callable] = None) -> bool:
    from numpy import ndarray
    from modules.predicter import predict_frame, predict_image, predict_video

    check_nsfw = None
    if isinstance(target, str):
        check_nsfw = predict_image if has_image_extension(target) else predict_video
    elif isinstance(target, ndarray):
        check_nsfw = predict_frame

    if check_nsfw and check_nsfw(target):
        if destroy:
            destroy(to_quit=False)
        update_status("Processing ignored!")
        return True
    return False


# ─── camera enumeration (unchanged from tk version) ──────────────────────


def get_available_cameras() -> Tuple[List[int], List[str]]:
    if platform.system() == "Windows":
        try:
            graph = FilterGraph()
            devices = graph.get_input_devices()
            if devices:
                return list(range(len(devices))), devices
            return [], ["No cameras found"]
        except Exception as exc:
            print(f"Error detecting cameras: {exc}")
            return [], ["No cameras found"]

    if platform.system() == "Darwin":
        return [0, 1], ["Camera 0", "Camera 1"]

    # Linux probe
    indices: List[int] = []
    names: List[str] = []
    for i in range(10):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            indices.append(i)
            names.append(f"Camera {i}")
            cap.release()
    return (indices, names) if names else ([], ["No cameras found"])


# ─── main window ─────────────────────────────────────────────────────────


def _make_image_drop(text: str, size: Tuple[int, int]) -> QLabel:
    label = QLabel(text)
    label.setObjectName("imageDrop")
    label.setAlignment(Qt.AlignmentFlag.AlignCenter)
    label.setFixedSize(size[0], size[1])
    label.setText(text)
    return label


class _Switch(QWidget):
    """Compact toggle switch with label + optional tooltip."""

    toggled = Signal(bool)

    def __init__(self, text: str, initial: bool, tooltip: str = ""):
        super().__init__()
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self._checkbox = QCheckBox(text)
        self._checkbox.setChecked(initial)
        self._checkbox.toggled.connect(self.toggled.emit)
        if tooltip:
            self._checkbox.setToolTip(tooltip)
        layout.addWidget(self._checkbox)
        layout.addStretch(1)

    def isChecked(self) -> bool:
        return self._checkbox.isChecked()

    def setChecked(self, value: bool) -> None:
        self._checkbox.setChecked(value)


class _SourceThumb(QLabel):
    """A small clickable thumbnail for one source photo. Click to remove it."""

    clicked = Signal(str)

    def __init__(self, path: str, pixmap: QPixmap):
        super().__init__()
        self._path = path
        self.setObjectName("sourceThumb")
        self.setPixmap(pixmap)
        self.setFixedSize(46, 46)
        self.setScaledContents(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setToolTip(_("Click to remove this photo"))

    def mousePressEvent(self, _event) -> None:
        self.clicked.emit(self._path)


class MainWindow(QMainWindow):
    def __init__(self, start_cb: Callable, destroy_cb: Callable):
        super().__init__()
        load_switch_states()
        self._start_cb = start_cb
        self._destroy_cb = destroy_cb

        self.setWindowTitle(
            f"{modules.metadata.name} {modules.metadata.version} {modules.metadata.edition}".strip()
        )
        self.setMinimumSize(ROOT_WIDTH, ROOT_HEIGHT)
        self.resize(ROOT_WIDTH, ROOT_HEIGHT)

        # Central content lives inside a scroll area so it never overlaps when
        # the content is taller than the window (small screens / high DPI) — a
        # scrollbar appears instead of widgets drawing on top of each other.
        root = QWidget()
        layout = QVBoxLayout(root)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        self._main_scroll = QScrollArea()
        self._main_scroll.setWidgetResizable(True)
        self._main_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._main_scroll.setWidget(root)
        self.setCentralWidget(self._main_scroll)

        # Source/Target row
        layout.addLayout(self._build_image_row())

        # Options grid
        layout.addWidget(self._build_options_card())

        # Sliders card
        layout.addWidget(self._build_sliders_card())

        # Action buttons
        layout.addLayout(self._build_action_row())

        # Camera selection
        layout.addWidget(self._build_camera_card())

        # Status & footer
        self._status_label = QLabel("")
        self._status_label.setObjectName("statusLabel")
        self._status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._status_label)

        footer = QLabel("Live Face Swap — by Peter")
        footer.setObjectName("linkLabel")
        footer.setAlignment(Qt.AlignmentFlag.AlignCenter)
        footer.setCursor(Qt.CursorShape.PointingHandCursor)
        footer.mousePressEvent = lambda _e: webbrowser.open("https://deeplivecam.net")
        layout.addWidget(footer)

    # ── image row ────────────────────────────────────────────────────────

    def _build_image_row(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(16)

        # Source column — supports one OR several photos of the SAME person.
        self._source_paths: List[str] = []
        src_col = QVBoxLayout()
        src_col.setSpacing(6)
        self.source_label = _make_image_drop(_("Source face"), (200, 200))
        src_col.addWidget(self.source_label, alignment=Qt.AlignmentFlag.AlignCenter)

        # Thumbnail strip: every added photo shows here, click a thumb to remove.
        self._source_thumb_row = QWidget()
        self._source_thumb_row.setFixedWidth(200)
        self._source_thumb_layout = QHBoxLayout(self._source_thumb_row)
        self._source_thumb_layout.setContentsMargins(0, 0, 0, 0)
        self._source_thumb_layout.setSpacing(4)
        self._source_thumb_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        src_col.addWidget(self._source_thumb_row, alignment=Qt.AlignmentFlag.AlignCenter)

        # Caption explaining the multi-photo benefit / current state.
        self._source_caption = QLabel("")
        self._source_caption.setObjectName("hintLabel")
        self._source_caption.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._source_caption.setWordWrap(True)
        self._source_caption.setFixedWidth(200)
        src_col.addWidget(self._source_caption, alignment=Qt.AlignmentFlag.AlignCenter)

        src_row = QHBoxLayout()
        src_row.setSpacing(6)
        self.btn_select_source = QPushButton(_("Select a face"))
        self.btn_select_source.setToolTip(
            _("Choose the source face image to swap onto the target")
        )
        self.btn_select_source.clicked.connect(self._on_select_source)
        self.btn_add_source = QPushButton(_("+ Add"))
        self.btn_add_source.setObjectName("secondary")
        self.btn_add_source.setFixedWidth(78)
        self.btn_add_source.setToolTip(
            _("Add more photos of the SAME person (different angles/sizes) for a sharper, more stable swap")
        )
        self.btn_add_source.clicked.connect(self._on_add_source)
        self.btn_random_face = QPushButton("🔄")
        self.btn_random_face.setObjectName("secondary")
        self.btn_random_face.setFixedWidth(40)
        self.btn_random_face.setToolTip(
            _("Get a random face from thispersondoesnotexist.com")
        )
        self.btn_random_face.clicked.connect(self._on_random_face)
        src_row.addWidget(self.btn_select_source)
        src_row.addWidget(self.btn_add_source)
        src_row.addWidget(self.btn_random_face)
        src_col.addLayout(src_row)
        self._refresh_source_ui()

        # Swap button column
        swap_col = QVBoxLayout()
        swap_col.addStretch(1)
        self.btn_swap = QPushButton("↔")
        self.btn_swap.setObjectName("secondary")
        self.btn_swap.setFixedSize(44, 44)
        self.btn_swap.setToolTip(_("Swap source and target images"))
        self.btn_swap.clicked.connect(self._on_swap_paths)
        swap_col.addWidget(self.btn_swap, alignment=Qt.AlignmentFlag.AlignCenter)
        swap_col.addStretch(1)

        # Target column
        tgt_col = QVBoxLayout()
        self.target_label = _make_image_drop(_("Target"), (200, 200))
        tgt_col.addWidget(self.target_label, alignment=Qt.AlignmentFlag.AlignCenter)
        self.btn_select_target = QPushButton(_("Select a target"))
        self.btn_select_target.setToolTip(
            _("Choose the target image or video to apply face swap to")
        )
        self.btn_select_target.clicked.connect(self._on_select_target)
        tgt_col.addWidget(self.btn_select_target)

        row.addLayout(src_col)
        row.addLayout(swap_col)
        row.addLayout(tgt_col)
        return row

    # ── options card ─────────────────────────────────────────────────────

    def _build_options_card(self) -> QGroupBox:
        card = QGroupBox(_("Options"))
        grid = QGridLayout(card)
        grid.setHorizontalSpacing(20)
        grid.setVerticalSpacing(6)

        def make(field, label, tip):
            sw = _Switch(_(label), getattr(modules.globals, field), _(tip))
            sw.toggled.connect(
                lambda v, f=field: (
                    setattr(modules.globals, f, v),
                    save_switch_states(),
                )
            )
            return sw

        self.sw_keep_fps = make("keep_fps", "Keep fps",
                                "Output video keeps the original frame rate")
        self.sw_keep_audio = make("keep_audio", "Keep audio",
                                  "Copy audio track from the source video to output")
        self.sw_keep_frames = make("keep_frames", "Keep frames",
                                   "Keep extracted frames on disk after processing")
        self.sw_many_faces = make("many_faces", "Many faces",
                                  "Swap every detected face, not just the primary one")
        self.sw_poisson = make("poisson_blend", "Poisson Blend",
                               "Blend face edges smoothly using Poisson blending")
        self.sw_color_fix = make("color_correction", "Fix Blueish Cam",
                                 "Fix blue/green color cast from some webcams")
        self.sw_face_color = make("face_color_match", "Match Face Color",
                                  "Match the swapped face's color/lighting to the target face")
        self.sw_source_tone = make("source_skin_tone", "Keep Source Skin Tone",
                                   "Keep the REAL skin colour of the photo you picked instead of "
                                   "repainting the face with your own. Only the lighting follows "
                                   "the room, so it still looks natural. Turn this on when the "
                                   "swap should look exactly like the person in the picture")
        self.sw_adaptive = make("adaptive_face_mask", "Auto-fit Face",
                                "Fit the swap to your real face outline so it covers your WHOLE face "
                                "even when the source photo's face is a different size/shape")
        self.sw_web_stream = make("web_stream", "Stream to Phone (Web)",
                                  "Serve the live output over Wi-Fi; open the printed URL on your phone's browser")
        self.sw_show_fps = make("show_fps", "Show FPS",
                                "Display frames-per-second counter on the live preview")

        # Map faces is special — closes mapper when toggled off.
        self.sw_map_faces = _Switch(_("Map faces"), modules.globals.map_faces,
                                    _("Manually assign which source face maps to which target face"))
        self.sw_map_faces.toggled.connect(self._on_map_faces_toggled)

        # Layout: 2 columns of switches
        items = [
            self.sw_keep_fps, self.sw_keep_audio,
            self.sw_keep_frames, self.sw_many_faces,
            self.sw_map_faces, self.sw_show_fps,
            self.sw_poisson, self.sw_color_fix,
            self.sw_face_color, self.sw_adaptive,
            self.sw_source_tone, self.sw_web_stream,
        ]
        for i, w in enumerate(items):
            grid.addWidget(w, i // 2, i % 2)

        # Face enhancer dropdown — place on the row AFTER the last switch row.
        # ceil(len/2) handles odd switch counts (a lone switch on the last row
        # would otherwise share a cell with this label).
        enhancer_row = (len(items) + 1) // 2
        enhancer_label = QLabel(_("Face Enhancer:"))
        grid.addWidget(enhancer_label, enhancer_row, 0)

        self.cb_enhancer = QComboBox()
        self.cb_enhancer.addItems(["None", "GFPGAN", "GPEN-512", "GPEN-256"])
        initial = "None"
        if modules.globals.fp_ui.get("face_enhancer", False):
            initial = "GFPGAN"
        elif modules.globals.fp_ui.get("face_enhancer_gpen512", False):
            initial = "GPEN-512"
        elif modules.globals.fp_ui.get("face_enhancer_gpen256", False):
            initial = "GPEN-256"
        self.cb_enhancer.setCurrentText(initial)
        self.cb_enhancer.currentTextChanged.connect(self._on_enhancer_change)
        self.cb_enhancer.setToolTip(_("Select a face enhancement model (None = no enhancement)"))
        grid.addWidget(self.cb_enhancer, enhancer_row, 1)

        # Teeth effect dropdown — an optional cosmetic overlay on the swapped
        # mouth. "None" = off (default); nothing else in the pipeline changes.
        teeth_row = enhancer_row + 1
        teeth_label = QLabel(_("Teeth Effect:"))
        grid.addWidget(teeth_label, teeth_row, 0)
        self.cb_teeth = QComboBox()
        self.cb_teeth.addItems(["None", "White", "Add Tooth", "Gold", "Silver", "Image…"])
        self.cb_teeth.setCurrentText(getattr(modules.globals, "teeth_effect", "None") or "None")
        self.cb_teeth.currentTextChanged.connect(self._on_teeth_change)
        self.cb_teeth.setToolTip(
            _("Show a teeth style on the swapped mouth (None = off): "
              "White = whiten, Add Tooth = fill a missing-tooth gap, "
              "Gold/Silver = solid grillz, Image… = overlay your own grillz picture")
        )
        grid.addWidget(self.cb_teeth, teeth_row, 1)

        return card

    # ── sliders card ─────────────────────────────────────────────────────

    def _build_sliders_card(self) -> QGroupBox:
        card = QGroupBox(_("Refinement"))
        grid = QGridLayout(card)
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(10)

        def slider(min_v, max_v, default, denom, on_change):
            s = QSlider(Qt.Orientation.Horizontal)
            s.setRange(int(min_v * denom), int(max_v * denom))
            s.setValue(int(default * denom))
            s.valueChanged.connect(lambda iv: on_change(iv / denom))
            return s

        # Transparency
        grid.addWidget(QLabel(_("Transparency")), 0, 0)
        self.s_transparency = slider(0.0, 1.0, 1.0, 100, self._on_transparency_change)
        self.s_transparency.setToolTip(
            _("Blend between original and swapped face (0% = original, 100% = fully swapped)")
        )
        grid.addWidget(self.s_transparency, 0, 1)

        # Sharpness
        grid.addWidget(QLabel(_("Sharpness")), 1, 0)
        self.s_sharpness = slider(0.0, 5.0, 0.0, 10, self._on_sharpness_change)
        self.s_sharpness.setToolTip(_("Sharpen the enhanced face output"))
        grid.addWidget(self.s_sharpness, 1, 1)

        # Color Match Strength — how strongly to pull the swapped face toward the
        # target's color/lighting (only has effect when "Match Face Color" is on).
        grid.addWidget(QLabel(_("Color Match")), 3, 0)
        self.s_color_match = slider(0.0, 1.0, 1.0, 100, self._on_color_match_strength_change)
        self.s_color_match.setToolTip(
            _("Strength of Match Face Color (0% = raw swap, 100% = full match to target)")
        )
        grid.addWidget(self.s_color_match, 3, 1)

        # Mouth mask — always starts at 0 (disabled) on launch
        grid.addWidget(QLabel(_("Mouth Mask")), 2, 0)
        self.s_mouth = slider(0.0, 100.0, 0.0, 1,
                              self._on_mouth_mask_change)
        self.s_mouth.sliderPressed.connect(self._on_mouth_mask_pressed)
        self.s_mouth.sliderReleased.connect(self._on_mouth_mask_released)
        self.s_mouth.setToolTip(
            _("0 = use swapped mouth, 100 = expose original mouth to chin area")
        )
        grid.addWidget(self.s_mouth, 2, 1)
        return card

    # ── action row ───────────────────────────────────────────────────────

    def _build_action_row(self) -> QHBoxLayout:
        row = QHBoxLayout()
        self.btn_start = QPushButton(_("Start"))
        self.btn_start.setToolTip(_("Begin processing the target image/video with selected face"))
        self.btn_start.clicked.connect(self._on_start)

        self.btn_destroy = QPushButton(_("Destroy"))
        self.btn_destroy.setObjectName("danger")
        self.btn_destroy.setToolTip(_("Stop processing and close the application"))
        self.btn_destroy.clicked.connect(lambda: self._destroy_cb())

        self.btn_preview = QPushButton(_("Preview"))
        self.btn_preview.setObjectName("secondary")
        self.btn_preview.setToolTip(_("Show/hide a preview of the processed output"))
        self.btn_preview.clicked.connect(self._on_toggle_preview)

        row.addWidget(self.btn_start)
        row.addWidget(self.btn_destroy)
        row.addWidget(self.btn_preview)
        return row

    # ── camera card ──────────────────────────────────────────────────────

    def _build_camera_card(self) -> QGroupBox:
        card = QGroupBox(_("Camera"))
        layout = QHBoxLayout(card)

        layout.addWidget(QLabel(_("Select Camera:")))
        self._camera_indices, self._camera_names = get_available_cameras()

        self.cb_camera = QComboBox()
        if not self._camera_names or self._camera_names[0] == "No cameras found":
            self.cb_camera.addItem("No cameras found")
            self.cb_camera.setEnabled(False)
            cam_ok = False
        else:
            self.cb_camera.addItems(self._camera_names)
            cam_ok = True
        self.cb_camera.setToolTip(_("Select which camera to use for live mode"))
        layout.addWidget(self.cb_camera, 1)

        self.btn_live = QPushButton(_("● Live"))
        self.btn_live.setObjectName("success")
        self.btn_live.setEnabled(cam_ok)
        self.btn_live.setToolTip(_("Start real-time face swap using webcam"))
        self.btn_live.clicked.connect(self._on_live)
        layout.addWidget(self.btn_live)

        return card

    # ── slot handlers ────────────────────────────────────────────────────

    def set_status(self, text: str) -> None:
        self._status_label.setText(text)

    def _on_select_source(self) -> None:
        # Pick 1 photo for a normal swap, or select several at once. This
        # REPLACES the current set.
        if _PREVIEW is not None:
            _PREVIEW.hide()
        paths, _filter = QFileDialog.getOpenFileNames(
            self, _("Select 1 photo — or 2+ photos of the SAME person for a better swap"),
            _RECENT_SOURCE_DIR or "",
            "Images (*.png *.jpg *.jpeg *.gif *.bmp)",
        )
        self._apply_source_paths(paths, append=False)

    def _on_add_source(self) -> None:
        # ADD more photos of the same person to the current set (keeps existing).
        if _PREVIEW is not None:
            _PREVIEW.hide()
        paths, _filter = QFileDialog.getOpenFileNames(
            self, _("Add more photos of the SAME person"),
            _RECENT_SOURCE_DIR or "",
            "Images (*.png *.jpg *.jpeg *.gif *.bmp)",
        )
        self._apply_source_paths(paths, append=True)

    def _apply_source_paths(self, paths, append: bool) -> None:
        """Set or extend the source-photo set, then rebuild the swap identity."""
        paths = [p for p in (paths or []) if p and is_image(p)]
        if not paths:
            return
        if append:
            combined = list(self._source_paths)
            for p in paths:
                if p not in combined:
                    combined.append(p)
            self._source_paths = combined
        else:
            self._source_paths = paths
        self._rebuild_source()

    def _remove_source_path(self, path: str) -> None:
        """Drop one photo (clicked in the thumbnail strip) and re-average."""
        self._source_paths = [p for p in self._source_paths if p != path]
        self._rebuild_source()

    def _rebuild_source(self) -> None:
        """Apply self._source_paths to globals: first photo is the preview, and
        2+ photos of the same person get their face embeddings averaged into a
        single stronger identity (source_face_override)."""
        global _RECENT_SOURCE_DIR
        paths = self._source_paths
        modules.globals.source_face_override = None
        if not paths:
            modules.globals.source_path = None
            self.source_label.clear()
            self.source_label.setText(_("Source face"))
            self._refresh_source_ui()
            return
        modules.globals.source_path = paths[0]
        _RECENT_SOURCE_DIR = os.path.dirname(paths[0])
        if len(paths) > 1:
            try:
                from modules.face_analyser import build_average_source_face
                avg = build_average_source_face(paths)
                if avg is not None:
                    modules.globals.source_face_override = avg
                    update_status(
                        f"Blended {len(paths)} photos of the same person — sharper, steadier face"
                    )
                else:
                    update_status("Couldn't read a face in the extra photos — using the first one")
            except Exception as exc:
                print(f"multi-source averaging failed, using first image: {exc}")
        self.source_label.setPixmap(render_image_preview(paths[0], (200, 200)))
        self.source_label.setText("")
        self._refresh_source_ui()

    def _refresh_source_ui(self) -> None:
        """Rebuild the thumbnail strip and the caption under the source box."""
        # Clear existing thumbnails.
        while self._source_thumb_layout.count():
            item = self._source_thumb_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()

        paths = self._source_paths
        n = len(paths)
        # Show a strip only when there are 2+ photos (a single photo is already
        # shown in the big preview box above).
        self._source_thumb_row.setVisible(n > 1)
        if n > 1:
            MAX_THUMBS = 6
            for p in paths[:MAX_THUMBS]:
                try:
                    pm = render_image_preview(p, (46, 46))
                except Exception:
                    continue
                thumb = _SourceThumb(p, pm)
                thumb.clicked.connect(self._remove_source_path)
                self._source_thumb_layout.addWidget(thumb)
            if n > MAX_THUMBS:
                more = QLabel(f"+{n - MAX_THUMBS}")
                more.setObjectName("hintLabel")
                more.setAlignment(Qt.AlignmentFlag.AlignCenter)
                self._source_thumb_layout.addWidget(more)

        if n == 0:
            self._source_caption.setText(
                _("Tip: add 2+ photos of the SAME person for a sharper swap")
            )
        elif n == 1:
            self._source_caption.setText(
                _("1 photo · click + Add for more of the same person")
            )
        elif modules.globals.source_face_override is not None:
            self._source_caption.setText(
                _("%d photos blended ✓  (click a thumbnail to remove)") % n
            )
        else:
            self._source_caption.setText(
                _("%d photos · using the 1st (no face found in the others)") % n
            )

    def _on_select_target(self) -> None:
        global _RECENT_TARGET_DIR
        if _PREVIEW is not None:
            _PREVIEW.hide()
        path, _filter = QFileDialog.getOpenFileName(
            self, _("select an target image or video"),
            _RECENT_TARGET_DIR or "",
            "Media (*.png *.jpg *.jpeg *.gif *.bmp *.mp4 *.mkv)",
        )
        if not path:
            return
        if is_image(path):
            modules.globals.target_path = path
            _RECENT_TARGET_DIR = os.path.dirname(path)
            self.target_label.setPixmap(render_image_preview(path, (200, 200)))
            self.target_label.setText("")
        elif is_video(path):
            modules.globals.target_path = path
            _RECENT_TARGET_DIR = os.path.dirname(path)
            pm = render_video_preview(path, (200, 200))
            if pm:
                self.target_label.setPixmap(pm)
                self.target_label.setText("")
        else:
            modules.globals.target_path = None
            self.target_label.clear()
            self.target_label.setText(_("Target"))

    def _on_random_face(self) -> None:
        if _PREVIEW is not None:
            _PREVIEW.hide()
        try:
            response = requests.get(
                "https://thispersondoesnotexist.com/",
                headers={"User-Agent": "Mozilla/5.0"},
                timeout=10,
            )
            response.raise_for_status()
            temp_path = os.path.join(tempfile.gettempdir(), "deep_live_cam_random_face.jpg")
            with open(temp_path, "wb") as f:
                f.write(response.content)
            self._source_paths = [temp_path]
            self._rebuild_source()
        except Exception as exc:
            print(f"Failed to fetch random face: {exc}")

    def _on_swap_paths(self) -> None:
        global _RECENT_SOURCE_DIR, _RECENT_TARGET_DIR
        sp = modules.globals.source_path
        tp = modules.globals.target_path
        if not (sp and tp and is_image(sp) and is_image(tp)):
            return
        modules.globals.source_path, modules.globals.target_path = tp, sp
        _RECENT_SOURCE_DIR = os.path.dirname(tp)
        _RECENT_TARGET_DIR = os.path.dirname(sp)
        if _PREVIEW is not None:
            _PREVIEW.hide()
        # The swapped-in source is now a single image — reset the multi-photo set
        # and clear any averaged identity so the strip/caption stay accurate.
        self._source_paths = [tp]
        modules.globals.source_face_override = None
        self.source_label.setPixmap(render_image_preview(tp, (200, 200)))
        self.target_label.setPixmap(render_image_preview(sp, (200, 200)))
        self.source_label.setText("")
        self.target_label.setText("")
        self._refresh_source_ui()

    def _on_map_faces_toggled(self, value: bool) -> None:
        modules.globals.map_faces = value
        save_switch_states()
        if not value:
            close_mapper_window()

    def _on_enhancer_change(self, choice: str) -> None:
        key_map = {
            "None": None,
            "GFPGAN": "face_enhancer",
            "GPEN-512": "face_enhancer_gpen512",
            "GPEN-256": "face_enhancer_gpen256",
        }
        for key in ("face_enhancer", "face_enhancer_gpen256", "face_enhancer_gpen512"):
            _update_tumbler(key, False)
        selected = key_map.get(choice)
        if selected:
            _update_tumbler(selected, True)
        save_switch_states()

    def _on_teeth_change(self, choice: str) -> None:
        if choice and choice.startswith("Image"):
            # Ask the user to pick a grillz/teeth image. On cancel, revert the
            # dropdown to whatever was selected before (don't silently arm an
            # image mode with no image).
            path, _f = QFileDialog.getOpenFileName(
                self,
                _("Select a teeth / grillz image (PNG with transparency works best)"),
                _RECENT_SOURCE_DIR or "",
                "Images (*.png *.jpg *.jpeg *.webp *.bmp)",
            )
            if path:
                modules.globals.teeth_image_path = path
                modules.globals.teeth_effect = choice
                save_switch_states()
                update_status(f"Teeth image: {os.path.basename(path)}")
            else:
                prev = getattr(modules.globals, "teeth_effect", "None") or "None"
                self.cb_teeth.blockSignals(True)
                self.cb_teeth.setCurrentText(prev if self.cb_teeth.findText(prev) >= 0 else "None")
                self.cb_teeth.blockSignals(False)
            return
        modules.globals.teeth_effect = choice or "None"
        save_switch_states()
        if choice and choice != "None":
            update_status(f"Teeth effect: {choice}")
        else:
            update_status("Teeth effect off")

    def _on_transparency_change(self, value: float) -> None:
        modules.globals.opacity = value
        pct = int(value * 100)
        if pct == 0:
            modules.globals.fp_ui["face_enhancer"] = False
            update_status("Transparency set to 0% - Face swapping disabled.")
        elif pct == 100:
            modules.globals.face_swapper_enabled = True
            update_status("Transparency set to 100%.")
        else:
            modules.globals.face_swapper_enabled = True
            update_status(f"Transparency set to {pct}%")

    def _on_sharpness_change(self, value: float) -> None:
        modules.globals.sharpness = value
        update_status(f"Sharpness set to {value:.1f}")

    def _on_color_match_strength_change(self, value: float) -> None:
        modules.globals.face_color_match_strength = value
        update_status(f"Color match strength set to {int(value * 100)}%")

    def _on_mouth_mask_change(self, value: float) -> None:
        modules.globals.mouth_mask_size = value
        modules.globals.mouth_mask = value > 0
        if value <= 0:
            modules.globals.show_mouth_mask_box = False

    def _on_mouth_mask_pressed(self) -> None:
        if modules.globals.mouth_mask_size > 0:
            modules.globals.show_mouth_mask_box = True

    def _on_mouth_mask_released(self) -> None:
        modules.globals.show_mouth_mask_box = False

    def _on_start(self) -> None:
        if _MAPPER is not None and _MAPPER.isVisible():
            update_status("Please complete pop-up or close it.")
            return
        if modules.globals.map_faces:
            modules.globals.source_target_map = []
            if is_image(modules.globals.target_path):
                update_status("Getting unique faces")
                get_unique_faces_from_target_image()
            elif is_video(modules.globals.target_path):
                update_status("Getting unique faces")
                get_unique_faces_from_target_video()
            if modules.globals.source_target_map:
                _open_mapper_dialog(self._start_cb, modules.globals.source_target_map)
            else:
                update_status("No faces found in target")
        else:
            self._select_output_and_start()

    def _select_output_and_start(self) -> None:
        global _RECENT_OUTPUT_DIR
        if is_image(modules.globals.target_path):
            path, _f = QFileDialog.getSaveFileName(
                self, _("save image output file"),
                os.path.join(_RECENT_OUTPUT_DIR or "", "output.png"),
                "Images (*.png *.jpg *.jpeg *.bmp)",
            )
        elif is_video(modules.globals.target_path):
            path, _f = QFileDialog.getSaveFileName(
                self, _("save video output file"),
                os.path.join(_RECENT_OUTPUT_DIR or "", "output.mp4"),
                "Videos (*.mp4 *.mkv)",
            )
        else:
            return
        if path:
            modules.globals.output_path = path
            _RECENT_OUTPUT_DIR = os.path.dirname(path)
            self._start_cb()

    def _on_toggle_preview(self) -> None:
        if _PREVIEW is None:
            return
        if _PREVIEW.isVisible():
            _PREVIEW.hide()
        elif modules.globals.source_path and modules.globals.target_path:
            _PREVIEW.init_for_target()
            _PREVIEW.refresh_frame(0)
            _PREVIEW.show()

    def _on_live(self) -> None:
        idx = self.cb_camera.currentIndex()
        if idx < 0 or idx >= len(self._camera_indices):
            update_status("No camera available")
            return
        camera_index = self._camera_indices[idx]
        if _LIVE_MAPPER is not None and _LIVE_MAPPER.isVisible():
            update_status("Source x Target Mapper is already open.")
            _LIVE_MAPPER.raise_()
            return
        if not modules.globals.map_faces:
            if modules.globals.source_path is None:
                update_status("Please select a source image first")
                return
            from modules.face_analyser import get_face_analyser
            from modules.processors.frame.face_swapper import get_face_swapper
            get_face_analyser()
            get_face_swapper()
            _open_webcam_preview(camera_index)
        else:
            modules.globals.source_target_map = []
            _open_live_mapper_dialog(camera_index, modules.globals.source_target_map)

    def closeEvent(self, event):
        # Treat OS-level close as Destroy click
        self._destroy_cb()
        event.accept()


def _update_tumbler(var: str, value: bool) -> None:
    modules.globals.fp_ui[var] = value
    save_switch_states()
    # If we're currently in a live preview, refresh frame processors so
    # toggling enhancers takes effect immediately.
    if _WEBCAM_PREVIEW is not None and _WEBCAM_PREVIEW.isVisible():
        get_frame_processors_modules(modules.globals.frame_processors)


# ─── preview window (still-image / video scrub) ──────────────────────────


class PreviewWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(_("Preview"))
        self.resize(PREVIEW_DEFAULT_WIDTH, PREVIEW_DEFAULT_HEIGHT)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._image_label = QLabel()
        self._image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._image_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        layout.addWidget(self._image_label, 1)

        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setRange(0, 0)
        self._slider.valueChanged.connect(self.refresh_frame)
        layout.addWidget(self._slider)

    def init_for_target(self) -> None:
        if is_image(modules.globals.target_path):
            self._slider.hide()
        elif is_video(modules.globals.target_path):
            total = get_video_frame_total(modules.globals.target_path)
            self._slider.setRange(0, max(0, total - 1))
            self._slider.setValue(0)
            self._slider.show()

    def refresh_frame(self, frame_number: int = 0) -> None:
        if not (modules.globals.source_path and modules.globals.target_path):
            return
        update_status("Processing...")
        temp_frame = get_video_frame(modules.globals.target_path, frame_number)
        if modules.globals.nsfw_filter and check_and_ignore_nsfw(temp_frame):
            return
        from modules.processors.frame.core import get_frame_processors_modules as _gfpm
        for fp in _gfpm(modules.globals.frame_processors):
            temp_frame = fp.process_frame(
                get_one_face(imread_unicode(modules.globals.source_path)), temp_frame
            )
        # Fit to current widget size while preserving aspect ratio.
        h, w = temp_frame.shape[:2]
        bound_w = min(PREVIEW_MAX_WIDTH, max(self.width(), PREVIEW_DEFAULT_WIDTH))
        bound_h = min(PREVIEW_MAX_HEIGHT, max(self.height(), PREVIEW_DEFAULT_HEIGHT))
        ratio = min(bound_w / w, bound_h / h)
        new_size = (max(1, int(w * ratio)), max(1, int(h * ratio)))
        temp_frame = cv2.resize(temp_frame, new_size, interpolation=cv2.INTER_LANCZOS4)
        self._image_label.setPixmap(_bgr_to_qpixmap(temp_frame))
        update_status("Processing succeed!")


# ─── webcam preview window ───────────────────────────────────────────────


class _CaptureWorker(QThread):
    """Reads frames from the camera into a bounded queue. Drops on overflow."""

    def __init__(self, cap, capture_queue: queue.Queue, stop_event: threading.Event):
        super().__init__()
        self._cap = cap
        self._queue = capture_queue
        self._stop = stop_event

    def run(self) -> None:
        consecutive_fail = 0
        while not self._stop.is_set():
            ret, frame = self._cap.read()
            # A phone-as-webcam (DroidCam/Iriun) can briefly return a bad/empty
            # frame on a Wi-Fi or low-battery hiccup. Skip those instead of
            # tearing down the live session; only give up after sustained loss.
            if not ret or frame is None or getattr(frame, "size", 0) == 0:
                consecutive_fail += 1
                if consecutive_fail > 150:
                    self._stop.set()
                    break
                continue
            consecutive_fail = 0
            try:
                self._queue.put_nowait(frame)
            except queue.Full:
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    pass
                try:
                    self._queue.put_nowait(frame)
                except queue.Full:
                    pass


class _ProcessingWorker(QThread):
    """Pulls raw frames, runs detect/swap/enhance, pushes processed frames."""

    # 90° rotations to try when auto-orienting a sideways camera feed.
    _ROT_CANDIDATES = (
        None,
        cv2.ROTATE_90_CLOCKWISE,
        cv2.ROTATE_180,
        cv2.ROTATE_90_COUNTERCLOCKWISE,
    )

    def __init__(self, capture_queue, processed_queue, stop_event, camera_fps: float):
        super().__init__()
        self._cq = capture_queue
        self._pq = processed_queue
        self._stop = stop_event
        self._fps = camera_fps
        # Auto-orientation state. Phone-as-webcam feeds (DroidCam/Iriun) often
        # arrive rotated 90°, which stops the detector from finding any face so
        # nothing gets swapped. We lock onto the orientation that yields a face.
        self._rot_code = None        # cv2 rotate code, or None for upright
        self._rot_locked = False
        self._rot_miss = 0

    def _auto_orient(self, frame):
        """Return `frame` rotated so a face is upright (best-effort).

        While unlocked, probe the four 90° orientations and lock onto the first
        that yields a detectable face. Once locked, just apply that rotation
        (one cheap cv2.rotate/frame). Fully guarded — any failure returns the
        frame untouched so the live loop is never broken.
        """
        try:
            if self._rot_locked:
                if self._rot_code is None:
                    return frame
                return cv2.rotate(frame, self._rot_code)
            # The modern detector finds a face even when it's rotated 90°, so we
            # can't just take the first orientation that yields a face (that locks
            # onto the sideways feed). Instead score each orientation by how
            # UPRIGHT the face is — eyes horizontal and above the mouth — using the
            # 5 keypoints, and lock onto the best.
            best = None  # (score, code, rotated_frame)
            for code in self._ROT_CANDIDATES:
                cand = frame if code is None else cv2.rotate(frame, code)
                face = detect_one_face_fast(cand)
                if face is None:
                    continue
                kps = getattr(face, "kps", None)
                if kps is None or len(kps) < 5:
                    continue
                le, re_, _nose, lm, rm = kps[0], kps[1], kps[2], kps[3], kps[4]
                eye_dx = abs(float(le[0]) - float(re_[0]))
                eye_dy = abs(float(le[1]) - float(re_[1]))
                eyes_y = (float(le[1]) + float(re_[1])) / 2.0
                mouth_y = (float(lm[1]) + float(rm[1])) / 2.0
                horiz = eye_dx / (eye_dx + eye_dy + 1e-6)   # ~1 when eyes level
                above = 1.0 if eyes_y < mouth_y else 0.0     # eyes over mouth
                # Detection confidence is highest for the genuinely upright face;
                # the geometry terms break the tie between the two 90° rotations
                # (both make eyes level, but only one is right-side-up).
                det_score = float(getattr(face, "det_score", 0.0) or 0.0)
                score = det_score + above * 0.5 + horiz * 0.3
                if best is None or score > best[0]:
                    best = (score, code, cand)
            if best is not None:
                self._rot_code = best[1]
                self._rot_locked = True
                self._rot_miss = 0
                return best[2]
        except Exception:
            pass
        return frame

    def run(self) -> None:
        frame_processors = get_frame_processors_modules(modules.globals.frame_processors)
        source_image = None
        last_source_key = None
        prev_time = time.time()
        fps_update_interval = 0.5
        frame_count = 0
        fps = 0.0
        det_count = 0
        cached_target_face = None
        cached_many_faces = None
        det_interval = max(1, round(self._fps * 0.08))

        while not self._stop.is_set():
            try:
                frame = self._cq.get(timeout=0.05)
            except queue.Empty:
                continue

            if frame is None or getattr(frame, "size", 0) == 0:
                continue
            frame = self._auto_orient(frame)
            if frame is None or getattr(frame, "size", 0) == 0:
                continue
            temp_frame = frame
            if modules.globals.live_mirror:
                temp_frame = gpu_flip(temp_frame, 1)

            if not modules.globals.map_faces:
                # Reload the source when the path OR the averaged-faces override
                # changes. If a multi-photo average is set, use it directly;
                # otherwise detect the face in the single source image.
                _override = getattr(modules.globals, "source_face_override", None)
                src_key = (modules.globals.source_path, id(_override))
                if modules.globals.source_path and src_key != last_source_key:
                    last_source_key = src_key
                    source_image = (
                        _override if _override is not None
                        else get_one_face(imread_unicode(modules.globals.source_path))
                    )

                det_count += 1
                if det_count % det_interval == 0:
                    if modules.globals.many_faces:
                        cached_target_face = None
                        cached_many_faces = detect_many_faces_fast(temp_frame)
                    else:
                        cached_target_face = detect_one_face_fast(temp_frame)
                        cached_many_faces = None

                    # If we've lost the face for a sustained stretch while a
                    # rotation is locked (e.g. the phone was turned), unlock so
                    # the next frame re-probes all four orientations.
                    if self._rot_locked:
                        _found = bool(cached_many_faces) or (cached_target_face is not None)
                        self._rot_miss = 0 if _found else self._rot_miss + 1
                        if self._rot_miss >= 15:
                            self._rot_locked = False
                            self._rot_code = None
                            self._rot_miss = 0

                cached_faces = None
                if cached_many_faces:
                    cached_faces = cached_many_faces
                elif cached_target_face is not None:
                    cached_faces = [cached_target_face]

                # Fast detection skips the 2d106 landmark model, but BOTH the
                # mouth mask AND the adaptive full-face mask need it. Attach
                # landmarks once per detection cycle (the helper no-ops if already
                # present) so the swap fits the whole real face, not a fixed oval.
                if (modules.globals.mouth_mask
                        or getattr(modules.globals, "adaptive_face_mask", True)) and cached_faces:
                    ensure_landmarks(temp_frame, cached_faces)

                for fp in frame_processors:
                    if fp.NAME == "DLC.FACE-ENHANCER":
                        if modules.globals.fp_ui["face_enhancer"]:
                            temp_frame = fp.process_frame(
                                None, temp_frame, detected_faces=cached_faces
                            )
                    elif fp.NAME == "DLC.FACE-ENHANCER-GPEN256":
                        if modules.globals.fp_ui.get("face_enhancer_gpen256", False):
                            temp_frame = fp.process_frame(
                                None, temp_frame, detected_faces=cached_faces
                            )
                    elif fp.NAME == "DLC.FACE-ENHANCER-GPEN512":
                        if modules.globals.fp_ui.get("face_enhancer_gpen512", False):
                            temp_frame = fp.process_frame(
                                None, temp_frame, detected_faces=cached_faces
                            )
                    elif fp.NAME == "DLC.FACE-SWAPPER":
                        swapped_bboxes = []
                        if modules.globals.many_faces and cached_many_faces:
                            result = temp_frame.copy()
                            for t_face in cached_many_faces:
                                result = fp.swap_face(source_image, t_face, result)
                                if hasattr(t_face, "bbox") and t_face.bbox is not None:
                                    swapped_bboxes.append(t_face.bbox.astype(int))
                            temp_frame = result
                        elif cached_target_face is not None:
                            temp_frame = fp.swap_face(
                                source_image, cached_target_face, temp_frame
                            )
                            if (
                                hasattr(cached_target_face, "bbox")
                                and cached_target_face.bbox is not None
                            ):
                                swapped_bboxes.append(cached_target_face.bbox.astype(int))
                        temp_frame = fp.apply_post_processing(temp_frame, swapped_bboxes)
                    else:
                        temp_frame = fp.process_frame(source_image, temp_frame)
            else:
                modules.globals.target_path = None
                for fp in frame_processors:
                    if fp.NAME == "DLC.FACE-ENHANCER":
                        if modules.globals.fp_ui["face_enhancer"]:
                            temp_frame = fp.process_frame_v2(temp_frame)
                    elif fp.NAME in ("DLC.FACE-ENHANCER-GPEN256", "DLC.FACE-ENHANCER-GPEN512"):
                        fp_key = fp.NAME.split(".")[-1].lower().replace("-", "_")
                        if modules.globals.fp_ui.get(fp_key, False):
                            temp_frame = fp.process_frame_v2(temp_frame)
                    else:
                        temp_frame = fp.process_frame_v2(temp_frame)

            current_time = time.time()
            frame_count += 1
            if current_time - prev_time >= fps_update_interval:
                fps = frame_count / (current_time - prev_time)
                frame_count = 0
                prev_time = current_time

            if modules.globals.show_fps:
                cv2.putText(
                    temp_frame, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
                )

            # Mirror the live output to a phone/browser over Wi-Fi when enabled.
            # push() lazily starts the server and is fully self-guarded; the URL is
            # shown once in the status bar (update_status is Signal-based = thread-safe).
            if modules.globals.web_stream:
                try:
                    import modules.web_stream as _web_stream
                    _web_stream.push(
                        temp_frame,
                        on_url=lambda u: update_status(f"📱 Phone stream ready — open {u} in your phone's browser"),
                    )
                except Exception:
                    pass

            try:
                self._pq.put_nowait(temp_frame)
            except queue.Full:
                try:
                    self._pq.get_nowait()
                except queue.Empty:
                    pass
                try:
                    self._pq.put_nowait(temp_frame)
                except queue.Full:
                    pass


class WebcamPreviewWindow(QWidget):
    def __init__(self, camera_index: int):
        super().__init__()
        self.setWindowTitle("Live Preview")
        self.resize(PREVIEW_DEFAULT_WIDTH, PREVIEW_DEFAULT_HEIGHT)
        # Aspect of the last frame shown; drives _match_window_to_frame so the
        # window takes the video's shape instead of letterboxing it.
        self._last_ar: Optional[float] = None
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self._image_label = QLabel()
        self._image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._image_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        layout.addWidget(self._image_label, 1)

        self._cap = VideoCapturer(camera_index)
        if not self._cap.start(PREVIEW_DEFAULT_WIDTH, PREVIEW_DEFAULT_HEIGHT, 60):
            update_status("Failed to start camera")
            QTimer.singleShot(0, self.close)
            return

        camera_fps = self._cap.actual_fps
        print(
            f"[webcam] Camera running at {self._cap.actual_width}x"
            f"{self._cap.actual_height}@{camera_fps:.0f}fps"
        )

        # maxsize=1 keeps latency minimal: the capture/processing workers always
        # work on the FRESHEST frame and drop stale ones, instead of letting a
        # 2-deep backlog build up (which adds lip-sync lag in a call).
        self._capture_queue: queue.Queue = queue.Queue(maxsize=1)
        self._processed_queue: queue.Queue = queue.Queue(maxsize=1)
        self._stop_event = threading.Event()

        self._capture_worker = _CaptureWorker(
            self._cap, self._capture_queue, self._stop_event
        )
        self._processing_worker = _ProcessingWorker(
            self._capture_queue, self._processed_queue, self._stop_event, camera_fps
        )
        self._capture_worker.start()
        self._processing_worker.start()

        # Poll at ~2x camera fps so we never block but also don't burn CPU.
        poll_ms = max(1, min(16, int(500 / max(camera_fps, 1))))
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(poll_ms)

    def _match_window_to_frame(self, frame_w: int, frame_h: int) -> None:
        """Reshape the window to the video's aspect so the picture FILLS it.

        fit_image_to_size letterboxes: it scales the frame to fit inside the
        window and leaves the remainder black. With a fixed 16:9 window a
        portrait phone camera therefore showed black bars down both sides.
        Rather than crop the video, give the window the same shape as the
        frame — then "fit" and "fill" are the same thing and no bar appears.

        Runs only when the aspect actually changes (first frame, or the phone
        is rotated), so it never fights the user resizing the window.
        """
        if frame_w <= 0 or frame_h <= 0:
            return
        ar = frame_w / frame_h
        if self._last_ar is not None and abs(ar - self._last_ar) < 0.01:
            return
        self._last_ar = ar
        w = min(PREVIEW_MAX_WIDTH, max(PREVIEW_DEFAULT_WIDTH, self.width()))
        h = int(round(w / ar))
        if h > PREVIEW_MAX_HEIGHT:            # too tall (portrait) — fit by height
            h = PREVIEW_MAX_HEIGHT
            w = max(1, int(round(h * ar)))
        self.resize(w, h)

    def _tick(self) -> None:
        if self._stop_event.is_set():
            self.close()
            return
        try:
            bgr_frame = self._processed_queue.get_nowait()
        except queue.Empty:
            return
        fh, fw = bgr_frame.shape[:2]
        self._match_window_to_frame(fw, fh)
        bgr_frame = fit_image_to_size(bgr_frame, self.width(), self.height())
        self._image_label.setPixmap(_bgr_to_qpixmap(bgr_frame))

    def closeEvent(self, event) -> None:
        self._stop_event.set()
        try:
            self._timer.stop()
        except Exception:
            pass
        for worker in (self._capture_worker, self._processing_worker):
            try:
                worker.wait(2000)
            except Exception:
                pass
        try:
            self._cap.release()
        except Exception:
            pass
        global _WEBCAM_PREVIEW
        if _WEBCAM_PREVIEW is self:
            _WEBCAM_PREVIEW = None
        event.accept()


def _open_webcam_preview(camera_index: int) -> None:
    global _WEBCAM_PREVIEW
    if _WEBCAM_PREVIEW is not None:
        _WEBCAM_PREVIEW.close()
    _WEBCAM_PREVIEW = WebcamPreviewWindow(camera_index)
    _WEBCAM_PREVIEW.show()


# ─── mapper dialogs (image/video + live) ────────────────────────────────


def _make_thumb(cv2_img: np.ndarray) -> QPixmap:
    rgb = gpu_cvt_color(cv2_img, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb).resize(
        (MAPPER_PREVIEW_SIZE, MAPPER_PREVIEW_SIZE), Image.LANCZOS
    )
    return _pil_to_qpixmap(image)


class MapperDialog(QDialog):
    """Source × Target mapper for image / video processing."""

    def __init__(self, start_cb: Callable, mapping: list):
        super().__init__(_MAIN)
        self._start_cb = start_cb
        self._map = mapping
        self.setWindowTitle(_("Source x Target Mapper"))
        self.resize(POPUP_WIDTH, POPUP_HEIGHT)
        layout = QVBoxLayout(self)

        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        layout.addWidget(self._scroll, 1)

        self._status = QLabel("")
        self._status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._status)

        btn_submit = QPushButton(_("Submit"))
        btn_submit.clicked.connect(self._on_submit)
        layout.addWidget(btn_submit, alignment=Qt.AlignmentFlag.AlignCenter)

        self._rebuild()

    def set_status(self, text: str) -> None:
        self._status.setText(_(text))

    def _rebuild(self) -> None:
        body = QWidget()
        grid = QGridLayout(body)
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(10)
        for item in self._map:
            row = item["id"]
            btn = QPushButton(_("Select source image"))
            btn.setFixedWidth(200)
            btn.clicked.connect(lambda _c, n=row: self._select_source(n))
            grid.addWidget(btn, row, 0)

            src_label = QLabel(f"S-{row}")
            src_label.setFixedSize(MAPPER_PREVIEW_SIZE, MAPPER_PREVIEW_SIZE)
            src_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            src_label.setStyleSheet("border: 1px dashed #555;")
            grid.addWidget(src_label, row, 1)
            if "source" in item:
                src_label.setPixmap(_make_thumb(item["source"]["cv2"]))
                src_label.setText("")

            x_label = QLabel("×")
            x_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            grid.addWidget(x_label, row, 2)

            tgt_label = QLabel(f"T-{row}")
            tgt_label.setFixedSize(MAPPER_PREVIEW_SIZE, MAPPER_PREVIEW_SIZE)
            tgt_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            tgt_label.setStyleSheet("border: 1px solid #555;")
            grid.addWidget(tgt_label, row, 3)
            if "target" in item:
                tgt_label.setPixmap(_make_thumb(item["target"]["cv2"]))
                tgt_label.setText("")

        grid.setRowStretch(grid.rowCount(), 1)
        self._scroll.setWidget(body)

    def _select_source(self, row: int) -> None:
        path, _f = QFileDialog.getOpenFileName(
            self, _("select an source image"),
            _RECENT_SOURCE_DIR or "",
            "Images (*.png *.jpg *.jpeg *.gif *.bmp)",
        )
        if not path:
            return
        cv2_img = imread_unicode(path)
        face = get_one_face(cv2_img)
        if face is None:
            self.set_status("Face could not be detected in last upload!")
            return
        x_min, y_min, x_max, y_max = face["bbox"]
        self._map[row]["source"] = {
            "cv2": cv2_img[int(y_min):int(y_max), int(x_min):int(x_max)],
            "face": face,
        }
        self._rebuild()

    def _on_submit(self) -> None:
        if has_valid_map():
            self.accept()
            _MAIN._select_output_and_start()
        else:
            self.set_status("Atleast 1 source with target is required!")


class LiveMapperDialog(QDialog):
    """Source × Target mapper for live webcam mode."""

    def __init__(self, camera_index: int, mapping: list):
        super().__init__(_MAIN)
        self._camera_index = camera_index
        self._map = mapping
        self.setWindowTitle(_("Source x Target Mapper"))
        self.resize(POPUP_LIVE_WIDTH, POPUP_LIVE_HEIGHT)
        layout = QVBoxLayout(self)

        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        layout.addWidget(self._scroll, 1)

        self._status = QLabel("")
        self._status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._status)

        btn_row = QHBoxLayout()
        for text, slot in (
            (_("Add"), self._on_add),
            (_("Clear"), self._on_clear),
            (_("Submit"), self._on_submit),
        ):
            b = QPushButton(text)
            b.clicked.connect(slot)
            btn_row.addWidget(b)
        layout.addLayout(btn_row)

        self._rebuild()

    def set_status(self, text: str) -> None:
        self._status.setText(_(text))

    def _rebuild(self) -> None:
        body = QWidget()
        grid = QGridLayout(body)
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(10)
        for item in self._map:
            row = item["id"]
            btn_s = QPushButton(_("Select source image"))
            btn_s.setFixedWidth(200)
            btn_s.clicked.connect(lambda _c, n=row: self._select_face(n, "source"))
            grid.addWidget(btn_s, row, 0)

            src_label = QLabel(f"S-{row}")
            src_label.setFixedSize(MAPPER_PREVIEW_SIZE, MAPPER_PREVIEW_SIZE)
            src_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            src_label.setStyleSheet("border: 1px dashed #555;")
            grid.addWidget(src_label, row, 1)
            if "source" in item:
                src_label.setPixmap(_make_thumb(item["source"]["cv2"]))
                src_label.setText("")

            x_label = QLabel("×")
            x_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            grid.addWidget(x_label, row, 2)

            btn_t = QPushButton(_("Select target image"))
            btn_t.setFixedWidth(200)
            btn_t.clicked.connect(lambda _c, n=row: self._select_face(n, "target"))
            grid.addWidget(btn_t, row, 3)

            tgt_label = QLabel(f"T-{row}")
            tgt_label.setFixedSize(MAPPER_PREVIEW_SIZE, MAPPER_PREVIEW_SIZE)
            tgt_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            tgt_label.setStyleSheet("border: 1px dashed #555;")
            grid.addWidget(tgt_label, row, 4)
            if "target" in item:
                tgt_label.setPixmap(_make_thumb(item["target"]["cv2"]))
                tgt_label.setText("")

        grid.setRowStretch(grid.rowCount(), 1)
        self._scroll.setWidget(body)

    def _select_face(self, row: int, kind: str) -> None:
        path, _f = QFileDialog.getOpenFileName(
            self, _("select an source image"),
            _RECENT_SOURCE_DIR or "",
            "Images (*.png *.jpg *.jpeg *.gif *.bmp)",
        )
        if not path:
            return
        cv2_img = imread_unicode(path)
        face = get_one_face(cv2_img)
        if face is None:
            self.set_status("Face could not be detected in last upload!")
            return
        x_min, y_min, x_max, y_max = face["bbox"]
        self._map[row][kind] = {
            "cv2": cv2_img[int(y_min):int(y_max), int(x_min):int(x_max)],
            "face": face,
        }
        self._rebuild()

    def _on_add(self) -> None:
        add_blank_map()
        self._rebuild()
        self.set_status("Please provide mapping!")

    def _on_clear(self) -> None:
        for item in self._map:
            item.pop("source", None)
            item.pop("target", None)
        self._rebuild()
        self.set_status("All mappings cleared!")

    def _on_submit(self) -> None:
        if has_valid_map():
            simplify_maps()
            self.set_status("Mappings successfully submitted!")
            self.accept()
            _open_webcam_preview(self._camera_index)
        else:
            self.set_status("At least 1 source with target is required!")


def _open_mapper_dialog(start_cb: Callable, mapping: list) -> None:
    global _MAPPER
    close_mapper_window()
    _MAPPER = MapperDialog(start_cb, mapping)
    _MAPPER.show()


def _open_live_mapper_dialog(camera_index: int, mapping: list) -> None:
    global _LIVE_MAPPER
    close_mapper_window()
    _LIVE_MAPPER = LiveMapperDialog(camera_index, mapping)
    _LIVE_MAPPER.show()


def close_mapper_window() -> None:
    global _MAPPER, _LIVE_MAPPER
    if _MAPPER is not None:
        _MAPPER.close()
        _MAPPER = None
    if _LIVE_MAPPER is not None:
        _LIVE_MAPPER.close()
        _LIVE_MAPPER = None


# ─── entry point ─────────────────────────────────────────────────────────


class _Window:
    """Thin wrapper exposing .mainloop() for core.py compatibility."""

    def __init__(self, app: QApplication, main_window: MainWindow):
        self._app = app
        self._main = main_window

    def mainloop(self) -> None:
        self._main.show()
        self._app.exec()


def init(
    start: Callable[[], None], destroy: Callable[[], None], lang: str
) -> _Window:
    global _APP, _MAIN, _PREVIEW, _LANG, _BRIDGE

    _LANG = LanguageManager(lang)
    if QApplication.instance() is None:
        _APP = QApplication(sys.argv)
    else:
        _APP = QApplication.instance()
    _APP.setStyleSheet(QSS)

    _BRIDGE = _UIBridge()
    _MAIN = MainWindow(start, destroy)
    _PREVIEW = PreviewWindow()

    # Route status updates onto the UI thread regardless of caller.
    _BRIDGE.statusChanged.connect(_MAIN.set_status)

    return _Window(_APP, _MAIN)
