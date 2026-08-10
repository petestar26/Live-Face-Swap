# --- START OF FILE globals.py ---

import os
from typing import List, Dict, Any

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
WORKFLOW_DIR = os.path.join(ROOT_DIR, "workflow")

file_types = [
    ("Image", ("*.png", "*.jpg", "*.jpeg", "*.gif", "*.bmp")),
    ("Video", ("*.mp4", "*.mkv")),
]

# Face Mapping Data
source_target_map: List[Dict[str, Any]] = [] # Stores detailed map for image/video processing
simple_map: Dict[str, Any] = {}             # Stores simplified map (embeddings/faces) for live/simple mode

# Paths
source_path: str | None = None
target_path: str | None = None
output_path: str | None = None
# Optional multi-photo source: a synthetic Face whose embedding is the average
# of several photos of the SAME person (different angles/sizes) for a more
# accurate, robust swap. None = use the single source_path image as normal.
source_face_override = None

# Processing Options
frame_processors: List[str] = []
keep_fps: bool = True
keep_audio: bool = True
keep_frames: bool = False
many_faces: bool = False         # Process all detected faces with default source
map_faces: bool = False          # Use source_target_map or simple_map for specific swaps
poisson_blend: bool = False      # Enable Poisson Blending for smoother face swaps
adaptive_face_mask: bool = True  # Fit the blend to the ACTUAL detected face outline
                                 # (2d106 landmarks) instead of a fixed oval, so the
                                 # swap covers the whole face regardless of how the
                                 # source/target face sizes/shapes differ.
color_correction: bool = False   # Enable color correction (implementation specific)
face_color_match: bool = False   # Match swapped-face color/lighting to the target face (LAB transfer)
face_color_match_strength: float = 1.0  # 0..1 blend of the color-matched face over the raw swap
# Keep the SOURCE photo's real skin colour instead of repainting the swap with
# the target's. Normal colour-match rewrites all three LAB channels toward the
# target, which hides the seam but also replaces the source person's skin tone
# with the camera person's. With this on, only the LIGHTING (LAB L) follows the
# room while the chroma (a,b) — the actual colour identity of the provided
# picture — is preserved. False = previous behaviour, byte-for-byte unchanged.
source_skin_tone: bool = False
web_stream: bool = False         # Stream the live output over HTTP so a phone/browser can view it
nsfw_filter: bool = False

# Video Output Options
video_encoder: str | None = None
video_quality: int | None = None # Typically a CRF value or bitrate

# Live Mode Options
live_mirror: bool = False
live_resizable: bool = True
camera_input_combobox: Any | None = None # Placeholder for UI element if needed
webcam_preview_running: bool = False
show_fps: bool = False

# System Configuration
max_memory: int | None = None        # Memory limit in GB? (Needs clarification)
execution_providers: List[str] = []  # e.g., ['CUDAExecutionProvider', 'CPUExecutionProvider']
execution_threads: int | None = None # Number of threads for CPU execution
headless: bool | None = None         # Run without UI?
log_level: str = "error"             # Logging level (e.g., 'debug', 'info', 'warning', 'error')

# Face Processor UI Toggles (Example)
fp_ui: Dict[str, bool] = {"face_enhancer": False, "face_enhancer_gpen256": False, "face_enhancer_gpen512": False}

# Face Swapper Specific Options
face_swapper_enabled: bool = True # General toggle for the swapper processor
opacity: float = 1.0              # Blend factor for the swapped face (0.0-1.0)
sharpness: float = 0.0            # Sharpness enhancement for swapped face (0.0-1.0+)

# Mouth Mask Options
mouth_mask: bool = False           # Enable mouth area masking/pasting
show_mouth_mask_box: bool = False  # Visualize the mouth mask area (for debugging)
mask_feather_ratio: int = 12       # Denominator for feathering calculation (higher = smaller feather)
mask_down_size: float = 0.1        # Expansion factor for lower lip mask (relative)
mask_size: float = 1.0             # Expansion factor for upper lip mask (relative)
mouth_mask_size: float = 0.0       # Mouth mask size (0-100; 0=off, 100=mouth to chin)

# Teeth Effect (optional cosmetic overlay on the mouth's teeth during the swap).
# "None" = disabled (default; the pipeline is byte-for-byte unchanged). Other
# values: "White", "Add Tooth", "Gold", "Silver", "Image…". Fully
# self-contained — see _apply_teeth_effect in face_swapper.py; a strict no-op
# unless a style is set.
teeth_effect: str = "None"
# Path to a user-chosen teeth/grillz image, used only when teeth_effect starts
# with "Image". Empty = none selected (the Image mode then no-ops).
teeth_image_path: str = ""

# --- START: Added for Frame Interpolation ---
enable_interpolation: bool = True # Toggle temporal smoothing
interpolation_weight: float = 0  # Blend weight for current frame (0.0-1.0). Lower=smoother.
# --- END: Added for Frame Interpolation ---

# --- END OF FILE globals.py ---

import threading
dml_lock = threading.Lock()
