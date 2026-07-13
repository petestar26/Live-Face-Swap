# K_P Face Swap — Live Face Swap Setup (Desktop)

A complete, no-surprises guide to running **real-time webcam face swap** on a desktop PC,
with the quality features (Match Face Color, Poisson Blend, face enhancer) enabled.

> *K_P Face Swap is built on the open-source Deep-Live-Cam engine; the clone URL below still
> points at the upstream repository.*

> This guide targets **Windows + NVIDIA GPU** first (the fastest, most common setup),
> with sections for **AMD/Intel (DirectML)** and **Linux** at the end.

---

## 0. Hardware — what you need (read this first)

Live swap is GPU-bound. Your graphics card decides whether this is smooth or a slideshow.

| Tier | GPU | Expected live FPS (1 face, 128 model) |
|------|-----|----------------------------------------|
| **Recommended** | NVIDIA RTX 4070 / 4080 / 4090, RTX 3080+ | 30–60+ FPS |
| **Good** | NVIDIA RTX 3060 (12 GB) / 4060 / 2070 | 20–30 FPS |
| **Minimum** | NVIDIA GTX 1660 / RTX 2060 (6 GB) | 12–20 FPS |
| AMD (Windows) | RX 6700 XT+ via **DirectML** | 10–20 FPS (slower than CUDA) |
| CPU only | any | 1–3 FPS — **not usable for live** |

**Practical minimums:** NVIDIA card with **6 GB+ VRAM** (8 GB+ recommended), 16 GB system RAM,
a recent CPU, and an SSD. A real webcam (or OBS Virtual Camera) for input.

---

## 1. Install prerequisites (Windows)

1. **Python 3.10** (64-bit) — https://www.python.org/downloads/release/python-31011/
   - During install: tick **"Add Python to PATH"**.
   - ⚠️ Avoid 3.12/3.13 — `insightface` / `onnxruntime` wheels are most reliable on 3.10.
2. **Git** — https://git-scm.com/download/win
3. **NVIDIA driver** — latest Game Ready or Studio driver: https://www.nvidia.com/Download/index.aspx
4. **Visual C++ Redistributable (x64)** — https://aka.ms/vs/17/release/vc_redist.x64.exe
   (insightface needs it; skipping this is the #1 cause of import errors.)
5. **ffmpeg** — https://www.gyan.dev/ffmpeg/builds/ → add its `bin` folder to PATH
   (needed for video files / audio; not strictly required for pure webcam live).

For NVIDIA acceleration, `onnxruntime-gpu` (installed in step 4) needs the **CUDA 12.x runtime
+ cuDNN 9.x**. The simplest reliable route: install the **CUDA Toolkit 12.x** from
https://developer.nvidia.com/cuda-downloads and **cuDNN 9** from
https://developer.nvidia.com/cudnn. (You can skip these if you instead use DirectML — see §8.)

---

## 2. Get the code

```bat
git clone https://github.com/hacksider/Deep-Live-Cam.git
cd Deep-Live-Cam
```

## 3. Create an isolated environment

```bat
python -m venv venv
venv\Scripts\activate
python -m pip install --upgrade pip
```

You should now see `(venv)` at the start of your prompt. Do everything below inside it.

## 4. Install dependencies

```bat
pip install -r requirements.txt
```

`requirements.txt` already pulls **`onnxruntime-gpu`** on non-Mac systems. **Verify the GPU is
visible to ONNX Runtime** — this single check saves hours of debugging:

```bat
python -c "import onnxruntime as ort; print(ort.get_available_providers())"
```

✅ You want to see **`CUDAExecutionProvider`** in the list (or `DmlExecutionProvider` for AMD/Intel).
If you only see `CPUExecutionProvider`, jump to **Troubleshooting → CUDA not detected** (§9).

## 5. Download the models

Put these two files in the **`models/`** folder:

- **Swap model** — `inswapper_128_fp16.onnx`
  https://huggingface.co/hacksider/deep-live-cam/resolve/main/inswapper_128_fp16.onnx?download=true
- **Face enhancer** — `GFPGANv1.4.pth`
  https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth

> The app also auto-downloads `inswapper_128.onnx` on first run if missing. GPEN enhancer
> weights download automatically the first time you select them in the UI.

---

## 6. First launch

**NVIDIA:**
```bat
python run.py --execution-provider cuda
```
(There's a convenience script too: `run-cuda.bat`.)

The GUI opens. On first run it may download/optimize the model — wait for
"Face swapper model loaded successfully." in the console.

---

## 7. Do a live webcam swap (the actual workflow)

1. **Select Source** → pick a clear, front-facing photo of the face you want to wear.
   (Good lighting, both eyes visible, no heavy occlusion = best result.)
2. Click **Live** (webcam preview window opens).
3. Look at the camera. The swap runs in real time.

### Turn on the quality features (recommended order)
In the controls panel:

- ✅ **Match Face Color** *(new)* — matches the swapped face's skin tone/lighting to **you**,
  so there's no color seam. Computed on the face region only for an accurate match.
- 🎚️ **Color Match slider** *(new, Refinement card)* — dials that match from 0% (raw swap) to
  100% (full match). Start at 100%; lower it if the target lighting is strongly colored and you
  want to keep some of it.
- ✅ **Poisson Blend** — smooths the face-edge boundary into your skin.
- **Face Enhancer → GPEN-512** (or GFPGAN) — restores fine detail/sharpness.
- **Refinement sliders** — small **Sharpness**, keep **Mouth Mask** off unless you need your
  real mouth to show through.

> **For your stated goal (removing the color/lighting seam):**
> **Match Face Color + Poisson Blend** is the combination. Match Face Color removes the *tone*
> difference; Poisson Blend removes the *edge*. Enable both.

Tip: enabling the **Face Enhancer** costs the most FPS. If live gets choppy, set it to **None**
or **GPEN-256** while live, and use GPEN-512 only when exporting a video.

---

## 8. Use the swap in Zoom / Meet / Teams / OBS

The live window is local. To pipe it into a meeting app, use a **virtual camera**:

1. Install **OBS Studio** (https://obsproject.com) → it provides "OBS Virtual Camera".
2. In OBS: add a **Window Capture** (or **Display Capture**) of the Deep-Live-Cam live window.
3. Click **Start Virtual Camera** in OBS.
4. In Zoom/Meet/Teams, choose **"OBS Virtual Camera"** as your webcam.

---

## 8b. Watch the swap on your phone 📱

**Important:** a phone **cannot run the swap itself** — it has no CUDA GPU and the models are
too heavy for real-time. The realistic and reliable setup is: **the desktop does the work, the
phone just watches** the result over your Wi-Fi. Two ways:

### Easiest — built-in web stream (no extra software)
1. Make sure the desktop and phone are on the **same Wi-Fi**.
2. In the app, tick **"Stream to Phone (Web)"**, then click **Live**.
3. The console prints a URL like `http://192.168.1.42:8765/`.
4. Open that URL in your phone's browser (Safari/Chrome). Tap the image for fullscreen.

That's a live, low-latency MJPEG view of the swap on your phone. (It's view-only — controls
stay on the desktop. If the page doesn't load, allow port **8765** through the desktop firewall.)

### For phone video calls (WhatsApp / Instagram / etc.)
The web view above is great for *watching*, but phone apps only accept the phone's own camera.
To get the swap **into** a phone call you'd route the desktop's virtual camera (§8) into a phone
via a camera-bridge app (e.g. **DroidCam/Iriun** in reverse, or **OBS + NDI**), or simply join
the call **from the desktop** instead — which is far simpler and higher quality.

> Bottom line: use a desktop with a good NVIDIA GPU as the engine; use the phone as a **screen**
> (web stream) or join calls from the desktop. Trying to run the model on the phone is not viable.

---

## 9. Troubleshooting

**"CUDA not detected" / only `CPUExecutionProvider` shows / runs at ~2 FPS**
- Update the NVIDIA driver.
- Ensure **CUDA 12.x** + **cuDNN 9.x** are installed and on PATH.
- Confirm GPU `onnxruntime`: `pip uninstall onnxruntime onnxruntime-gpu -y && pip install onnxruntime-gpu`.
- Re-check: `python -c "import onnxruntime as ort; print(ort.get_available_providers())"`.
- If CUDA is a hassle, use **DirectML** instead (works on any DX12 GPU incl. AMD/Intel):
  `pip install onnxruntime-directml` then `python run.py --execution-provider dml`.

**`ImportError` / DLL load failed (insightface)** → install the **Visual C++ x64 Redistributable** (§1.4).

**No face detected** → use a clearer, larger, front-facing source photo; improve your lighting.

**Out of memory (VRAM)** → close other GPU apps; set Face Enhancer to None/GPEN-256; lower webcam resolution.

**Choppy / low FPS** → Face Enhancer = None while live; ensure CUDA (not CPU); close background apps;
lower the webcam resolution to 720p.

**Colors look washed out / wrong tone** → that's exactly what **Match Face Color** fixes — enable it.

---

## 10. Other platforms (quick notes)

**AMD / Intel GPU (Windows, DirectML):**
```bat
pip install onnxruntime-directml
python run.py --execution-provider dml     :: or run-directml.bat
```

**Linux (NVIDIA):**
```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python run.py --execution-provider cuda
```
Install the matching NVIDIA driver + CUDA 12.x + cuDNN 9.x from your distro / NVIDIA.

---

## 11. Responsible use

This is a deepfake tool. Only swap faces with **consent**, follow the laws where you live, and
respect the platform rules of any service you stream into. Don't use it to deceive, harass, or
impersonate. The project ships an NSFW filter and a disclaimer — keep them intact.
