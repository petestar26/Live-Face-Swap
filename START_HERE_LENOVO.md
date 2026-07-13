# START HERE — K_P Face Swap on Lenovo LOQ 15AHP10

Your machine: **Ryzen 7 250 · RTX 5050 8GB (Blackwell) · 32GB DDR5 · 512GB NVMe · Win 11**.
This is a strong setup. Follow this once and you're live. Full reference: `SETUP_LIVE.md`.

## 0. Do these 3 machine-specific things first
1. **Used machine → clean-reinstall Windows 11** (97GB was the previous owner's). Security + a fresh
   start. Your license is digital and reactivates automatically.
2. **Hybrid graphics (RTX 5050 + Radeon 780M):** Windows → *Settings → Display → Graphics* → add
   `python.exe` → **High performance (NVIDIA)**. If Lenovo Vantage has a **dGPU/Hybrid** setting,
   pick dedicated dGPU when plugged in.
3. **Blackwell is new → newest NVIDIA driver (570+) + CUDA 12.8+.** Don't use an older
   `onnxruntime-gpu` than the pinned one — it won't run on a 50-series GPU.

## 1. Install
```bat
:: prerequisites: Python 3.10 (add to PATH), Git, VC++ x64 Redistributable, ffmpeg
git clone https://github.com/hacksider/Deep-Live-Cam.git
cd Deep-Live-Cam
git apply path\to\deeplivecam_quality.patch    :: my changes + the K_P Face Swap rebrand
python -m venv venv && venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 2. The one check that proves the GPU works
```bat
python -c "import onnxruntime as ort; print(ort.get_available_providers())"
```
Must list **`CUDAExecutionProvider`**. If not, fix drivers/CUDA before going further (see
`SETUP_LIVE.md` §9). Anything else means you'd be on CPU = 2 FPS.

## 3. Models → `models/`
- `inswapper_128_fp16.onnx` — https://huggingface.co/hacksider/deep-live-cam/resolve/main/inswapper_128_fp16.onnx?download=true
- `GFPGANv1.4.pth` — https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth

## 4. Run
```bat
python run.py --execution-provider cuda
```

## 5. Recommended preset for your RTX 5050 (8GB)
You have the VRAM for full quality. Set:

| Control | Value | Why |
|---|---|---|
| **Match Face Color** | ON, strength **100%** | removes the color/lighting seam |
| **Poisson Blend** | ON | smooths the face-edge boundary |
| **Face Enhancer** | **GPEN-512** | restores detail (8GB handles it) |
| **Sharpness** | ~0.5 | light crispness |
| **Stream to Phone (Web)** | ON if you want to watch on your phone | URL appears in the status bar |
| Mouth Mask | off | unless you need your real mouth |
| Many faces | off | single face = best FPS/VRAM |
| Webcam | 720p | smooth live |

**Expected:** ~30–40 FPS (Match Color + Poisson), ~20–28 FPS with GPEN-512 on.
**If it stutters:** Enhancer → **GPEN-256** first, then drop webcam to 720p.

## 6. Watch it on your phone
Tick **Stream to Phone (Web)** → click **Live** → the status bar shows
`http://<your-ip>:8765/`. Open that in your phone's browser (same Wi-Fi). Allow port **8765**
through Windows Firewall the first time.

> Use only with consent; follow your local laws and each platform's rules.
