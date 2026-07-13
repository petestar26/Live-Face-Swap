# Live Face Swap — by Peter

Real-time webcam face swap for Windows + NVIDIA GPU, built on the open-source
[Deep-Live-Cam](https://github.com/hacksider/Deep-Live-Cam) engine.

This fork adds a set of quality and usability improvements on top of the base project:

- **Full-face coverage** — the swap covers the whole detected face instead of a
  small cropped patch, even when the source and target faces differ in size/shape.
- **Cleaner blending** — reduces the blotchy two-tone look from the raw 128px
  model (local-highlight flattening + global relight-to-target), plus feathered
  sharpening and an adaptive face-mask blend.
- **Match Face Color** — matches the swapped face's skin tone/lighting to yours
  so there's no color seam.
- **Teeth Effect (optional)** — choose *None / White / Add Tooth / Gold / Silver*
  or overlay your own grillz image. Only applied when the mouth is open; a strict
  no-op when set to *None*.
- **Phone web stream** — view the live result on your phone's browser over Wi‑Fi
  (MJPEG over HTTP, view-only).

## Setup

See **[SETUP_LIVE.md](SETUP_LIVE.md)** for a complete, step-by-step guide
(Python 3.10, NVIDIA driver + CUDA/cuDNN, virtual environment, models, and first launch).

Quick start (Windows + NVIDIA):

```bat
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python run.py --execution-provider cuda
```

## Responsible use

This is a deepfake tool. Only swap faces **with consent**, follow the laws where
you live, and respect the rules of any platform you stream into. Do not use it to
deceive, harass, or impersonate anyone. Keep the built-in NSFW filter and
disclaimer intact.

## Credits

- Engine: [Deep-Live-Cam](https://github.com/hacksider/Deep-Live-Cam) and its contributors.
- Models: `inswapper` (InsightFace), GFPGAN, GPEN.
