# Live Face Swap → OBS → WhatsApp — complete setup

How to get the face swap into a WhatsApp (or Zoom / Meet / Teams) video call,
and why each setting is what it is. Written for this PC, after the fixes we made.

---

## 0. How the video actually travels

```
   camera  →  Live Face Swap  →  web stream  →  OBS browser source
 (phone or      (does the        (MJPEG on         (Browser 2)
  laptop)        swapping)       port 8765)              │
                                                         ▼
                                                    OBS canvas
                                                   (1280 × 720)
                                                         │
                                                         ▼
                                              OBS Virtual Camera
                                                         │
                                                         ▼
                                                    WhatsApp
```

**The golden rule: every stage must be the same shape (16:9).**
Almost every "small picture / black bars" problem is one stage having a
different aspect ratio from the next, so something gets letterboxed.

---

## 1. One-time OBS setup

Do this once. It sticks as long as you **close OBS properly** (see §5).

### 1a. Canvas — Settings → Video
| Field | Value |
|---|---|
| Base (Canvas) Resolution | **1280x720** |
| Output (Scaled) Resolution | **1280x720** |
| FPS | 30 |

> If the fields are greyed out with *"Video output is currently active"*,
> click **Stop Virtual Camera** first, change them, then start it again.

### 1b. The source — Sources → Browser (or Browser 2) → Properties
| Field | Value |
|---|---|
| URL | `http://127.0.0.1:8765/?k=YOUR_PRIVATE_STREAM_KEY` |
| Width | **1280** |
| Height | **720** |
| Custom CSS | *(replace everything with the block below)* |

```css
html,body{background:transparent!important;margin:0!important;padding:0!important;overflow:hidden!important;width:100%!important;height:100%!important}
img{position:absolute!important;inset:0!important;width:100%!important;height:100%!important;object-fit:cover!important}
#fs{display:none!important}
```

What each part does:
- `object-fit:cover` — the video **fills** the source instead of being fitted
  inside it. This is what removes the black frame.
- `background:transparent` — any leftover area is transparent, never black.
- `#fs{display:none}` — hides the fullscreen button (that button is for the
  phone view; it must not appear in the call).

> ⚠️ OBS resets Custom CSS to its default if you ever click **Defaults** in
> this dialog. If the black frame comes back, this is the first thing to check.

### 1c. Make the source fill the canvas
Click the source in the preview (or in the Sources list) → press **Ctrl + F**
("Fit to screen"). The picture should now touch all four edges.

### 1d. Start the output
**Controls → Start Virtual Camera.** It must say *Stop Virtual Camera* while
you are in a call.

---

## 2. Connect WhatsApp

1. Open **WhatsApp → Settings → Video** (or the ⚙ during a call).
2. Set **Camera → OBS Virtual Camera**.
3. If you are already in a call, turn the camera **off and on** — WhatsApp
   locks the video size when the call starts and won't pick up changes.

Same procedure for Zoom / Google Meet / Teams — just pick *OBS Virtual Camera*.

---

## 3. Every time you want to use it (order matters)

1. **Start the swap app** — `START_HOST.bat` (or `START.bat`).
2. **Select a face** → click **Live**. The console/status bar prints the
   stream URL. The swap must be live *before* OBS can show anything.
3. **Open OBS** → click **Start Virtual Camera**.
   - If the source is blank/black: right-click it → **Refresh**.
4. **Open WhatsApp** and make the call.

If you open OBS before the swap is live, the browser source loads an error
page — just hit **Refresh** on the source once the swap is running.

---

## 4. Camera choice — this matters more than any setting

| Source camera | Shape | Result |
|---|---|---|
| **Laptop "Integrated Camera"** | 16:9 already | ✅ Best. Fills everything, nothing cropped, no setup |
| **Phone held sideways (landscape)** | 16:9 | ✅ Also perfect, and higher quality with HD mode on |
| Phone held upright (portrait) | 3:4 | ⚠️ Must be cropped to fit 16:9 — you lose the top and bottom |

For DroidCam: turn on **HD mode** in the phone app for 720p instead of 480p,
and hold/mount the phone **sideways**.

---

## 5. Things that will bite you (all of these actually happened)

**The black frame came back**
→ Browser source Custom CSS was reset to OBS's default. Re-paste the CSS in §1b.

**Small picture on the other side**
→ Some stage isn't 16:9. Check all three: canvas (§1a), source Width/Height
(§1b), and that the source fills the canvas (§1c).

**Settings don't survive a restart**
→ OBS writes its config **when it exits**. Always close it with **File → Exit**.
Force-closing (or killing it) throws the changes away, and you'll also get the
*"OBS Studio did not properly shut down"* dialog next launch.

**"OBS is already running"**
→ Click **Cancel**. Two instances fight over the config and the virtual camera.
Only ever run one.

**Virtual Camera isn't on after restart**
→ It doesn't auto-start on this install. Click **Start Virtual Camera** each time.

**WhatsApp still shows the old/raw camera**
→ It's using the phone camera directly instead of OBS. Re-pick **OBS Virtual
Camera** in WhatsApp, and toggle the camera off/on in the call.

---

## 6. What can NOT be fixed (honest limits)

**The far side still sees black above and below on their phone.**
WhatsApp Desktop always transmits a **16:9 landscape** frame. A phone screen is
tall, so a wide video is shown fitted, with bars. Measured three times:
a 9:16 feed predicted 187px wide and measured 186px; a 3:4 feed predicted 332px
and measured 332px. Nothing on this PC changes that.

What genuinely helps:
- **Ask them to rotate their phone sideways** → fills their screen completely.
- They can **pinch-zoom once**; it stays zoomed for the call.
- Fill the frame with yourself so it reads as a normal close-up.

True full-screen portrait would require the call to be placed *from a phone*,
which can't be done with a PC virtual camera.

---

## 7. Quick reference

| Item | Value |
|---|---|
| Stream URL (OBS) | `http://127.0.0.1:8765/?k=YOUR_PRIVATE_STREAM_KEY` |
| Stream URL (phone) | `http://<your-LAN-IP>:8765/?k=YOUR_PRIVATE_STREAM_KEY` |
| Access key | Use your locally configured private key; never commit it to GitHub. |
| Port | 8765 |
| OBS canvas | 1280 × 720 |
| Browser source | 1280 × 720 + the CSS in §1b |
| WhatsApp camera | OBS Virtual Camera |
