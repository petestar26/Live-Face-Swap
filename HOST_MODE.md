# Host Mode — let others watch, keep your code private

This is the **real protection**: the app and the AI models stay on **your** computer.
Anyone you share with only sees the **live video output** in their browser — there is
nothing on their device to copy. They watch; you keep the code.

The stream is protected by a secret **access key**, so only people you send the exact
link to can view it.

---

## The access key (automatic)

When you tick **“Stream to Phone (Web)”** in the app and click **Live**, the console
prints a private link that ends in `?k=XXXXXXXX`:

```
[web_stream] Private stream live — share ONLY this link: http://192.168.1.42:8765/?k=Ab3xK9pQ
```

- Only someone with the **full link including `?k=...`** can watch. Without the key they
  get **“Access denied.”**
- A **new random key** is generated each launch by default.
- Want a **stable link** that doesn’t change between runs? Set your own key (below).
- Want it open with **no key** (trusted home Wi‑Fi only)? Set `KP_STREAM_NOAUTH=1`.

---

## A) Same Wi‑Fi (easiest)

1. Host PC and the viewer’s device are on the **same Wi‑Fi**.
2. In the app, tick **“Stream to Phone (Web)”**, then click **Live**.
3. Copy the printed link (e.g. `http://192.168.1.42:8765/?k=Ab3xK9pQ`) to the viewer.
4. **First time only:** if it won’t load, allow port **8765** through Windows Firewall
   (approve the pop‑up for Python, or add an inbound rule for TCP 8765).

---

## B) Over the internet (viewer is somewhere else) — secure tunnel

Use a free tunnel so you don’t have to open router/firewall ports. The tunnel also adds
**HTTPS encryption** on top of your access key.

### Cloudflare Tunnel (free, no account needed for quick tunnels)
1. Download **cloudflared** (one small .exe):
   https://github.com/cloudflare/cloudflared/releases → `cloudflared-windows-amd64.exe`
2. Start the app with a **fixed key** so the link is stable — use **START_HOST.bat**
   (edit the key at the top), or set it yourself (see below). Tick “Stream to Phone”, go Live.
3. Open a **separate** console in the folder with cloudflared and run:
   ```
   cloudflared tunnel --url http://localhost:8765
   ```
4. It prints a public address like `https://random-words.trycloudflare.com`.
5. Give the viewer that address **plus your key**:
   ```
   https://random-words.trycloudflare.com/?k=YOURKEY
   ```
   (Take the `?k=YOURKEY` part from the app’s printed link and append it to the tunnel URL.)

### Alternative: ngrok (needs a free account)
```
ngrok http 8765
```
Then share the printed `https://….ngrok-free.app/?k=YOURKEY`.

---

## Setting a fixed key (stable link)

In a console **before** launching:
```
set KP_STREAM_TOKEN=choose-a-long-secret-no-spaces
START.bat
```
…or just use **START_HOST.bat**, which sets the key for you (edit the value at the top of
that file to your own secret).

To turn the key **off** (open on LAN):
```
set KP_STREAM_NOAUTH=1
```

---

## Security & good‑practice notes

- **Share the `?k=` link only with people you trust.** Anyone who has it can watch while
  you’re live. Treat it like a password.
- **Key + tunnel = private and encrypted.** The key controls *who* can watch; the tunnel’s
  HTTPS protects the video in transit.
- **To stop sharing:** close the app (the stream ends) or press **Ctrl+C** in the tunnel window.
- **Rotate the key** any time by restarting (new random key) or changing `KP_STREAM_TOKEN`.
- **Consent & ethics:** only stream face swaps with the **consent** of the people involved.
  Don’t use it to impersonate real, identifiable people. Keep it legal and honest.
