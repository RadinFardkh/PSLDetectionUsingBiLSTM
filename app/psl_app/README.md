# Persian Sign Language — Kivy App

## Setup

```bash
pip install kivy opencv-python mediapipe numpy tensorflow
```

(If you're deploying somewhere `tensorflow` is too heavy, install
`tflite-runtime` instead — `engine.py` already falls back to it
automatically if `tensorflow` isn't importable.)

## Files you need to add before running

1. **Model**: place your `.tflite` model into `models/` as
   `sign_model.tflite`, and `class_map.json` into `models/` as well.
   Both filenames are set in `app_config.py` (`TFLITE_MODEL_NAME`,
   `CLASS_MAP_NAME`) — change them there if your files are named
   differently, or just overwrite the files in place.

## Running

```bash
python main.py
```

## Hot-swapping the model

Overwrite `models/sign_model.tflite` and `models/class_map.json`,
then call:

```python
app.camera_screen.engine.reload()
```

from anywhere with access to the running `App` instance (e.g. wire it
to a debug key or long-press). It safely reloads the interpreter and
resets the temporal buffers so old and new feature dimensions never
mix mid-sequence.

## Performance notes already built in

- MediaPipe still runs every frame (needed for the camera-mirror /
  skeleton), but **TFLite inference itself only runs after "Start" is
  pressed**, and even then only every `INFER_EVERY_N_FRAMES` frames
  (default 3), matching the original script's throttling.
- The skeleton overlay is only computed and drawn when **Developer
  mode** is on — with it off, `canvas.clear()` means zero per-frame
  vector-graphics cost, not just an invisible layer.
- Camera frames are blitted directly into a persistent GL `Texture`
  buffer (`blit_buffer`) instead of round-tripping through PNG/JPEG
  encoding, which is the usual Kivy-camera-preview bottleneck.
- `model_complexity=1` and the `.55` MediaPipe confidence thresholds
  are unchanged from your original script; dropping to
  `model_complexity=0` is the next lever if you need more headroom on
  low-end hardware, at some accuracy cost.
- Further optional levers if you still need more speed: downscale the
  camera capture resolution (`cap.set(cv2.CAP_PROP_FRAME_WIDTH/HEIGHT,
  ...)`), or drop the Clock interval below 30fps (e.g. `1/20.0`).

## Notes on things I made judgment calls on

- **Screen transition** ("hovers to the top… new window comes in from
  below"): implemented via Kivy's `ScreenManager` `SlideTransition(direction="up")`,
  which is the idiomatic way to do this rather than manually animating
  widget `y` positions — same visual effect, far less code to maintain.
- **Camera flip button**: toggles between capture index `0` and `1`
  since that's the common front/rear convention on most systems; if
  your rear camera isn't at index `1`, adjust `camera_index` values in
  `camera_screen.py`.
- I could not run or visually verify this app in the sandbox I wrote
  it in — there's no camera, no display server, and no Kivy install
  available here. Please treat this as a strong first pass and expect
  to do a debugging pass on your own machine, especially around
  camera indices, which are very environment dependent.
