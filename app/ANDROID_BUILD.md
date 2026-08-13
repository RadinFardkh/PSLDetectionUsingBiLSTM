# PSL AI — ARM64 Android Build

This package contains the fixed PSL AI Kivy application configured for an
ARM64 Android APK build.

## Required model files

The source ZIP used to create this package does NOT contain the trained model
files. Before building, place these files here:

    psl_app/models/sign_model.tflite
    psl_app/models/class_map.json

`app_config.py` already points the application at that directory.

## Build environment

Buildozer is intended to be run on Linux or WSL2. Native Windows builds are
not the recommended Buildozer workflow.

Install the required system packages for your Linux distribution, then:

    python3 -m venv .venv
    source .venv/bin/activate
    pip install --upgrade pip setuptools wheel
    pip install buildozer cython

From this directory:

    buildozer android debug

The generated APK will normally appear under:

    bin/

For a release build:

    buildozer android release

## ARM64

The included `buildozer.spec` uses:

    android.arch = arm64-v8a

This avoids building 32-bit ARM and keeps the resulting APK smaller.

## Important dependency note

This project uses:

- Kivy
- OpenCV
- MediaPipe
- NumPy
- TFLite Runtime

These include native Android components. If your installed Buildozer /
python-for-android version does not provide compatible recipes for one of
these packages, the build will stop at dependency compilation. In that case,
use a compatible python-for-android recipe/version rather than replacing the
native dependency with a desktop-only pip wheel.

## Camera

The app requests the Android CAMERA permission at runtime and also declares
it in `buildozer.spec`.

## App entry point

The Android entry point is:

    psl_app/main.py
