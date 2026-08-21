# =============================================================================
# buildozer.spec — PSL AI (Persian Sign Language Recognizer)
#
# Android build configuration for the Kivy-based PSL AI application.
#
# Project layout:
#
#   app/
#   └── psl_app/
#       ├── main.py
#       ├── app_config.py
#       ├── engine.py
#       ├── screens/
#       ├── models/
#       │   ├── sign_model.tflite
#       │   ├── class_map.json
#       │   ├── palm_detection_full.tflite
#       │   └── hand_landmark_full.tflite
#       └── buildozer.spec
#
# Buildozer is executed from app/psl_app, so source.dir = .
#
# IMPORTANT:
#   MediaPipe is intentionally NOT included in requirements.
#   The Android engine uses the raw TFLite hand-detection models instead.
#
#   Full TensorFlow is also intentionally NOT included.
#   tflite-runtime is used for inference on Android.
#
# =============================================================================

[app]

# -----------------------------------------------------------------------------
# Application identity
# -----------------------------------------------------------------------------

title = PSL AI

package.name = pslai
package.domain = org.radinfardkh

version = 1.0.0
android.numeric_version = 1

# -----------------------------------------------------------------------------
# Source layout
# -----------------------------------------------------------------------------

# main.py is in the same directory as this buildozer.spec.
source.dir = .

# Python/UI resources are bundled normally.
#
# Model files and class_map.json are added explicitly with android.add_assets
# below, which keeps the packaging configuration deterministic.
source.include_exts = py,png,jpg,jpeg,kv,atlas

# Do not package development/build artifacts.
source.exclude_patterns = \
    tests/*, \
    *.spec, \
    __pycache__/*, \
    *.pyc, \
    *.pyo, \
    .git/*, \
    .github/*, \
    .buildozer/*, \
    bin/*, \
    venv/*, \
    .venv/*

# -----------------------------------------------------------------------------
# Python requirements
# -----------------------------------------------------------------------------
#
# Python 3.12 is intentionally selected because the stable python-for-android
# branch is intended for Python <= 3.12. The development branch has moved on
# to newer Python versions and should not be used for this application unless
# the whole dependency stack is revalidated.
#
# Kivy 2.3.1 is used because it is the current 2.3.x release and is also
# supported by recent p4a releases.
#
# DO NOT add:
#   mediapipe
#   tensorflow
#   opencv-python
#
# Use the p4a recipes:
#   opencv
#   numpy
#   tflite-runtime
#
requirements = python3==3.12.9,hostpython3==3.12.9,kivy==2.3.1,numpy,opencv,tflite-runtime

# -----------------------------------------------------------------------------
# Android assets
# -----------------------------------------------------------------------------
#
# These are required by engine.py at runtime.
#
# sign_model.tflite:
#   BiLSTM sign-recognition model.
#
# class_map.json:
#   Maps model class indices to sign names.
#
# palm_detection_full.tflite:
#   Android hand/palm detector.
#
# hand_landmark_full.tflite:
#   Android 21-point hand landmark model.
#
# They are explicitly copied to the same relative paths expected by
# app_config.py:
#
#   <app>/models/...
#
android.add_assets = \
    models/sign_model.tflite:models/sign_model.tflite, \
    models/class_map.json:models/class_map.json, \
    models/palm_detection_full.tflite:models/palm_detection_full.tflite, \
    models/hand_landmark_full.tflite:models/hand_landmark_full.tflite

# -----------------------------------------------------------------------------
# Android permissions
# -----------------------------------------------------------------------------

# The application accesses the device camera.
android.permissions = android.permission.CAMERA

# -----------------------------------------------------------------------------
# Android API levels
# -----------------------------------------------------------------------------
#
# API 35 is used as the target for this build.
#
# minapi and ndk_api are both 24 because the current NumPy recipe requires
# NDK API 24+ for Android builds.
#
# Do not confuse:
#
#   android.api
#       target/compile API
#
#   android.minapi
#       minimum Android version supported by the APK
#
#   android.ndk_api
#       minimum API used by native NDK compilation
#
android.api = 35
android.minapi = 24
android.ndk_api = 24

# -----------------------------------------------------------------------------
# Architecture
# -----------------------------------------------------------------------------
#
# The application is intentionally arm64-only.
#
# tflite-runtime does not support x86_64 in the referenced p4a recipe, and
# arm64-v8a is the appropriate native architecture for physical Android
# phones targeted by this project.
#
android.archs = arm64-v8a

# -----------------------------------------------------------------------------
# Orientation / display
# -----------------------------------------------------------------------------

fullscreen = 0
orientation = portrait

# -----------------------------------------------------------------------------
# Android launch screen
# -----------------------------------------------------------------------------

android.presplash_color = #090C13

# Uncomment once you actually add an icon/presplash image:
#
# presplash.filename = %(source.dir)s/assets/presplash.png
# icon.filename = %(source.dir)s/assets/icon.png

# -----------------------------------------------------------------------------
# Android application data / backup
# -----------------------------------------------------------------------------

android.allow_backup = True

# -----------------------------------------------------------------------------
# Android SDK license handling
# -----------------------------------------------------------------------------

# Required for unattended GitHub Actions builds.
android.accept_sdk_license = True

# -----------------------------------------------------------------------------
# python-for-android
# -----------------------------------------------------------------------------
#
# Use the latest STABLE p4a release that is compatible with Python 3.12.
#
# v2026.05.09 is the current stable release identified by the p4a project.
# Pinning the exact commit makes the CI build reproducible instead of silently
# changing when p4a/master moves.
#
# Release commit:
#   58d21141...
#
p4a.fork = kivy
p4a.branch = master
p4a.commit = 58d21141

# -----------------------------------------------------------------------------
# Buildozer
# -----------------------------------------------------------------------------

[buildozer]

# Maximum useful logging for CI diagnosis.
log_level = 2

# GitHub Actions does not normally run Buildozer as root, so keep the warning.
warn_on_root = 1
