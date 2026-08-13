# =============================================================================
# buildozer.spec — PSL AI (Persian Sign Language Recognizer)
#
# Purpose: Instructs Buildozer (https://github.com/kivy/buildozer) how to
# cross-compile this Kivy/Python app into an Android APK using the
# python-for-android (p4a) toolchain.
#
# CRITICAL SIZE NOTE:
#   The Python `mediapipe` package has NO arm64/aarch64 wheel on PyPI.
#   Listing it in `requirements` would either fail the build or silently
#   embed an x86_64 .so that crashes at runtime with:
#       ImportError: ... is for EM_X86_64 (62) instead of EM_AARCH64 (183)
#   (Source: github.com/google/mediapipe/issues/3852)
#
#   Instead, engine.py already handles landmark detection via the raw
#   .tflite hand-landmark model files (Option C in our architecture notes).
#   This keeps the APK well under 50 MB when built arm64-only.
#
# SIZE BUDGET (approximate, arm64-v8a debug build):
#   Python runtime + Kivy SDL2 bootstrap  ~18 MB
#   numpy                                  ~4 MB
#   opencv (arm64 p4a recipe)              ~8 MB
#   tflite-runtime                         ~3 MB
#   sign_model.tflite (your model)         ~1–4 MB
#   App Python source                      <1 MB
#   Total (compressed APK)               ~30–38 MB  ← safely under 50 MB
#
# References:
#   Buildozer docs:        https://buildozer.readthedocs.io
#   p4a recipe list:       https://github.com/kivy/python-for-android/tree/develop/pythonforandroid/recipes
#   c4k_tflite_example:   https://github.com/Android-for-Python/c4k_tflite_example
#   tflite-runtime p4a:   https://github.com/Android-for-Python/c4k_tflite_example (recipe included)
# =============================================================================

[app]

# ── Identity ──────────────────────────────────────────────────────────────────

# (str) Title shown on the Android launcher icon and in app-info.
title = PSL AI

# (str) Internal package name. Must be a valid Java identifier (no spaces,
# no hyphens). Used as the folder name on the device (/data/data/<domain>.<name>).
package.name = pslai

# (str) Reverse-domain package identifier required by Android.
# Change to your own domain before publishing to Google Play.
package.domain = org.example

# ── Source layout ─────────────────────────────────────────────────────────────

# (str) Root directory that contains main.py.
# '.' means the repo root — adjust if main.py is inside a sub-folder.
source.dir = .

# (list) Extensions Buildozer will bundle from source.dir.
# 'tflite' must be listed explicitly so the model file is included.
# 'json' covers class_map.json.  'kv' is for any Kivy language files.
source.include_exts = py,png,jpg,jpeg,kv,atlas,tflite,json

# (list) Glob patterns to EXCLUDE from the bundle.
# Exclude the models/ directory from source scanning — we add the exact
# files we need via android.add_assets below, which avoids double-packing.
# Also exclude desktop-only extras that bloat the APK.
source.exclude_patterns = tests/*,*.spec,__pycache__/*,*.pyc,*.pyo,.git/*,.github/*

# ── Versioning ────────────────────────────────────────────────────────────────

version = 1.0.0

# ── Requirements (the most critical section for size) ─────────────────────────
#
# Each token here triggers a p4a recipe build, which cross-compiles the
# library from source for arm64-v8a. Only list what the app actually imports.
#
# DO NOT add:
#   mediapipe   — no arm64 wheel; will fail or crash (see header note above)
#   tensorflow  — 200 MB+; tflite-runtime is the correct lightweight choice
#   matplotlib  — not needed at runtime
#
# python3        : CPython runtime (p4a manages the exact version)
# kivy           : UI framework + SDL2 bootstrap
# numpy          : used by engine.py for feature vectors and array math
# opencv         : used by camera_screen.py (cv2.flip, cv2.cvtColor, etc.)
#                  The p4a recipe builds only the core modules (~8 MB compressed).
#                  Do NOT use 'opencv-python' — that is the desktop wheel name.
# tflite-runtime : lightweight TFLite interpreter; replaces full TensorFlow.
#                  The p4a recipe for this is provided by the c4k_tflite_example
#                  reference repo. The engine.py already falls back to it
#                  when `import tensorflow` fails.
requirements = python3,kivy==2.3.0,numpy,opencv,tflite-runtime

# ── Assets (model files) ──────────────────────────────────────────────────────
#
# Pack the .tflite model and class map into the APK assets/ directory.
# At runtime, app_config.py resolves these paths via BASE_DIR.
# If your models/ folder is in the repo root alongside main.py, this works
# out of the box. If not, adjust the path to match your project layout.
#
# Format: source_path:destination_relative_path
# (leave destination empty to keep the same relative path)
android.add_assets = models/sign_model.tflite:models/sign_model.tflite,models/class_map.json:models/class_map.json

# ── Permissions ───────────────────────────────────────────────────────────────
#
# CAMERA is required for cv2.VideoCapture to open the device camera.
# Without this, capture.isOpened() returns False on all modern Android versions.
# The app already requests this permission at runtime in main.py via
# android.permissions.request_permissions([Permission.CAMERA]).
android.permissions = android.permission.CAMERA

# ── Android API levels ────────────────────────────────────────────────────────
#
# api = the targetSdkVersion Gradle uses when compiling the APK.
# Should be kept current (33 = Android 13). Google Play requires >= 33 since 2024.
#
# minapi = the minimum Android version the APK will install on.
# 21 = Android 5.0 (Lollipop) — covers virtually every device in use.
# If you want to exclude very old devices that may lack NEON vector instructions
# (which numpy and opencv require), raise this to 23 (Android 6.0).
android.api = 33
android.minapi = 24

# ── Architecture ──────────────────────────────────────────────────────────────
#
# arm64-v8a = 64-bit ARM. All Android phones since ~2017 support it.
# Building for ONE architecture only (instead of the default fat binary
# that includes armeabi-v7a + arm64-v8a) is the single biggest APK size
# reduction available: it roughly halves the native library payload.
#
# The tflite-runtime p4a recipe does NOT currently build for x86_64,
# so do not add x86_64 here unless you provide your own recipe.
# (Source: c4k_tflite_example README)
android.archs = arm64-v8a

# ── Display & orientation ─────────────────────────────────────────────────────

# Kivy apps run full-screen by default on Android.
fullscreen = 0

# Support portrait only. The camera screen layout is designed for portrait.
# Change to 'all' if you add landscape support later.
orientation = portrait

# ── p4a (python-for-android) hooks ────────────────────────────────────────────
#
# If you adopt Camera4Kivy later (to replace the bare cv2.VideoCapture),
# uncomment the line below and copy the camerax_provider/ folder from:
#   https://github.com/Android-for-Python/c4k_tflite_example
# into your project root. It configures the Gradle CameraX dependency.
#
# p4a.hook = camerax_provider/gradle_options.py

# ── Icon & splash (optional but recommended for a polished APK) ───────────────
#
# Uncomment and add your own image files when ready.
# presplash.filename = %(source.dir)s/assets/presplash.png
# icon.filename      = %(source.dir)s/assets/icon.png

# Presplash background color while the Python runtime loads.
# Matches COLOR_BACKGROUND from app_config.py (dark navy).
android.presplash_color = #090C13

# ── Backup (Android 6+ auto-backup) ───────────────────────────────────────────
#
# True = Android backs up app data to the user's Google account.
# This app stores no sensitive user data, so auto-backup is harmless.
android.allow_backup = True

# ── NDK & SDK (let Buildozer download automatically) ──────────────────────────
#
# Leave these commented out. Buildozer downloads the correct versions
# automatically on first build. Pinning specific versions here can cause
# incompatibilities with newer p4a recipes.
#
# android.ndk = 25b
# android.sdk = 33

# ── Accept SDK licenses automatically (needed for CI/GitHub Actions) ──────────
#
# In a headless CI environment there is no interactive terminal, so Buildozer
# must accept the Android SDK license agreement without prompting.
# This flag is equivalent to running `yes | sdkmanager --licenses`.
android.accept_sdk_license = True

# =============================================================================
# [buildozer] section — controls Buildozer's own behavior, not the APK
# =============================================================================

[buildozer]

# (int) Verbosity: 0 = errors only, 1 = info, 2 = full debug output.
# Use 2 during development; drop to 1 once the build is stable to reduce CI log noise.
log_level = 2

# (int) Warn if buildozer is run as root. Set to 0 only inside a Docker container
# (e.g., the official kivy/buildozer image used by GitHub Actions) where root is expected.
warn_on_root = 1
