[app]

# (str) Title of your application
title = PSL AI

# (str) Package name
package.name = pslai

# (str) Package domain (unique identifier)
package.domain = org.pslai

# (str) Source code directory
source.dir = psl_app

# (str) Application version
version = 1.0.0

# (str) Supported source file extensions
source.include_exts = py,json,tflite,png,jpg,jpeg,kv

# (str) Android requirements.
# ARM64 only is configured below.
requirements = python3,kivy,numpy,opencv,mediapipe,tflite-runtime

# (str) Presplash
presplash.filename =

# (str) Icon
icon.filename =

# (str) Supported orientation
orientation = portrait

# (str) Fullscreen
fullscreen = 1

# (str) Android permissions
android.permissions = CAMERA

# (str) Android architecture
android.arch = arm64-v8a

# (str) Android API / build targets
android.api = 35
android.minapi = 24
android.ndk = 27c
android.accept_sdk_license = True

# Keep the APK focused on the target architecture.
android.add_src =

# (str) Python entry point
entrypoint = main.py

# (str) Logcat verbosity
log_level = 2

# (str) Build mode defaults
[buildozer]

# (str) Warn about old buildozer versions
warn_on_root = 1
