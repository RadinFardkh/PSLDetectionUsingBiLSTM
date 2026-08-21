[app]

# (str) Title of your application
title = PSL AI

# (str) Package name
package.name = pslai

# (str) Package domain
package.domain = org.pslai

# (str) Source code directory
source.dir = .

# (str) Main Python file
source.main = main.py

# (list) List of source files to include
source.include_exts = py,png,jpg,jpeg,kv,atlas,json,tflite

# (list) List of source directories to exclude
source.exclude_dirs = tests,bin,.git,.github,.buildozer

# (str) Application version
version = 1.0

# (list) Application requirements
requirements = python3==3.11.9,hostpython3==3.11.9,kivy==2.3.1,numpy,opencv,tflite-runtime

# (str) Supported orientation
orientation = portrait

# (bool) Fullscreen mode
fullscreen = 0

# (str) Presplash
presplash.filename = %(source.dir)s/data/presplash.png

# (str) Icon
icon.filename = %(source.dir)s/data/icon.png


[buildozer]

# (int) Log level
log_level = 2

# (str) Build directory
build_dir = ./.buildozer

# (str) Output directory
bin_dir = ./bin


[app:android]

# (list) Android architectures
android.archs = arm64-v8a

# (int) Minimum Android API
android.minapi = 24

# (int) Target Android API
android.api = 35

# (str) Android NDK version
android.ndk = 28c

# (bool) Copy libraries
android.copy_libs = 1

# (bool) Allow backup
android.allow_backup = True

# (str) Android application theme
android.presplash_color = #000000


[python-for-android]

# Use the Buildozer-managed NDK
p4a.ndk_api = 24
