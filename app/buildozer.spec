[app]

# App name
title = PSL AI

# Package name
package.name = pslai

# Package domain
package.domain = org.pslai

# Source directory
source.dir = .

# Python files and other files to include
source.include_exts = py,png,jpg,jpeg,json,tflite,txt

# Application version
version = 1.0

# Python requirements
requirements = python3,kivy==2.3.0,numpy,opencv,tflite-runtime

# Orientation
orientation = portrait

# Fullscreen
fullscreen = 0

# Android permissions
android.permissions = CAMERA

# Android API
android.api = 35

# Minimum Android API
android.minapi = 24

# Android architecture
android.arch = arm64-v8a

# Don't compile unnecessary files
android.add_src =

# Don't add extra Java code
android.add_jars =

# Build settings
android.accept_sdk_license = True


[buildozer]

# Log level
log_level = 2

# Warning timeout
warn_on_root = 1