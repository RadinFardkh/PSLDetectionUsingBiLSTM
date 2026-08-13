"""Central configuration for the Persian Sign Language Android app."""

import os
from kivy.metrics import dp

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
PROCESSED_DIR = os.path.join(BASE_DIR, "models")
TFLITE_MODEL_NAME = "sign_model.tflite"
CLASS_MAP_NAME = "class_map.json"

FEATURE_TYPE = "both"
SEQUENCE_LENGTH = 60
STABLE_REQUIRED_TIME = 2.0
INFER_EVERY_N_FRAMES = 3
PREDICTION_WINDOW = 15
VOTE_RATIO_THRESHOLD = 0.75

# Android-first design tokens. Centralized to keep screens visually coherent.
COLOR_BACKGROUND = (0.035, 0.047, 0.075, 1)
COLOR_SURFACE = (0.075, 0.090, 0.130, 0.96)
COLOR_SURFACE_ALT = (0.105, 0.122, 0.168, 0.96)
COLOR_PRIMARY = (0.24, 0.55, 0.98, 1)
COLOR_PRIMARY_PRESSED = (0.18, 0.45, 0.86, 1)
COLOR_ON_PRIMARY = (1, 1, 1, 1)
COLOR_TEXT_PRIMARY = (0.96, 0.97, 1, 1)
COLOR_TEXT_SECONDARY = (0.68, 0.72, 0.82, 1)
COLOR_TEXT_DARK = (0.07, 0.09, 0.13, 1)
COLOR_BORDER = (0.20, 0.23, 0.31, 0.70)
COLOR_GLASS = (0.055, 0.070, 0.105, 0.78)
COLOR_GLASS_PRESSED = (0.12, 0.15, 0.22, 0.95)
COLOR_SUCCESS = (0.25, 0.86, 0.57, 1)
COLOR_WARNING = (1.0, 0.73, 0.27, 1)
COLOR_ERROR = (1.0, 0.35, 0.38, 1)
COLOR_HEADER_TOP = (0.08, 0.14, 0.28, 1)
COLOR_HEADER_BOTTOM = (0.20, 0.38, 0.67, 1)
COLOR_BG = COLOR_BACKGROUND
COLOR_BUTTON = COLOR_PRIMARY
COLOR_BUTTON_TEXT = COLOR_ON_PRIMARY

SPACING_XS = dp(6)
SPACING_SM = dp(10)
SPACING_MD = dp(16)
SPACING_LG = dp(24)
SPACING_XL = dp(32)
MIN_TOUCH = dp(48)


def model_path():
    return os.path.join(MODEL_DIR, TFLITE_MODEL_NAME)


def class_map_path():
    return os.path.join(PROCESSED_DIR, CLASS_MAP_NAME)
