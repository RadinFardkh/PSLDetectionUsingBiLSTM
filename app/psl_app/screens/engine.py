"""
Recognition engine — adapted from the original mrealtime.py script.

Same math (One-Euro filtering, joint-angle + normalized-coord features,
majority voting, stability confirmation) but restructured so a Kivy
Clock-driven loop can push one frame at a time and read back state,
instead of owning its own `while True` / `cv2.imshow` loop.

Call `reload()` at any point to hot-swap the .tflite model and/or
class_map.json referenced in app_config.py — e.g. after dropping in a
newer version.
"""

import json
import os
import time
from collections import deque

import numpy as np
import cv2
import mediapipe as mp

try:
    import tensorflow as tf
    _TFLITE_BACKEND = "tensorflow"
except ImportError:
    # Full TF is heavy; tflite-runtime is the lightweight alternative
    # and is a drop-in for Interpreter. Falls back automatically.
    import tflite_runtime.interpreter as tf_lite_runtime
    _TFLITE_BACKEND = "tflite_runtime"

import app_config as config


# ----------------------------- Filters -------------------------------------

class LowPassFilter:
    def __init__(self, alpha):
        self.alpha = alpha
        self.y_prev = None

    def filter(self, x):
        if self.y_prev is None:
            self.y_prev = x
            return x
        y = self.alpha * x + (1 - self.alpha) * self.y_prev
        self.y_prev = y
        return y


class CausalOneEuroFilter:
    def __init__(self, min_cutoff=1.0, beta=0.0, d_cutoff=1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_prev = None
        self.dx_prev = None
        self.lp_x = LowPassFilter(self._alpha(min_cutoff))
        self.lp_dx = LowPassFilter(self._alpha(d_cutoff))
        self.first_time = True

    def _alpha(self, cutoff, dt=1.0):
        tau = 1.0 / (2 * np.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def filter(self, x, dt=1.0):
        if self.first_time:
            self.x_prev = x
            self.dx_prev = 0.0
            self.first_time = False
            return x
        dx = (x - self.x_prev) / dt
        dx_hat = self.lp_dx.filter(dx)
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        self.lp_x = LowPassFilter(self._alpha(cutoff, dt))
        x_hat = self.lp_x.filter(x)
        self.x_prev = x_hat
        return x_hat


def create_filter_bank():
    bank = []
    for _hand in range(2):
        hand_bank = []
        for _lm in range(21):
            lm_bank = [CausalOneEuroFilter(min_cutoff=1.0, beta=0.05) for _ in range(3)]
            hand_bank.append(lm_bank)
        bank.append(hand_bank)
    return bank


# ------------------------- Feature extraction -------------------------------

HAND_TRIPLETS = [
    (0, 1, 2), (1, 2, 3), (2, 3, 4),
    (0, 5, 6), (5, 6, 7), (6, 7, 8),
    (0, 9, 10), (9, 10, 11), (10, 11, 12),
    (0, 13, 14), (13, 14, 15), (14, 15, 16),
    (0, 17, 18), (17, 18, 19), (18, 19, 20),
    (5, 0, 9), (9, 0, 13), (13, 0, 17), (1, 0, 5), (17, 0, 5)
]


def compute_joint_angles(hand_coords):
    angles = np.zeros(20, dtype=np.float32)
    for i, (a, b, c) in enumerate(HAND_TRIPLETS):
        v1 = hand_coords[a] - hand_coords[b]
        v2 = hand_coords[c] - hand_coords[b]
        dot = np.dot(v1, v2)
        norm = np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8
        angles[i] = np.arccos(np.clip(dot / norm, -1.0, 1.0))
    return angles


def normalise_hand_coords(hand_coords):
    wrist = hand_coords[0]
    centered = hand_coords - wrist
    dists = np.linalg.norm(centered, axis=1)
    scale = np.max(dists) + 1e-8
    return (centered / scale).flatten()


def extract_features_from_landmarks(left_lm, right_lm, feature_type):
    left_angles = np.zeros(20, dtype=np.float32)
    right_angles = np.zeros(20, dtype=np.float32)
    left_coords = np.zeros(63, dtype=np.float32)
    right_coords = np.zeros(63, dtype=np.float32)

    if np.any(left_lm):
        left_angles = compute_joint_angles(left_lm)
        left_coords = normalise_hand_coords(left_lm)
    if np.any(right_lm):
        right_angles = compute_joint_angles(right_lm)
        right_coords = normalise_hand_coords(right_lm)

    if feature_type == "angles":
        return np.concatenate([left_angles, right_angles])
    elif feature_type == "coords":
        return np.concatenate([left_coords, right_coords])
    else:
        return np.concatenate([left_angles, right_angles, left_coords, right_coords])


# ------------------------------ Engine --------------------------------------

class SignRecognitionEngine:
    """
    Headless per-frame recognition engine. Feed it BGR frames via
    `process_frame`; it mutates and returns a small state dict with
    everything the UI needs to render (skeleton landmarks, current
    prediction, confidence, stability, confirmed history).

    Designed for low overhead: MediaPipe + inference only run at the
    frame rate you call `process_frame`, and TFLite inference itself
    is throttled to every Nth frame (config.INFER_EVERY_N_FRAMES),
    matching the original script's "save compute" strategy.
    """

    def __init__(self):
        self.interpreter = None
        self.input_details = None
        self.output_details = None
        self.class_map = {}
        self.idx_to_class = {}

        self._mp_holistic = mp.solutions.holistic
        self.detector = self._mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.55,
            min_tracking_confidence=0.55,
        )

        self.sequence_buffer = deque(maxlen=config.SEQUENCE_LENGTH)
        self.prediction_window = deque(maxlen=config.PREDICTION_WINDOW)
        self.confirmed_history = deque(maxlen=10)

        self.filter_bank = create_filter_bank()

        self.current_prediction = None
        self.current_confidence = 0.0
        self.stable_prediction = None
        self.stable_start_time = None
        self.stable_duration = 0.0

        self._frame_counter = 0

        self.reload()

    # -- Hot-swappable model loading -----------------------------------
    def reload(self):
        """(Re)load the .tflite model and class_map.json from
        app_config.py's current paths. Safe to call at runtime after
        swapping in a newer model version."""
        model_file = config.model_path()
        class_map_file = config.class_map_path()

        if not os.path.exists(model_file):
            raise FileNotFoundError(f"Model not found: {model_file}")
        if not os.path.exists(class_map_file):
            raise FileNotFoundError(f"Class map not found: {class_map_file}")

        if _TFLITE_BACKEND == "tensorflow":
            self.interpreter = tf.lite.Interpreter(model_path=model_file)
        else:
            self.interpreter = tf_lite_runtime.Interpreter(model_path=model_file)

        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        with open(class_map_file) as f:
            self.class_map = json.load(f)
        self.idx_to_class = {v: k for k, v in self.class_map.items()}

        # Reset temporal state so a stale buffer doesn't mix with a new model
        self.sequence_buffer.clear()
        self.prediction_window.clear()
        self.current_prediction = None
        self.current_confidence = 0.0
        self.stable_prediction = None
        self.stable_start_time = None

    def reset_session(self):
        """Clear history/state, e.g. when the user presses Start again."""
        self.sequence_buffer.clear()
        self.prediction_window.clear()
        self.confirmed_history.clear()
        self.current_prediction = None
        self.current_confidence = 0.0
        self.stable_prediction = None
        self.stable_start_time = None
        self.filter_bank = create_filter_bank()

    # -- Per-frame processing --------------------------------------------
    def process_frame(self, bgr_frame, run_inference=True):
        """
        bgr_frame: OpenCV BGR ndarray from the camera.
        run_inference: if False, still runs MediaPipe (for skeleton
            drawing) but skips the TFLite step entirely — useful to
            save compute when the "Start" flow hasn't started yet.

        Returns a dict:
            {
              'left_landmarks': list[(x,y)] or None,   # normalized 0..1
              'right_landmarks': list[(x,y)] or None,
              'left_connections' / 'right_connections': HAND_CONNECTIONS
              'prediction': str or None,
              'confidence': float,
              'stable_duration': float,
              'confirmed_history': list[dict],
            }
        """
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self.detector.process(rgb)

        left_lm_out = None
        right_lm_out = None
        if results.left_hand_landmarks:
            left_lm_out = [(lm.x, lm.y) for lm in results.left_hand_landmarks.landmark]
        if results.right_hand_landmarks:
            right_lm_out = [(lm.x, lm.y) for lm in results.right_hand_landmarks.landmark]

        if not run_inference:
            return {
                "left_landmarks": left_lm_out,
                "right_landmarks": right_lm_out,
                "connections": list(self._mp_holistic.HAND_CONNECTIONS),
                "prediction": self.idx_to_class.get(self.current_prediction),
                "confidence": self.current_confidence,
                "stable_duration": self.stable_duration,
                "confirmed_history": list(self.confirmed_history),
            }

        left_raw = np.zeros((21, 3), dtype=np.float32)
        right_raw = np.zeros((21, 3), dtype=np.float32)
        if results.left_hand_landmarks:
            left_raw = np.array([[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark])
        if results.right_hand_landmarks:
            right_raw = np.array([[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark])

        left_filt, right_filt = self._filter_landmarks(left_raw, right_raw)
        feat = extract_features_from_landmarks(left_filt, right_filt, config.FEATURE_TYPE)
        self.sequence_buffer.append(feat)

        self._frame_counter += 1

        if (len(self.sequence_buffer) == config.SEQUENCE_LENGTH
                and self._frame_counter % config.INFER_EVERY_N_FRAMES == 0):
            self._run_inference()

        return {
            "left_landmarks": left_lm_out,
            "right_landmarks": right_lm_out,
            "connections": list(self._mp_holistic.HAND_CONNECTIONS),
            "prediction": self.idx_to_class.get(self.current_prediction),
            "confidence": self.current_confidence,
            "stable_duration": self.stable_duration,
            "confirmed_history": list(self.confirmed_history),
        }

    def _filter_landmarks(self, left_raw, right_raw):
        left_out = np.zeros_like(left_raw)
        right_out = np.zeros_like(right_raw)
        for lm in range(21):
            for c in range(3):
                left_out[lm, c] = self.filter_bank[0][lm][c].filter(left_raw[lm, c])
                right_out[lm, c] = self.filter_bank[1][lm][c].filter(right_raw[lm, c])
        return left_out, right_out

    def _run_inference(self):
        seq = np.array(self.sequence_buffer, dtype=np.float32)
        seq = np.expand_dims(seq, axis=0)
        self.interpreter.set_tensor(self.input_details[0]["index"], seq)
        self.interpreter.invoke()
        output = self.interpreter.get_tensor(self.output_details[0]["index"])[0]

        pred_class = int(np.argmax(output))
        confidence = float(output[pred_class])
        self.prediction_window.append(pred_class)

        if len(self.prediction_window) < self.prediction_window.maxlen:
            return

        counts = np.bincount(self.prediction_window)
        dominant = int(np.argmax(counts))
        vote_ratio = counts[dominant] / len(self.prediction_window)

        if vote_ratio < config.VOTE_RATIO_THRESHOLD:
            self.current_prediction = None
            self.current_confidence = 0.0
            self.stable_prediction = None
            self.stable_start_time = None
            self.stable_duration = 0.0
            return

        self.current_prediction = dominant
        self.current_confidence = confidence

        now = time.monotonic()
        if self.stable_prediction != self.current_prediction:
            self.stable_prediction = self.current_prediction
            self.stable_start_time = now

        self.stable_duration = now - self.stable_start_time

        if self.stable_duration >= config.STABLE_REQUIRED_TIME:
            sign_name = self.idx_to_class[self.current_prediction]
            if not self.confirmed_history or self.confirmed_history[-1]["prediction"] != sign_name:
                self.confirmed_history.append({
                    "prediction": sign_name,
                    "confidence": self.current_confidence,
                    "time": now,
                })

    def close(self):
        self.detector.close()
