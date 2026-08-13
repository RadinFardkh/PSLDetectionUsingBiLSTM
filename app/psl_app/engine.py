"""
engine.py — Sign Recognition Engine (Android-optimized, MediaPipe-free)

This module is the computational core of the PSL AI app. It receives raw
camera frames from CameraScreen, extracts hand-landmark features, feeds
them through the BiLSTM TFLite model, and returns a small state dict that
the Kivy UI reads to render predictions.

KEY ANDROID CHANGE vs. the desktop version:
    The original code imports `mediapipe as mp` at the top level and uses
    `mp.solutions.holistic.Holistic` for landmark detection.  That works
    fine on a desktop (Linux/macOS/Windows x86_64), but MediaPipe publishes
    NO arm64/aarch64 Python wheel on PyPI.  Running `import mediapipe` inside
    an arm64-v8a APK raises:
        ImportError: ... _framework_bindings.so is for EM_X86_64, not EM_AARCH64
    (See: github.com/google/mediapipe/issues/3852)

    The replacement used here is the raw MediaPipe hand-landmark TFLite model
    loaded directly through tflite-runtime (Option C from our architecture
    discussion).  This is fully offline, needs no Kotlin/JNI bridge, and
    keeps the APK under 50 MB.

    Model files required (place them in models/):
        palm_detection_full.tflite   — detects hand bounding boxes
        hand_landmark_full.tflite    — 21 3-D landmarks from a cropped hand

    Both files can be downloaded from:
        https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
    or individually from the MediaPipe repository:
        mediapipe/modules/hand_landmark/

    For the initial prototype we still provide a DESKTOP FALLBACK that
    imports mediapipe normally when available, so development on a laptop
    continues to work without code changes.  The Android-safe landmark
    extractor is used automatically when mediapipe is not importable.

References:
    TFLite Interpreter API:  https://www.tensorflow.org/lite/api_docs/python/tf/lite/Interpreter
    tflite-runtime:          https://pypi.org/project/tflite-runtime/
    One Euro Filter paper:   https://cristal.univ-lille.fr/~casiez/1euro/
    MediaPipe Hand model:    https://developers.google.com/mediapipe/solutions/vision/hand_landmarker
"""

import json
import os
import time
from collections import deque

import numpy as np
import cv2

import app_config as config


# =============================================================================
# TFLite backend selection
# =============================================================================
"""
We prefer the full `tensorflow` package (which bundles tf.lite.Interpreter)
when running on a developer machine, because it is already installed there.
On Android the full TensorFlow package is ~200 MB and is not practical to
bundle.  `tflite-runtime` provides an identical Interpreter class at a
fraction of the size (~3 MB compressed), so we fall back to it automatically.

This try/except pattern lets one code-base work on both environments without
conditional imports scattered throughout the file.
"""
try:
    import tensorflow as tf
    _TFLITE_BACKEND = "tensorflow"
except ImportError:
    """
    Full TensorFlow is unavailable (expected on Android).
    tflite_runtime.interpreter.Interpreter is a drop-in replacement for
    tf.lite.Interpreter — same constructor signature, same tensor API.
    """
    import tflite_runtime.interpreter as _tflite_runtime_module
    _TFLITE_BACKEND = "tflite_runtime"


# =============================================================================
# MediaPipe availability check
# =============================================================================
"""
On a developer machine, `mediapipe` is available and we can use the high-level
Holistic solution which handles detection + landmark tracking internally.

On Android (arm64) there is no compatible mediapipe wheel, so we fall back to
our own lightweight landmark extractor that calls the raw .tflite model files
directly. _MEDIAPIPE_AVAILABLE controls which path is taken inside
SignRecognitionEngine.__init__().
"""
try:
    import mediapipe as mp
    _MEDIAPIPE_AVAILABLE = True
except ImportError:
    _MEDIAPIPE_AVAILABLE = False


# =============================================================================
# Section 1: Noise filters
# =============================================================================

class LowPassFilter:
    """
    Simple exponential moving-average (EMA) low-pass filter.

    The formula   y[t] = alpha * x[t] + (1 - alpha) * y[t-1]   smooths the
    signal by blending the new sample with the previous output.  A high alpha
    (close to 1) makes the filter very responsive (but noisy); a low alpha
    (close to 0) makes it very smooth (but laggy).

    This class is used internally by CausalOneEuroFilter as its position and
    derivative sub-filters.

    Reference: standard EMA formula, widely documented in signal processing
    literature (e.g. scipy.signal.lfilter with b=[alpha], a=[1, alpha-1]).
    """

    def __init__(self, alpha):
        """
        alpha : float in (0, 1]
            Smoothing coefficient. Derived from the cutoff frequency in
            CausalOneEuroFilter._alpha().
        """
        self.alpha = alpha
        self.y_prev = None  # Will be initialized with the first sample.

    def filter(self, x):
        """
        Apply the filter to a single scalar sample x.
        Returns the smoothed output y.
        On the very first call, the output equals the input (no history yet).
        """
        if self.y_prev is None:
            """
            Bootstrap: set the internal state to the first observed value.
            Without this, the filter would start from 0.0 and take many
            frames to 'settle', causing a large transient at the start.
            """
            self.y_prev = x
            return x
        y = self.alpha * x + (1 - self.alpha) * self.y_prev
        self.y_prev = y
        return y


class CausalOneEuroFilter:
    """
    The One Euro Filter — a low-latency, adaptive filter for noisy 1-D signals.

    Unlike a fixed-cutoff low-pass filter, the One Euro Filter raises its
    cutoff frequency when the signal is moving fast (to stay responsive) and
    lowers it when the signal is nearly static (to suppress jitter).  This is
    ideal for real-time landmark tracking: a hand at rest should be rock-steady
    on screen, while a fast-moving hand should be tracked without lag.

    Parameters:
        min_cutoff : float
            Minimum cutoff frequency in Hz.  Lower values give more smoothing
            at rest.  Default 1.0 Hz is suitable for hand landmarks.
        beta : float
            Speed coefficient.  Higher values make the filter react faster to
            quick movements.  0.05 is a reasonable default for hand tracking.
        d_cutoff : float
            Cutoff frequency for the derivative sub-filter. Usually left at 1.0.

    Reference:
        Casiez, G., Roussel, N. and Vogel, D. (2012) '1€ Filter: A Simple
        Speed-Based Low-Pass Filter for Noisy Input in Interactive Systems',
        CHI 2012. https://cristal.univ-lille.fr/~casiez/1euro/
    """

    def __init__(self, min_cutoff=1.0, beta=0.0, d_cutoff=1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_prev = None
        self.dx_prev = None
        """
        Two LowPassFilter instances:
            lp_x  : filters the position (the landmark coordinate itself)
            lp_dx : filters the derivative (the speed of the landmark)
        The derivative filter uses a fixed cutoff (d_cutoff) because we want
        a stable speed estimate.
        """
        self.lp_x = LowPassFilter(self._alpha(min_cutoff))
        self.lp_dx = LowPassFilter(self._alpha(d_cutoff))
        self.first_time = True

    def _alpha(self, cutoff, dt=1.0):
        """
        Convert a cutoff frequency (Hz) into the EMA coefficient alpha.

        The formula comes from the bilinear transform of a first-order RC
        low-pass filter:  alpha = 1 / (1 + tau/dt)  where tau = 1/(2*pi*fc).

        dt defaults to 1.0, which assumes a fixed per-frame time step.
        For a more accurate filter at varying frame rates, pass the real dt
        (seconds since last frame) to filter().
        """
        tau = 1.0 / (2 * np.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def filter(self, x, dt=1.0):
        """
        Filter a single scalar sample x.  dt is the time step in seconds.

        Algorithm:
            1. Estimate the speed (derivative) of x.
            2. Smooth the speed with lp_dx.
            3. Compute an adaptive cutoff:  fc = min_cutoff + beta * |speed|
            4. Smooth x with a low-pass filter at the adaptive cutoff.
        """
        if self.first_time:
            """
            Bootstrap on first call — same rationale as LowPassFilter.
            Derivative is initialized to 0.0 (assume signal starts at rest).
            """
            self.x_prev = x
            self.dx_prev = 0.0
            self.first_time = False
            return x

        dx = (x - self.x_prev) / dt          # Raw derivative (speed).
        dx_hat = self.lp_dx.filter(dx)       # Smoothed speed.
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)  # Adaptive cutoff.
        """
        Re-create lp_x with the new cutoff. This is slightly inefficient
        (one object allocation per call) but acceptable given that we have
        2 hands × 21 landmarks × 3 axes = 126 filter instances total,
        each called once per processed frame.
        """
        self.lp_x = LowPassFilter(self._alpha(cutoff, dt))
        x_hat = self.lp_x.filter(x)         # Smoothed position.
        self.x_prev = x_hat
        return x_hat


def create_filter_bank():
    """
    Allocate a complete set of One Euro Filters for all landmarks and axes.

    Structure: bank[hand_index][landmark_index][axis_index]
        hand_index  : 0 = left hand, 1 = right hand
        landmark_index : 0–20 (MediaPipe hand has 21 landmarks)
        axis_index  : 0 = x, 1 = y, 2 = z

    Returning a fresh bank (rather than resetting in-place) is intentional:
    it guarantees a clean state on session reset, because stale 'y_prev'
    values from the previous session cannot leak into the new one.
    """
    bank = []
    for _hand in range(2):
        hand_bank = []
        for _lm in range(21):
            """
            Each axis gets its own independent filter because x, y, z
            coordinates are statistically independent and have different
            noise profiles (z, the depth estimate, is noisier than x/y).
            """
            lm_bank = [CausalOneEuroFilter(min_cutoff=1.0, beta=0.05)
                       for _ in range(3)]
            hand_bank.append(lm_bank)
        bank.append(hand_bank)
    return bank


# =============================================================================
# Section 2: Feature extraction
# =============================================================================

"""
HAND_TRIPLETS defines the joint angle calculations.

Each tuple (a, b, c) represents a joint where:
    b is the vertex (the knuckle or joint whose angle we measure)
    a and b are the two neighboring landmarks

The 20 triplets cover:
    - 5 finger chains (4 joints each = 20 inter-joint angles for one hand)
    - 5 palm-span angles (spread between finger bases)

MediaPipe hand landmark IDs (0–20) are documented at:
    https://developers.google.com/mediapipe/solutions/vision/hand_landmarker

Notable IDs:
    0  = WRIST
    1–4  = THUMB (CMC → MCP → IP → TIP)
    5–8  = INDEX
    9–12 = MIDDLE
    13–16 = RING
    17–20 = PINKY
"""
HAND_TRIPLETS = [
    (0, 1, 2), (1, 2, 3), (2, 3, 4),          # Thumb chain
    (0, 5, 6), (5, 6, 7), (6, 7, 8),          # Index chain
    (0, 9, 10), (9, 10, 11), (10, 11, 12),    # Middle chain
    (0, 13, 14), (13, 14, 15), (14, 15, 16),  # Ring chain
    (0, 17, 18), (17, 18, 19), (18, 19, 20),  # Pinky chain
    (5, 0, 9), (9, 0, 13), (13, 0, 17),       # Lateral palm spans
    (1, 0, 5), (17, 0, 5),                    # Thumb–index and pinky–index spans
]


def compute_joint_angles(hand_coords):
    """
    Compute 20 joint angles (in radians) from the 21 landmark coordinates
    of a single hand.

    Args:
        hand_coords : np.ndarray shape (21, 3)
            Landmark positions in the MediaPipe normalized coordinate system
            (x, y ∈ [0,1] relative to image width/height; z is depth relative
            to wrist, in the same scale as x).

    Returns:
        angles : np.ndarray shape (20,) dtype float32
            One angle per HAND_TRIPLET, in radians ∈ [0, π].

    Implementation detail:
        The angle at vertex b between rays b→a and b→c is computed using
        the dot-product formula:
            cos(θ) = (v1 · v2) / (|v1| · |v2|)
        The +1e-8 in the denominator prevents division-by-zero when two
        landmarks coincide (degenerate case, e.g. when the model has zero
        confidence and all landmarks collapse to the same point).
    """
    angles = np.zeros(20, dtype=np.float32)
    for i, (a, b, c) in enumerate(HAND_TRIPLETS):
        v1 = hand_coords[a] - hand_coords[b]   # Ray from vertex to landmark a.
        v2 = hand_coords[c] - hand_coords[b]   # Ray from vertex to landmark c.
        dot = np.dot(v1, v2)
        norm = np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8
        """
        np.clip ensures the argument to arccos stays in [-1, 1].
        Floating-point rounding can push dot/norm slightly outside this
        range, which would make arccos return NaN.
        """
        angles[i] = np.arccos(np.clip(dot / norm, -1.0, 1.0))
    return angles


def normalise_hand_coords(hand_coords):
    """
    Normalize a hand's 21 landmark coordinates to be translation- and
    scale-invariant.

    Steps:
        1. Subtract the wrist (landmark 0) to remove absolute hand position.
        2. Divide by the maximum wrist-to-landmark distance to remove hand size.

    This makes the feature robust to:
        - How far the hand is from the camera (scale).
        - Where in the frame the hand is positioned (translation).
        - Different people's hand sizes (scale normalization).

    The output is flattened to a 1-D vector of shape (63,) = 21 × 3.
    The training preprocessing applies the identical normalization, so the
    feature space seen at inference matches what the BiLSTM was trained on.
    """
    wrist = hand_coords[0]
    centered = hand_coords - wrist            # Shift origin to the wrist.
    dists = np.linalg.norm(centered, axis=1) # Distance of each LM from wrist.
    scale = np.max(dists) + 1e-8             # Max distance (avoids /0 when all LMs coincide).
    return (centered / scale).flatten()       # Flatten (21, 3) → (63,).


def extract_features_from_landmarks(left_lm, right_lm, feature_type):
    """
    Assemble the full feature vector for one frame from left and right
    hand landmark arrays.

    Args:
        left_lm  : np.ndarray shape (21, 3) — left hand landmarks.
        right_lm : np.ndarray shape (21, 3) — right hand landmarks.
        feature_type : str — one of 'angles', 'coords', 'both'.

    Returns:
        feature : np.ndarray (float32)
            'angles' → 40 values  (20 left angles + 20 right angles)
            'coords' → 126 values (63 left coords + 63 right coords)
            'both'   → 166 values (angles concatenated with coords)

    IMPORTANT: The feature_type must match what was used during training
    (set in app_config.FEATURE_TYPE).  Changing it without retraining will
    silently produce garbage predictions because the model expects a specific
    input dimension and layout.

    The np.any() checks guard against the all-zeros placeholder that the
    engine inserts when a hand is not detected (see _run_inference).  If we
    computed angles on an all-zeros array we'd get valid-looking angles
    (all π/2 by coincidence of the geometry) that would confuse the model.
    """
    left_angles  = np.zeros(20,  dtype=np.float32)
    right_angles = np.zeros(20,  dtype=np.float32)
    left_coords  = np.zeros(63,  dtype=np.float32)
    right_coords = np.zeros(63,  dtype=np.float32)

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
        """
        'both' is the default (app_config.FEATURE_TYPE = 'both').
        Order matters: the model was trained with angles first, then coords.
        """
        return np.concatenate([left_angles, right_angles,
                                left_coords, right_coords])


# =============================================================================
# Section 3: Android-safe landmark detector
#            (used when mediapipe is not importable)
# =============================================================================

class TFLiteHandDetector:
    """
    Lightweight hand landmark detector that calls MediaPipe's raw .tflite
    models directly via tflite-runtime, with no dependency on the Python
    `mediapipe` package.

    This is Option C from the project architecture discussion:
        - Fully offline (no network calls).
        - No Kotlin/JNI bridge required.
        - Works on arm64-v8a with tflite-runtime.

    Pipeline (simplified):
        1. palm_detection.tflite   → detect hand bounding boxes in the full frame.
        2. hand_landmark.tflite    → predict 21 3-D landmarks in the cropped hand ROI.

    Model files:
        Download from https://storage.googleapis.com/mediapipe-models/ or
        extract from the MediaPipe Python package on a desktop:
            site-packages/mediapipe/modules/hand_landmark/
                palm_detection_full.tflite
                hand_landmark_full.tflite

    NOTE: This is a simplified implementation that detects up to 2 hands using
    independent ROI crops. The full MediaPipe pipeline is more sophisticated
    (it tracks ROIs across frames to avoid re-running detection every frame).
    A future iteration can add that optimization.

    Input resolution for the landmark model: 224×224 pixels (as per Google's spec).
    Reference: https://developers.google.com/mediapipe/solutions/vision/hand_landmarker/index#models
    """

    # MediaPipe hand landmark model expects 224×224 RGB input.
    LM_INPUT_SIZE = 224

    def __init__(self, palm_model_path, landmark_model_path):
        """
        palm_model_path     : str — path to palm_detection_full.tflite
        landmark_model_path : str — path to hand_landmark_full.tflite
        """
        """
        Load the palm detection interpreter.
        Palm detection runs on the full (resized) frame and returns
        bounding box proposals for each hand it finds.
        """
        if _TFLITE_BACKEND == "tensorflow":
            self._palm_interp = tf.lite.Interpreter(model_path=palm_model_path)
            self._lm_interp   = tf.lite.Interpreter(model_path=landmark_model_path)
        else:
            self._palm_interp = _tflite_runtime_module.Interpreter(model_path=palm_model_path)
            self._lm_interp   = _tflite_runtime_module.Interpreter(model_path=landmark_model_path)

        self._palm_interp.allocate_tensors()
        self._lm_interp.allocate_tensors()

        self._palm_input  = self._palm_interp.get_input_details()[0]
        self._palm_output = self._palm_interp.get_output_details()
        self._lm_input    = self._lm_interp.get_input_details()[0]
        self._lm_output   = self._lm_interp.get_output_details()

    def detect(self, rgb_frame):
        """
        Run palm detection + landmark estimation on an RGB frame.

        Args:
            rgb_frame : np.ndarray (H, W, 3) uint8 — RGB image from the camera.

        Returns:
            hands : list of dict, one per detected hand:
                {
                    'landmarks': np.ndarray (21, 3) float32,  # normalized 0..1
                    'handedness': 'Left' or 'Right'           # heuristic
                }
            Returns an empty list if no hands are detected.

        Implementation note:
            The palm detection model uses a 192×192 input for the full-frame
            scan. The landmark model then works on a 224×224 crop aligned to
            the detected palm bounding box. Both input sizes are fixed by
            Google's model architecture and cannot be changed without retraining.
        """
        h, w = rgb_frame.shape[:2]

        """
        === Stage 1: Palm Detection ===
        Resize the frame to 192×192 (the detection model's expected input size),
        normalize pixels to [0, 1] float32, and add the batch dimension.
        """
        palm_size = self._palm_input['shape'][1]  # Usually 192.
        palm_input_img = cv2.resize(rgb_frame, (palm_size, palm_size))
        palm_input_img = (palm_input_img.astype(np.float32) / 255.0)
        palm_input_img = np.expand_dims(palm_input_img, axis=0)

        self._palm_interp.set_tensor(self._palm_input['index'], palm_input_img)
        self._palm_interp.invoke()

        """
        The palm detection model outputs:
            output[0]: bounding box regressions (x_center, y_center, w, h,
                       plus keypoints) — shape (1, N_anchors, ...)
            output[1]: classification scores (palm probability per anchor)
        We use a simple score threshold to filter detections.
        """
        scores = self._palm_interp.get_tensor(self._palm_output[1]['index'])[0]
        boxes  = self._palm_interp.get_tensor(self._palm_output[0]['index'])[0]

        DETECTION_THRESHOLD = 0.5  # Hands with score below this are ignored.
        detected_boxes = []
        for i, score in enumerate(scores):
            if score > DETECTION_THRESHOLD:
                """
                boxes[i] stores [x_center, y_center, width, height] normalized
                to the 192×192 palm-detection input, not the original frame.
                We convert back to original-frame pixel coordinates.
                """
                xc, yc, bw, bh = boxes[i, :4]
                x1 = int((xc - bw / 2) * w)
                y1 = int((yc - bh / 2) * h)
                x2 = int((xc + bw / 2) * w)
                y2 = int((yc + bh / 2) * h)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                if x2 > x1 and y2 > y1:
                    detected_boxes.append((x1, y1, x2, y2, float(score)))

        if not detected_boxes:
            return []

        """
        Limit to 2 hands (one per standard sign language hand use).
        Take the 2 highest-scoring detections.
        """
        detected_boxes.sort(key=lambda b: b[4], reverse=True)
        detected_boxes = detected_boxes[:2]

        """
        === Stage 2: Landmark Estimation ===
        For each detected palm bounding box, crop the region, resize to
        224×224, run the landmark model, and collect the 21 3-D landmarks.
        """
        hands = []
        for (x1, y1, x2, y2, _score) in detected_boxes:
            crop = rgb_frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            lm_input_img = cv2.resize(crop, (self.LM_INPUT_SIZE, self.LM_INPUT_SIZE))
            lm_input_img = (lm_input_img.astype(np.float32) / 255.0)
            lm_input_img = np.expand_dims(lm_input_img, axis=0)

            self._lm_interp.set_tensor(self._lm_input['index'], lm_input_img)
            self._lm_interp.invoke()

            """
            The landmark model output tensor has shape (1, 21*3) = (1, 63):
            a flat array of [x0,y0,z0, x1,y1,z1, ..., x20,y20,z20].
            x, y are normalized to [0,1] within the 224×224 crop.
            z is depth relative to the wrist in the same scale as x.
            We map x and y back to the original full-frame coordinate space
            so they are comparable to what MediaPipe Holistic would return.
            """
            raw_lm = self._lm_interp.get_tensor(self._lm_output[0]['index'])[0]
            landmarks = raw_lm.reshape(21, 3).copy()

            crop_w = x2 - x1
            crop_h = y2 - y1
            landmarks[:, 0] = (landmarks[:, 0] * crop_w + x1) / w  # x → full frame
            landmarks[:, 1] = (landmarks[:, 1] * crop_h + y1) / h  # y → full frame
            # z stays as-is (relative depth, no frame mapping needed)

            """
            Handedness heuristic: whichever detection has a smaller x_center
            (further left in the mirrored selfie view) is classified as the
            RIGHT hand (appears on the left because the frame is mirrored).
            This is a rough approximation — the full MediaPipe pipeline uses
            the model's own handedness classifier output.
            """
            x_center = (x1 + x2) / 2.0
            handedness = "Right" if x_center < w / 2 else "Left"

            hands.append({"landmarks": landmarks, "handedness": handedness})

        return hands

    def close(self):
        """Release interpreter resources (no-op for tflite-runtime, but good practice)."""
        pass


# =============================================================================
# Section 4: Main recognition engine
# =============================================================================

class SignRecognitionEngine:
    """
    Headless per-frame recognition engine. Feed it BGR frames via
    `process_frame`; it mutates and returns a small state dict with
    everything the UI needs to render (skeleton landmarks, current
    prediction, confidence, stability, confirmed history).

    Designed for low overhead:
        - Landmark detection runs every frame (needed for smooth skeleton display).
        - TFLite BiLSTM inference runs every INFER_EVERY_N_FRAMES frames
          (configurable in app_config.py, default=3) to save compute.
        - Majority-voting over the last PREDICTION_WINDOW frames smooths
          transient misclassifications.
        - Stability confirmation (STABLE_REQUIRED_TIME seconds) prevents
          the same sign from being added to the history multiple times.
    """

    def __init__(self):
        """
        Initialize TFLite interpreter, landmark detector, and all state.
        Calls self.reload() at the end to load the model and class map.
        """
        self.interpreter    = None
        self.input_details  = None
        self.output_details = None
        self.class_map      = {}
        self.idx_to_class   = {}

        """
        Choose the landmark detection backend based on availability:
            - If `mediapipe` is importable (desktop), use Holistic for
              best-in-class detection quality and built-in handedness.
            - If not (Android), use TFLiteHandDetector with the raw model files.
        """
        if _MEDIAPIPE_AVAILABLE:
            self._mp_holistic = mp.solutions.holistic
            self.detector = self._mp_holistic.Holistic(
                static_image_mode=False,
                model_complexity=1,
                min_detection_confidence=0.55,
                min_tracking_confidence=0.55,
            )
            self._use_mediapipe = True
        else:
            """
            Android path: load the raw TFLite landmark models.
            The model files must be present in the models/ directory.
            Their paths are resolved relative to app_config.MODEL_DIR.
            """
            palm_model = os.path.join(
                config.MODEL_DIR, "palm_detection_full.tflite"
            )
            lm_model = os.path.join(
                config.MODEL_DIR, "hand_landmark_full.tflite"
            )
            self.detector = TFLiteHandDetector(
                palm_model_path=palm_model,
                landmark_model_path=lm_model,
            )
            self._use_mediapipe = False

        """
        sequence_buffer: rolling window of the last SEQUENCE_LENGTH feature
        vectors (one per processed frame).  When full, it is fed to the BiLSTM.
        Using a deque with maxlen avoids manual index wrapping.
        """
        self.sequence_buffer = deque(maxlen=config.SEQUENCE_LENGTH)

        """
        prediction_window: the last PREDICTION_WINDOW class predictions from
        the BiLSTM.  Majority voting over this window provides temporal
        smoothing — a single noisy inference does not flip the displayed word.
        """
        self.prediction_window  = deque(maxlen=config.PREDICTION_WINDOW)
        self.confirmed_history  = deque(maxlen=10)

        self.filter_bank = create_filter_bank()

        self.current_prediction = None
        self.current_confidence = 0.0
        self.stable_prediction  = None
        self.stable_start_time  = None
        self.stable_duration    = 0.0

        self._frame_counter = 0

        self.reload()

    # ── Hot-swappable model loading ────────────────────────────────────────────

    def reload(self):
        """
        (Re)load the BiLSTM .tflite model and class_map.json from the paths
        defined in app_config.py.

        This method is safe to call at runtime, e.g. to hot-swap a new model
        version without restarting the app.  It clears all temporal state
        (sequence buffer, prediction window) so that stale features from the
        previous model do not contaminate the new one's first inference.

        Raises FileNotFoundError if either file is missing.
        """
        model_file     = config.model_path()
        class_map_file = config.class_map_path()

        if not os.path.exists(model_file):
            raise FileNotFoundError(f"Model not found: {model_file}")
        if not os.path.exists(class_map_file):
            raise FileNotFoundError(f"Class map not found: {class_map_file}")

        """
        Load the TFLite interpreter for the BiLSTM sign-recognition model.
        allocate_tensors() pre-allocates the input/output tensor buffers.
        This must be called before set_tensor/invoke.
        Reference: https://www.tensorflow.org/lite/api_docs/python/tf/lite/Interpreter
        """
        if _TFLITE_BACKEND == "tensorflow":
            self.interpreter = tf.lite.Interpreter(model_path=model_file)
        else:
            self.interpreter = _tflite_runtime_module.Interpreter(model_path=model_file)

        self.interpreter.allocate_tensors()
        self.input_details  = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        """
        class_map.json maps sign name strings to integer class indices,
        e.g. {"سلام": 0, "ممنون": 1, ...}.
        We invert it (idx_to_class) so we can look up the name by argmax index.
        """
        with open(class_map_file, encoding="utf-8") as f:
            self.class_map = json.load(f)
        self.idx_to_class = {v: k for k, v in self.class_map.items()}

        self.sequence_buffer.clear()
        self.prediction_window.clear()
        self.current_prediction = None
        self.current_confidence = 0.0
        self.stable_prediction  = None
        self.stable_start_time  = None

    def reset_session(self):
        """
        Clear all temporal state.  Called when the user presses Start again
        to begin a fresh recognition session (empty sentence history).
        A new filter_bank is created to avoid stale smoothing state from
        the previous session bleeding into the first frames of the new one.
        """
        self.sequence_buffer.clear()
        self.prediction_window.clear()
        self.confirmed_history.clear()
        self.current_prediction = None
        self.current_confidence = 0.0
        self.stable_prediction  = None
        self.stable_start_time  = None
        self.filter_bank = create_filter_bank()

    # ── Per-frame processing ───────────────────────────────────────────────────

    def process_frame(self, bgr_frame, run_inference=True):
        """
        Process one camera frame.

        Args:
            bgr_frame    : np.ndarray (H, W, 3) uint8 — OpenCV BGR frame.
            run_inference: bool — if False, runs landmark detection only
                (for skeleton display) but skips feature extraction and
                the TFLite BiLSTM call. Use this when the user hasn't
                pressed Start yet, to save compute.

        Returns:
            dict with keys:
                'left_landmarks'  : list[(x, y)] normalized 0..1, or None
                'right_landmarks' : list[(x, y)] normalized 0..1, or None
                'connections'     : list of (int, int) pairs (for skeleton drawing)
                'prediction'      : str sign name or None
                'confidence'      : float 0..1
                'stable_duration' : float seconds the current prediction has held
                'confirmed_history': list of dicts {prediction, confidence, time}
        """
        """
        Convert BGR (OpenCV default) to RGB (required by both MediaPipe
        and the TFLite landmark model).
        Setting writeable=False lets OpenCV/NumPy skip a copy internally
        when passing the array to the C++ layer.
        """
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False

        left_lm_out  = None
        right_lm_out = None

        if self._use_mediapipe:
            """
            Desktop path: use the high-level MediaPipe Holistic detector.
            results.left_hand_landmarks and right_hand_landmarks contain
            NormalizedLandmarkList objects with .landmark[i].x/.y/.z attributes.
            """
            results = self.detector.process(rgb)

            if results.left_hand_landmarks:
                left_lm_out = [(lm.x, lm.y)
                               for lm in results.left_hand_landmarks.landmark]
            if results.right_hand_landmarks:
                right_lm_out = [(lm.x, lm.y)
                                for lm in results.right_hand_landmarks.landmark]
        else:
            """
            Android path: use our TFLiteHandDetector.
            It returns a list of {'landmarks': (21,3) array, 'handedness': str}.
            We split the list into left and right hands.
            """
            hands = self.detector.detect(rgb)
            for hand in hands:
                lm_2d = [(lm[0], lm[1]) for lm in hand["landmarks"]]
                if hand["handedness"] == "Left":
                    left_lm_out = lm_2d
                else:
                    right_lm_out = lm_2d

        """
        HAND_CONNECTIONS is used by camera_screen.py to draw the skeleton.
        On desktop MediaPipe provides this constant.  On Android we define
        it ourselves — it's just the list of edges in the hand graph.
        """
        if self._use_mediapipe:
            connections = list(self._mp_holistic.HAND_CONNECTIONS)
        else:
            connections = _HAND_CONNECTIONS

        if not run_inference:
            """
            Early return: UI only needs skeleton data, not a new prediction.
            We still return the last known prediction so the label doesn't
            flicker to empty while the user is panning or adjusting their hand.
            """
            return {
                "left_landmarks":   left_lm_out,
                "right_landmarks":  right_lm_out,
                "connections":      connections,
                "prediction":       self.idx_to_class.get(self.current_prediction),
                "confidence":       self.current_confidence,
                "stable_duration":  self.stable_duration,
                "confirmed_history": list(self.confirmed_history),
            }

        """
        Extract 3-D landmark arrays for the filter bank and feature extraction.
        Zeros are used as placeholders when a hand is not detected — this keeps
        the feature vector dimension constant (required by the BiLSTM).
        """
        left_raw  = np.zeros((21, 3), dtype=np.float32)
        right_raw = np.zeros((21, 3), dtype=np.float32)

        if self._use_mediapipe:
            if results.left_hand_landmarks:
                left_raw = np.array([[lm.x, lm.y, lm.z]
                                     for lm in results.left_hand_landmarks.landmark])
            if results.right_hand_landmarks:
                right_raw = np.array([[lm.x, lm.y, lm.z]
                                      for lm in results.right_hand_landmarks.landmark])
        else:
            for hand in hands:
                if hand["handedness"] == "Left":
                    left_raw  = hand["landmarks"]
                else:
                    right_raw = hand["landmarks"]

        left_filt, right_filt = self._filter_landmarks(left_raw, right_raw)
        feat = extract_features_from_landmarks(
            left_filt, right_filt, config.FEATURE_TYPE
        )
        self.sequence_buffer.append(feat)
        self._frame_counter += 1

        """
        Run BiLSTM inference only when:
            (a) the sequence buffer is full (enough temporal context), AND
            (b) it's the Nth frame (throttle to save compute).
        The modulo check distributes inference work evenly across frames
        rather than bunching it at the start of a new buffer fill.
        """
        if (len(self.sequence_buffer) == config.SEQUENCE_LENGTH
                and self._frame_counter % config.INFER_EVERY_N_FRAMES == 0):
            self._run_inference()

        return {
            "left_landmarks":   left_lm_out,
            "right_landmarks":  right_lm_out,
            "connections":      connections,
            "prediction":       self.idx_to_class.get(self.current_prediction),
            "confidence":       self.current_confidence,
            "stable_duration":  self.stable_duration,
            "confirmed_history": list(self.confirmed_history),
        }

    def _filter_landmarks(self, left_raw, right_raw):
        """
        Apply One Euro filters to every coordinate of both hands.

        Filtering is always applied regardless of whether a hand is detected.
        When a hand is absent, left_raw/right_raw are all-zeros, and the
        filter smoothly tends toward zero — this avoids a sudden jump when
        the hand re-appears.

        The nested loop visits every (hand, landmark, axis) combination.
        With 2 × 21 × 3 = 126 filter calls per frame at 24–30 FPS, the
        overhead is negligible (~0.2 ms on a mid-range ARM CPU).
        """
        left_out  = np.zeros_like(left_raw)
        right_out = np.zeros_like(right_raw)
        for lm in range(21):
            for c in range(3):
                left_out[lm, c]  = self.filter_bank[0][lm][c].filter(left_raw[lm, c])
                right_out[lm, c] = self.filter_bank[1][lm][c].filter(right_raw[lm, c])
        return left_out, right_out

    def _run_inference(self):
        """
        Run one BiLSTM inference pass and update self.current_prediction,
        self.current_confidence, and self.stable_duration.

        Steps:
            1. Stack the sequence buffer into a (1, SEQ_LEN, FEAT_DIM) tensor.
            2. Set the input tensor and invoke the interpreter.
            3. Apply majority voting over prediction_window.
            4. Update the stability timer and emit a confirmed prediction if held.
        """
        """
        The BiLSTM expects input shape (batch=1, timesteps, features).
        np.expand_dims adds the batch dimension.
        """
        seq = np.array(self.sequence_buffer, dtype=np.float32)
        seq = np.expand_dims(seq, axis=0)

        self.interpreter.set_tensor(self.input_details[0]["index"], seq)
        self.interpreter.invoke()

        """
        output shape: (1, num_classes) — softmax probabilities.
        We take [0] to drop the batch dimension and get the probability vector.
        """
        output = self.interpreter.get_tensor(self.output_details[0]["index"])[0]

        pred_class = int(np.argmax(output))       # Class with highest probability.
        confidence = float(output[pred_class])    # Its probability (0..1).
        self.prediction_window.append(pred_class)

        """
        Wait until the prediction window is full before making any decision.
        During the fill-up period (first PREDICTION_WINDOW inference calls)
        there is not enough history for reliable voting.
        """
        if len(self.prediction_window) < self.prediction_window.maxlen:
            return

        """
        Majority voting: count how often each class appeared in the window.
        np.bincount(list_of_ints) returns a 1-D array where index i is the
        count of value i.  The dominant class is the argmax of that count.
        """
        counts   = np.bincount(self.prediction_window)
        dominant = int(np.argmax(counts))
        vote_ratio = counts[dominant] / len(self.prediction_window)

        if vote_ratio < config.VOTE_RATIO_THRESHOLD:
            """
            Inconclusive vote: the window is too mixed to be confident.
            Reset prediction state so the UI shows nothing rather than a
            wrong word.
            """
            self.current_prediction = None
            self.current_confidence = 0.0
            self.stable_prediction  = None
            self.stable_start_time  = None
            self.stable_duration    = 0.0
            return

        self.current_prediction = dominant
        self.current_confidence = confidence

        now = time.monotonic()
        if self.stable_prediction != self.current_prediction:
            """
            The dominant class just changed — start a fresh stability timer.
            We don't emit a confirmed prediction until the sign has been
            held for STABLE_REQUIRED_TIME seconds, preventing accidental
            captures of transitional hand shapes.
            """
            self.stable_prediction = self.current_prediction
            self.stable_start_time = now

        self.stable_duration = now - self.stable_start_time

        if self.stable_duration >= config.STABLE_REQUIRED_TIME:
            """
            The sign has been held steadily for long enough — confirm it.
            The de-duplicate check (last element ≠ current) prevents the
            same sign being appended on every subsequent inference call
            while the user continues to hold the pose.
            """
            sign_name = self.idx_to_class[self.current_prediction]
            if (not self.confirmed_history
                    or self.confirmed_history[-1]["prediction"] != sign_name):
                self.confirmed_history.append({
                    "prediction": sign_name,
                    "confidence": self.current_confidence,
                    "time":       now,
                })

    def close(self):
        """Cleanly shut down the landmark detector (releases MediaPipe resources)."""
        self.detector.close()


# =============================================================================
# Section 5: Hand skeleton connections
# =============================================================================

"""
HAND_CONNECTIONS defines the edges of the hand graph for skeleton drawing.
On desktop, camera_screen.py reads this from mediapipe.solutions.holistic.HAND_CONNECTIONS.
On Android, we define it ourselves here so the skeleton drawing code in
camera_screen.py works unchanged via the 'connections' key in the state dict.

The 21 MediaPipe hand landmarks are connected as follows:
    Thumb:  0-1-2-3-4
    Index:  0-5-6-7-8
    Middle: 0-9-10-11-12
    Ring:   0-13-14-15-16
    Pinky:  0-17-18-19-20
    Palm:   5-9-13-17 (cross-palm connections)

Reference:
    https://developers.google.com/mediapipe/solutions/vision/hand_landmarker/index#models
"""
_HAND_CONNECTIONS = frozenset([
    # Thumb
    (0, 1), (1, 2), (2, 3), (3, 4),
    # Index finger
    (5, 6), (6, 7), (7, 8),
    # Middle finger
    (9, 10), (10, 11), (11, 12),
    # Ring finger
    (13, 14), (14, 15), (15, 16),
    # Pinky
    (17, 18), (18, 19), (19, 20),
    # Palm (wrist to finger bases)
    (0, 5), (0, 9), (0, 13), (0, 17),
    # Palm cross connections
    (5, 9), (9, 13), (13, 17),
])
