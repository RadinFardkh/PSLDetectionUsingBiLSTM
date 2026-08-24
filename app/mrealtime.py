import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import json
import os
from collections import deque
import config

"""
166 original (40 hand angles + 126 hand coords)
+ 6  (hand-to-shoulder position, per hand x,y,z)
+ 6  (elbow position relative to same-side shoulder, per arm x,y,z)
+ 2  (shoulder-elbow-wrist angle, per arm)
= 180 total
(x, y, z) means x and y in the 2d space and z is a distance to the normalizing point (in here it's wrist)
"""
EXPECTED_FEATURES = 180
print(f"Expected feature dimension per frame: {EXPECTED_FEATURES}")

# Load TFlite model and class_map.json
interpreter = tf.lite.Interpreter(
    model_path=os.path.join(config.MODEL_DIR, 'psl_model.tflite')
)
# Now tensors are available to be used in memory
interpreter.allocate_tensors()
# Shows detail about the input tensor (=which tensor should we give the data)
input_details = interpreter.get_input_details()
# Shows detail about the output tensor (=which tensor should we look at for prediction)
output_details = interpreter.get_output_details()

with open(os.path.join(config.PROCESSED_DIR, "class_map.json")) as f:
    class_map = json.load(f)
indexes_in_clsmap = {value: key for key, value in class_map.items()}


# Filters as always
# This time it's frame-by-frame which realllly helps the flow of prediction
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


# Create a bank of filters – one per hand landmark coordinate
# The bank is a list which we use to put our data and add the filter to it.
def create_filter_bank():
    bank = []
    for hand in range(2):  # left, right
        hand_bank = []
        for lm in range(21):  # 21 coords in every hand
            landmark_bank = [CausalOneEuroFilter(min_cutoff=1.0, beta=0.05) for _ in range(3)]
            hand_bank.append(landmark_bank)
        bank.append(hand_bank)
    return bank


filter_bank = create_filter_bank()


# Applies causal filter to current frame's landmarks.
def filter_landmarks(left_hand_raw_data, right_hand_raw_data):
    left_hand_filtered = np.zeros_like(left_hand_raw_data)
    right_hand_filtered = np.zeros_like(right_hand_raw_data)
    for landmark in range(21):  # 21 landmarks in each hand
        for column in range(3):  # (x, y, z)
            # filters every one of 21 landmark using its 3 coordinates (x (coord), y (coord), z (distance to wrist))
            # Adds each to the filter bank
            left_hand_filtered[landmark, column] = filter_bank[0][landmark][column].filter(
                left_hand_raw_data[landmark, column])
            right_hand_filtered[landmark, column] = filter_bank[1][landmark][column].filter(
                right_hand_raw_data[landmark, column])
    return left_hand_filtered, right_hand_filtered


# Indexes of each pose in MediaPipe
POSE_LEFT_SHOULDER_INDEX = 11
POSE_RIGHT_SHOULDER_INDEX = 12
POSE_LEFT_ELBOW_INDEX = 13
POSE_RIGHT_ELBOW_INDEX = 14

# Angle Computation (Indexing for angles)
# (x, y, z) -> angle between landmark x and landmark z (that is y)
HAND_ANGLE_INDEXES = [
    # Thumb
    (0, 1, 2),
    (1, 2, 3),
    (2, 3, 4),
    # Index finger
    (0, 5, 6),
    (5, 6, 7),
    (6, 7, 8),
    # Middle finger
    (0, 9, 10),
    (9, 10, 11),
    (10, 11, 12),
    # Ring finger
    (0, 13, 14),
    (13, 14, 15),
    (14, 15, 16),
    # Pinky
    (0, 17, 18),
    (17, 18, 19),
    (18, 19, 20),
    # Knuckle abduction angles (angles BETWEEN fingers at MCP)
    # for normalization based on wrist
    (5, 0, 9),  # index_mcp → wrist → middle_mcp
    (9, 0, 13),  # middle_mcp → wrist → ring_mcp
    (13, 0, 17),  # ring_mcp → wrist → pinky_mcp
    (1, 0, 5),  # thumb_cmc → wrist → index_mcp
    (17, 0, 5),  # pinky_mcp → wrist → index_mcp (full span over hands)
]


# Computes joint angles
def compute_joint_angles(hand_coords):
    angles = np.zeros(20, dtype=np.float32)
    for i, (a, b, c) in enumerate(HAND_ANGLE_INDEXES):
        v1 = hand_coords[a] - hand_coords[b]
        v2 = hand_coords[c] - hand_coords[b]
        dot = np.dot(v1, v2)
        norm = np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8
        angles[i] = np.arccos(np.clip(dot / norm, -1.0, 1.0))
    return angles


# Computes and returns the finalized angles for the pose
def compute_pose_angles(a, b, c):
    v1 = a - b
    v2 = c - b
    dot = np.dot(v1, v2)
    norm = np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8
    return np.arccos(np.clip(dot / norm, -1.0, 1.0))


# NORMALIZINGGGGG
def normalise_hand_coords(hand_coords):
    wrist = hand_coords[0]
    centered = hand_coords - wrist
    dists = np.linalg.norm(centered, axis=1)
    scale = np.max(dists) + 1e-8
    return (centered / scale).flatten()


# A bit easier than the hand normalizing
def normalise_pose_position(point_xyz, reference_xyz, shoulder_width):
    return (point_xyz - reference_xyz) / shoulder_width


def compute_and_norm_landmarks(left_landmarks, right_landmarks, pose_landmarks):
    """
    ALWAYS returns an array of exactly EXPECTED_FEATURES=180 length.
    """
    # Start with zeros
    result = np.zeros(EXPECTED_FEATURES, dtype=np.float32)
    result[:20] = compute_joint_angles(left_landmarks)
    result[40:103] = normalise_hand_coords(left_landmarks).flatten()
    result[20:40] = compute_joint_angles(right_landmarks)
    result[103:166] = normalise_hand_coords(right_landmarks).flatten()

    left_shoulder = pose_landmarks[POSE_LEFT_SHOULDER_INDEX]
    right_shoulder = pose_landmarks[POSE_RIGHT_SHOULDER_INDEX]
    left_elbow = pose_landmarks[POSE_LEFT_ELBOW_INDEX]
    right_elbow = pose_landmarks[POSE_RIGHT_ELBOW_INDEX]

    shoulder_width = np.linalg.norm(left_shoulder - right_shoulder) + 1e-8

    left_wrist = left_landmarks[0]
    right_wrist = right_landmarks[0]

    result[166:169] = normalise_pose_position(
        left_wrist,
        left_shoulder,
        shoulder_width
    )

    result[169:172] = normalise_pose_position(
        right_wrist,
        right_shoulder,
        shoulder_width
    )

    result[172:175] = normalise_pose_position(
        left_elbow,
        left_shoulder,
        shoulder_width
    )

    result[175:178] = normalise_pose_position(
        right_elbow,
        right_shoulder,
        shoulder_width
    )

    result[178] = compute_pose_angles(
        left_shoulder,
        left_elbow,
        left_wrist
    )

    result[179] = compute_pose_angles(
        right_shoulder,
        right_elbow,
        right_wrist
    )

    return result


# Prediction Tracking

frame_60_sequence = deque(maxlen=config.SEQUENCE_LENGTH)

current_prediction = None
current_confidence = 0.0


# MediaPipe Setup
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

mediapipe_detector = mp.solutions.holistic.Holistic(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

camera_capture = cv2.VideoCapture(0)

if not camera_capture.isOpened():
    print("Cannot open camera")
    exit()

print("Real-time sign recognition started. Press 'q' to quit.")


"""
# added
# KIVY
"""

from kivy.app import App
from kivy.clock import Clock
from kivy.graphics import Color, Rectangle
from kivy.graphics.texture import Texture
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.image import Image
from kivy.uix.label import Label
from kivy.core.window import Window

"""
# added
# Persian text support
"""

import arabic_reshaper
from bidi.algorithm import get_display

"""
# added
# Persian conversion
"""

def persian_text(text):
    return get_display(
        arabic_reshaper.reshape(text)
    )


"""
# added
# Kivy application
"""

app = App()
layout = FloatLayout()

Window.clearcolor = (0.05, 0.05, 0.05, 1)

"""
# added
# Camera widget
"""

camera_image = Image(
    size_hint=(1, 0.78),
    pos_hint={"x": 0, "y": 0.22},
    allow_stretch=True,
    keep_ratio=True
)

layout.add_widget(camera_image)


"""
# added
# Black background for prediction
"""

with layout.canvas.after:
    Color(0, 0, 0, 0.85)
    prediction_background = Rectangle(
        pos=(20, 20),
        size=(Window.width - 40, 90)
    )

"""
# added
# Prediction text
"""

prediction_label = Label(
    text=persian_text("پیش‌بینی: ---"),
    color=(1, 1, 1, 1),
    font_size=28,
    halign="right",
    valign="middle",
    size_hint=(1, 0.08),
    pos_hint={"x": 0, "y": 0.12}
)

prediction_label.bind(
    size=lambda instance, value: setattr(
        instance,
        "text_size",
        value
    )
)

layout.add_widget(prediction_label)


"""
# added
# Confidence text
"""

confidence_label = Label(
    text=persian_text("اطمینان: ---"),
    color=(0.8, 0.8, 0.8, 1),
    font_size=20,
    halign="right",
    valign="middle",
    size_hint=(1, 0.05),
    pos_hint={"x": 0, "y": 0.05}
)

confidence_label.bind(
    size=lambda instance, value: setattr(
        instance,
        "text_size",
        value
    )
)

layout.add_widget(confidence_label)


"""
# added
# Camera update
"""

def update_camera(dt):

    global current_prediction
    global current_confidence

    ret, frame = camera_capture.read()

    if not ret:
        return

    moz_counter = int(
        getattr(update_camera, "moz_counter", 0)
    ) + 1

    update_camera.moz_counter = moz_counter

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    results = mediapipe_detector.process(rgb)

    """
    # added
    # Draw left hand
    """

    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(
            frame,
            results.left_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style()
        )

    """
    # added
    # Draw right hand
    """

    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(
            frame,
            results.right_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style()
        )

    """
    # added
    # Draw pose
    """

    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_holistic.POSE_CONNECTIONS,
            mp_drawing_styles.get_default_pose_landmarks_style()
        )

    left_hand_raw_data = np.zeros(
        (21, 3),
        dtype=np.float32
    )

    right_hand_raw_data = np.zeros(
        (21, 3),
        dtype=np.float32
    )

    pose_raw_data = np.zeros(
        (33, 3),
        dtype=np.float32
    )

    if results.left_hand_landmarks:
        left_hand_raw_data = np.array(
            [
                [landmark.x, landmark.y, landmark.z]
                for landmark in results.left_hand_landmarks.landmark
            ]
        )

    if results.right_hand_landmarks:
        right_hand_raw_data = np.array(
            [
                [landmark.x, landmark.y, landmark.z]
                for landmark in results.right_hand_landmarks.landmark
            ]
        )

    if results.pose_landmarks:
        pose_raw_data = np.array(
            [
                [landmark.x, landmark.y, landmark.z]
                for landmark in results.pose_landmarks.landmark
            ],
            dtype=np.float32
        )

    left_hand_filtered, right_hand_filtered = filter_landmarks(
        left_hand_raw_data,
        right_hand_raw_data
    )

    feature_data = compute_and_norm_landmarks(
        left_hand_filtered,
        right_hand_filtered,
        pose_raw_data
    )

    frame_60_sequence.append(feature_data)

    if (
        len(frame_60_sequence) == config.SEQUENCE_LENGTH
        and moz_counter % 3 == 0
    ):

        sequence = np.array(
            frame_60_sequence,
            dtype=np.float32
        )

        sequence = np.expand_dims(
            sequence,
            axis=0
        )

        interpreter.set_tensor(
            input_details[0]['index'],
            sequence
        )

        interpreter.invoke()

        output = interpreter.get_tensor(
            output_details[0]['index']
        )[0]

        prediction_list = np.argmax(output)

        current_prediction = int(
            prediction_list
        )

        current_confidence = float(
            output[prediction_list]
        )

    """
    # added
    # Convert prediction to Persian BEFORE giving it to Kivy
    """

    if current_prediction is not None:

        sign_name = indexes_in_clsmap.get(
            current_prediction,
            "---"
        )

        prediction_label.text = persian_text(
            "پیش‌بینی: " + sign_name
        )

        confidence_label.text = persian_text(
            f"اطمینان: {current_confidence * 100:.1f}%"
        )

    """
    # added
    # Convert OpenCV frame to Kivy texture
    """

    frame_rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    height, width, channels = frame_rgb.shape

    texture = Texture.create(
        size=(width, height),
        colorfmt="rgb"
    )

    texture.blit_buffer(
        frame_rgb.tobytes(),
        colorfmt="rgb",
        bufferfmt="ubyte"
    )

    texture.flip_vertical()

    camera_image.texture = texture


"""
# added
# Update rectangle when window changes
"""

def update_rectangle(*args):

    prediction_background.pos = (
        20,
        20
    )

    prediction_background.size = (
        Window.width - 40,
        90
    )


Window.bind(
    size=update_rectangle
)


"""
# added
# Start camera update
"""

Clock.schedule_interval(
    update_camera,
    1 / 30
)


"""
# added
# Run Kivy
"""

app.root = layout
app.run()


"""
# added
# Cleanup
"""

camera_capture.release()
cv2.destroyAllWindows()

try:
    mediapipe_detector.close()
except ValueError:
    pass

print("Demo finished.")

# Done!