import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import json
import os
from collections import deque
import config

EXPECTED_FEATURES = 166  # 40 for pose + 126 for hand landmarks

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
indexes_to_persian = ['چی', 'درسته', 'اسم', 'غذا', 'خانه', 'خانواده', 'خوب', 'خوشحال', 'لطفا', 'مادر', 'من', 'متشکرم', 'پدر', 'سلام']


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


# NORMALIZINGGGGG
def normalise_hand_coords(hand_coords):
    wrist = hand_coords[0]
    centered = hand_coords - wrist
    dists = np.linalg.norm(centered, axis=1)
    scale = np.max(dists) + 1e-8
    return (centered / scale).flatten()


def compute_and_norm_landmarks(left_landmarks, right_landmarks):
    """
    ALWAYS returns an array of exactly EXPECTED_FEATURES=166 length.
    """
    # Start with zeros
    result = np.zeros(EXPECTED_FEATURES, dtype=np.float32)  # float32 because coordinates might get float
    # .flatten() converts the multidimensional array to a 1D array because the model expects a 1D array
    # Half of result:
    # * Left hand angles
    result[:20] = compute_joint_angles(left_landmarks)
    # * Left hand coords
    result[40:103] = normalise_hand_coords(left_landmarks).flatten()
    # Half of result
    # * Right hand angles
    result[20:40] = compute_joint_angles(right_landmarks)
    # * Right hand coords
    result[103:166] = normalise_hand_coords(right_landmarks).flatten()

    return result


# Prediction Tracking

frame_60_sequence = deque(maxlen=config.SEQUENCE_LENGTH)

# The current prediction and confidence that is for now untouched
current_prediction = None
current_confidence = 0.0

# MediaPipe Setup
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mediapipe_detector = mp.solutions.holistic.Holistic(
    static_image_mode=False,  # False if video processing, True if image processing
    model_complexity=1,  # 1 for slower but more accurate mode
    # Minimum confidence
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

camera_capture = cv2.VideoCapture(0)
if not camera_capture.isOpened():
    print("Cannot open camera")
    exit()

print("Real‑time sign recognition started. Press 'q' to quit.")

moz_counter = 0
while True:
    ret, frame = camera_capture.read()
    if not ret:
        break

    moz_counter += 1

    # Compatibility
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = mediapipe_detector.process(rgb)
    # Draw left hand
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(
            frame,
            results.left_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style()
        )

    # Draw right hand
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(
            frame,
            results.right_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style()
        )

    # Extracts raw landmarks
    left_hand_raw_data = np.zeros((21, 3), dtype=np.float32)
    right_hand_raw_data = np.zeros((21, 3), dtype=np.float32)
    if results.left_hand_landmarks:
        left_hand_arr = np.array(
            [[landmark.x, landmark.y, landmark.z] for landmark in results.left_hand_landmarks.landmark])
    if results.right_hand_landmarks:
        right_hand_arr = np.array(
            [[landmark.x, landmark.y, landmark.z] for landmark in results.right_hand_landmarks.landmark])

    # Applies causal filtering
    left_hand_filtered, right_hand_filtered = filter_landmarks(left_hand_raw_data, right_hand_raw_data)

    # Writes all landmarks in feature_data
    feature_data = compute_and_norm_landmarks(left_hand_filtered, right_hand_filtered)
    frame_60_sequence.append(feature_data)

    # Run inference every 3 frames and check if we have 60 frames (we will give our model 60 frames)
    if len(frame_60_sequence) == config.SEQUENCE_LENGTH and moz_counter % 3 == 0:
        sequence = np.array(frame_60_sequence, dtype=np.float32)  # (60, 166)
        # adds the one which shows that we give ONE input at a time
        sequence = np.expand_dims(sequence, axis=0)  # (1, 60, 166)

        # input_details[0]['index']: the input tensor and its index
        interpreter.set_tensor(input_details[0]['index'], sequence)  # Gives the input tensor the sequence
        # Runs the neural network
        interpreter.invoke()
        # Gets the output
        # Output is a list of numbers which each shows the confidence of each class.
        output = interpreter.get_tensor(
            output_details[0]['index']
        )[0]  # the 0 index gives us the prediction

        # We want the one that has the most confidence, this gives us its index
        prediction_list = np.argmax(output)
        current_prediction = int(prediction_list)
        current_confidence = float(output[prediction_list])

    # Display result
    if current_prediction is not None:
        # Find the index in class_map
        sign_name = indexes_in_clsmap[current_prediction]
        cv2.putText(
            frame,
            f"Sign: {sign_name}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"Confidence: {current_confidence * 100:.1f}%",
            (10, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 0),
            2
        )
    else:
        cv2.putText(frame, "No sign", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow('Real-time PSL Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera_capture.release()
cv2.destroyAllWindows()
mediapipe_detector.close()
print("Demo finished.")

# Done!
