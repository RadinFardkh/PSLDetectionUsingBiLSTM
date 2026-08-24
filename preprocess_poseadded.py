import cv2
import mediapipe as mp
import numpy as np
import os
import json
from pathlib import Path
from tqdm import tqdm
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

'''
Filters
This part just filters jittering in detection
important but don't know the details that much because of the hard math
'''


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


class OneEuroFilter:
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


# Applies the jittering filter to the landmarks (smoothens)
def smooth_landmarks(landmarks_3d, min_cutoff=1.0, beta=0.05):
    T, N, _ = landmarks_3d.shape
    smoothed = np.zeros_like(landmarks_3d)
    for n in range(N):
        for coord in range(3):
            filter_ = OneEuroFilter(min_cutoff=min_cutoff, beta=beta, d_cutoff=1.0)
            for t in range(T):
                smoothed[t, n, coord] = filter_.filter(landmarks_3d[t, n, coord])
    return smoothed


# MediaPipe setup
mediapipe_detector = mp.solutions.holistic.Holistic(
    static_image_mode=False,  # False if video processing, True if image processing
    model_complexity=1,  # 1 for slower but more accurate mode
    # Minimum confidence
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

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


# This 2 part are exceptionally hard to understand, so I gave to AI
# Computes and returns the finalized angles for the hand
def compute_joint_angles(hand_coords):
    # An array of zeros for storing
    angles = np.zeros(len(HAND_ANGLE_INDEXES), dtype=np.float32)
    # This part is for computing the angle in every index of HAND_ANGLE_INDEXES
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


# Normalizes hand coordination
def normalise_hand_coords(hand_coords):
    wrist = hand_coords[0]
    centered = hand_coords - wrist
    dists = np.linalg.norm(centered, axis=1)
    scale = np.max(dists) + 1e-8
    return centered / scale


# A bit easier than the hand normalizing
def normalise_pose_position(point_xyz, reference_xyz, shoulder_width):
    return (point_xyz - reference_xyz) / shoulder_width


# Returns the finalized and complete result of extraction
def compute_and_norm_landmarks(left_landmarks, right_landmarks, pose_landmarks):
    """
    ALWAYS returns an array of exactly EXPECTED_FEATURES=180 length.
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
    # Everything side-by-side
    # Pose coords
    left_shoulder = pose_landmarks[POSE_LEFT_SHOULDER_INDEX]
    right_shoulder = pose_landmarks[POSE_RIGHT_SHOULDER_INDEX]
    left_elbow = pose_landmarks[POSE_LEFT_ELBOW_INDEX]
    right_elbow = pose_landmarks[POSE_RIGHT_ELBOW_INDEX]
    shoulder_width = np.linalg.norm(left_shoulder - right_shoulder) + 1e-8
    # Normalizing points
    left_wrist = left_landmarks[0]
    right_wrist = right_landmarks[0]

    result[166:169] = normalise_pose_position(left_wrist, left_shoulder, shoulder_width)
    result[169:172] = normalise_pose_position(right_wrist, right_shoulder, shoulder_width)
    result[172:175] = normalise_pose_position(left_elbow, left_shoulder, shoulder_width)
    result[175:178] = normalise_pose_position(right_elbow, right_shoulder, shoulder_width)
    result[178] = compute_pose_angles(left_shoulder, left_elbow, left_wrist)
    result[179] = compute_pose_angles(right_shoulder, right_elbow, right_wrist)

    return result


# Video Processing
# Processes the video sample and returns the features.
def process_video(video_path):
    # Captures the 1-sec video from video_path
    camera_capture = cv2.VideoCapture(video_path)
    print(video_path)
    print("Opened:", camera_capture.isOpened())
    print("Frame count:", int(camera_capture.get(cv2.CAP_PROP_FRAME_COUNT)))
    raw_left_hand_arr, raw_right_hand_arr, raw_pose_arr = [], [], []
    # Frame reading loop
    while True:
        ret, frame = camera_capture.read()
        if not ret:
            break
        # Converts to RGB for compatibility because mediapipe can't read BGR
        rgbD = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = mediapipe_detector.process(rgbD)

        # The main process, then appends to the empty lists
        # (21, 3): 21 for hand landmarks, 3 for x, y and z
        left_hand_arr = np.zeros((21, 3), dtype=np.float32)
        right_hand_arr = np.zeros((21, 3), dtype=np.float32)
        # (3, 3): 33 for pose landmarks, 3 for x, y and z
        pose_arr = np.zeros((33, 3), dtype=np.float32)
        '''
        Outputs of Holistic process function
        The results have multiple returns which one is left hand, one is right hand landmarks and one is pose lms.
        landmark.x and landmark.y is x and y axis on 2d space
        landmark.z is kinda normalising and shows the distance to wrist
        '''
        if results.left_hand_landmarks:
            left_hand_arr = np.array(
                [[landmark.x, landmark.y, landmark.z] for landmark in results.left_hand_landmarks.landmark])
        if results.right_hand_landmarks:
            right_hand_arr = np.array(
                [[landmark.x, landmark.y, landmark.z] for landmark in results.right_hand_landmarks.landmark])
        if results.pose_landmarks:
            pose_arr = np.array(
                [[landmark.x, landmark.y, landmark.z] for landmark in results.pose_landmarks.landmark])
        # Appending results to raw array
        raw_left_hand_arr.append(left_hand_arr)
        raw_right_hand_arr.append(right_hand_arr)
        raw_pose_arr.append(pose_arr)
    camera_capture.release()

    final_left_arr = np.array(raw_left_hand_arr)
    final_right_arr = np.array(raw_right_hand_arr)
    final_pose_arr = np.array(raw_pose_arr)

    # Applies jittering filter (Pose doesn't need because the filters are resource-demanding, and it's not necessary)
    final_left_arr = smooth_landmarks(final_left_arr)
    final_right_arr = smooth_landmarks(final_right_arr)

    # Build processed — every frame guaranteed same dimension
    processed = np.zeros((len(final_left_arr), EXPECTED_FEATURES), dtype=np.float32)
    for i in range(len(final_left_arr)):
        processed[i] = compute_and_norm_landmarks(final_left_arr[i], final_right_arr[i], final_pose_arr[i])

    return processed


# Main func
def main():
    # Makes the 'processed' folder and subfolders
    Path(config.PROCESSED_DIR).mkdir(parents=True, exist_ok=True)
    # Finds every class in the data folder
    class_dirs = sorted([d for d in os.listdir(config.DATA_DIR) if os.path.isdir(os.path.join(config.DATA_DIR, d))])
    # Labels every class
    class_map = {cls: idx for idx, cls in enumerate(class_dirs)}
    # Dumps it into class_map.json
    with open(os.path.join(config.PROCESSED_DIR, "class_map.json"), "w") as f:
        json.dump(class_map,  # Writing class_map to
                  f,  # f
                  indent=2  # Beautify!!!!!
                  )

    # Processes every video and dumps everything in 'processed' subfolders
    all_samples = []
    for cls in class_dirs:
        # Path of data
        cls_path = os.path.join(config.DATA_DIR, cls)
        # Finds the video_files
        video_files = sorted(Path(cls_path).glob("*.mp4"))
        moz = 120
        # tqdm makes a progress bar, for having a better look when processing, took a while to understand!
        for vf in tqdm(video_files):
            frames = process_video(str(vf))
            # Saves them as .npy
            out_name = f"{cls}_{moz}.npy"
            moz += 1
            np.save(os.path.join(config.PROCESSED_DIR, out_name), frames)
            all_samples.append({
                "file": out_name,
                "class": cls,
                "label": class_map[cls]
            })
    # A .json file which has the label, class and file name of every sample.
    with open(os.path.join(config.PROCESSED_DIR, "samples.json"), "w") as f:
        json.dump(all_samples, f, indent=2)
    print(f"Preprocessing done. Total samples: {len(all_samples)}")


if __name__ == "__main__":
    main()
