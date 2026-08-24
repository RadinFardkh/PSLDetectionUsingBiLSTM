import os

# Paths
DATA_DIR = "dataRadin"  # folder with class subfolders
PROCESSED_DIR = "processed"  # where .npy files go
MODEL_DIR = "models"    # where models go bruh

# Preprocessing
MAX_FRAMES = None  # keep all frames, or set to a fixed number (e.g., 60)
SEQUENCE_LENGTH = 60  # for padded training, fixed length (pads/truncates)

# Normalisation for coordinates (if used)
COORD_NORM = "wrist"  # "wrist" (hand translation) or "shoulders" (body)
# For angles, no extra normalisation needed


# Training
BATCH_SIZE = 32
EPOCHS = 200
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
DROPOUT = 0.3
REDUCE_LR_FACTOR = 0.5
REDUCE_LR_PATIENCE = 7
EARLY_STOPPING_PATIENCE = 20    # stop after x epochs if no loss
LSTM_UNITS = 128    # no need for 64 unless you have worm
LSTM_DROPOUT = DROPOUT  # the default dropout

# Train and TFLite Exporting
KERAS_MODEL_NAME = 'final_model.keras'
TFLITE_MODEL_NAME = "psl_model.tflite"

# Other
SEED = 42   # random, 42 is just popular
NUM_CLASSES = 39  # TODO: should change after any increase in database
