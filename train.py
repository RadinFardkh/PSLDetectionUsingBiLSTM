import os
import json
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import config
import random
import matplotlib.pyplot as plt

# Set seeds
random.seed(config.SEED)
np.random.seed(config.SEED)
tf.random.set_seed(config.SEED)

# Loads samples
# A .json file which has the label, class and file name of every sample.
with open(os.path.join(config.PROCESSED_DIR, "samples.json")) as f:
    samples = json.load(f)
# A .json file which has classes and their labels
with open(os.path.join(config.PROCESSED_DIR, "class_map.json")) as f:
    class_map = json.load(f)

# number of classes
NUM_CLASSES = len(class_map)
print(f"Found {NUM_CLASSES} classes.")

# Split: train 70%, val 15%, test 15% (stratified in order to have equal number of samples for every class)
labels = [sample["label"] for sample in samples]
# Train and test
train_files, test_files = train_test_split(samples, test_size=0.3, stratify=labels, random_state=config.SEED)
# Validation and test
'''
Validation is a way of accuracy measurement.
After every epoch, we show some tests to our model to see how much it 
improved in answering samples it hasn't seen ever.
Validation DOESN'T IMPROVE THE MODEL ORE CHANGE THE WEIGHTS
'''
validation_files, test_files = train_test_split(test_files, test_size=0.5, stratify=[s["label"] for s in test_files],
                                                random_state=config.SEED)

print(f"Train: {len(train_files)}, Validation: {len(validation_files)}, Test: {len(test_files)}")

# Just checks the input dimension
sample_check = np.load(os.path.join(config.PROCESSED_DIR, train_files[0]["file"]))
input_dimension = sample_check.shape[1]


# Dataset creation
# These part are also exclusivity written by AI
def load_sequence(file_path, label):
    seq = tf.numpy_function(
        lambda p: np.load(p.decode()).astype(np.float32),
        [file_path], tf.float32
    )
    # sequence shape is (T, features) but unknown statically
    seq.set_shape([None, None])  # set rank to 2, but dims unknown
    return seq, label


def pad_and_augment(seq, label, is_training):
    # Pad/truncate to SEQUENCE_LENGTH
    T = tf.shape(seq)[0]
    seq = tf.cond(
        T > config.SEQUENCE_LENGTH,
        lambda: seq[:config.SEQUENCE_LENGTH],
        lambda: tf.pad(seq, [[0, config.SEQUENCE_LENGTH - T], [0, 0]])
    )
    # Training augmentation: random frame drop (mask frames)
    if is_training:
        mask = tf.cast(tf.random.uniform([config.SEQUENCE_LENGTH]) < 0.5, tf.float32)
        # ensure at least 2 frames kept
        mask = tf.cond(
            tf.reduce_sum(mask) < 2.0,
            lambda: tf.constant([1.0, 1.0] + [0.0] * (config.SEQUENCE_LENGTH - 2)),
            lambda: mask
        )
        seq = seq * tf.expand_dims(mask, 1)
    # Enforce static shape
    seq = tf.ensure_shape(seq, [config.SEQUENCE_LENGTH, input_dimension])
    return seq, label


def create_dataset(sample_list, is_training=False):
    def generator():
        for s in sample_list:
            file_path = os.path.join(config.PROCESSED_DIR, s["file"])
            yield file_path, s["label"]

    output_signature = (
        tf.TensorSpec(shape=(), dtype=tf.string),
        tf.TensorSpec(shape=(), dtype=tf.int32)
    )
    ds = tf.data.Dataset.from_generator(generator, output_signature=output_signature)
    ds = ds.map(lambda path, label: load_sequence(path, label),
                num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.map(lambda seq, label: pad_and_augment(seq, label, is_training),
                num_parallel_calls=tf.data.AUTOTUNE)
    if is_training:
        ds = ds.shuffle(200)
    ds = ds.batch(config.BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds


# Now create datasets
train_dataset = create_dataset(train_files, is_training=True)  # Train's in this part
validation_dataset = create_dataset(validation_files, is_training=False)
test_dataset = create_dataset(test_files, is_training=False)


# Model core
def build_bilstm(input_dimension, num_classes):
    """
    Builds a BiLSTM (Bidirectional Large Short-Term Memory) The difference between BiLSTM and LSTM is that BiLSTM has
    two running LSTMs which one starts from the first and the other from the last. This improves the core_model in a way
    that it can handle the prediction with full confidence over the past, the current and the future state.
    """
    # The structure of the core_model which uses Masking, BiLSTM and Dropout
    core_model = tf.keras.Sequential([
        # The Input Layer
        tf.keras.layers.Input(shape=(config.SEQUENCE_LENGTH, input_dimension)),
        # Skips zero and incorrect data
        tf.keras.layers.Masking(mask_value=0.0),
        # core of the core
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(
            config.LSTM_UNITS,  # Units of neural network
            return_sequences=False,  # Not necessary
            dropout=config.LSTM_DROPOUT,  # What percentage of neural network should sleep
            recurrent_dropout=0.0  # weakens some of the memory for generalization, but we don't wantttt
        )),
        # Creates a layer of 256 neural neurons
        tf.keras.layers.Dense(256, activation='relu'),
        # Randomly drops some neurons
        tf.keras.layers.Dropout(config.DROPOUT),
        # number of probabilities (NUM_CLASSES which are the words we added)
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    return core_model


# Build model
model = build_bilstm(input_dimension, NUM_CLASSES)

# Tells a summary of the model
model.summary()

# Callbacks are certain functions that Keras automatically executes at certain points during training.
os.makedirs(config.MODEL_DIR, exist_ok=True)
callbacks = [
    # ReduceLROnPlateau reduces the learning rate when the model stops improving.
    # This makes the model learn slower but more precise.
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=config.REDUCE_LR_FACTOR,  # Monitors validation loss
        patience=config.REDUCE_LR_PATIENCE, min_lr=1e-6  # The default learning rate is 1e-3 (config.LEARNING_RATE)
    ),
    # EarlyStopping checks if the model hasn't improved in x epochs. If yes, Stops the model completely.
    tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy', patience=config.EARLY_STOPPING_PATIENCE,  # Monitors validation accuracy
        restore_best_weights=True  # Uses the best results for the model
    ),
    # A sort of checkpoint for the model. Saves the best model in case of crashes etc.
    tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(config.MODEL_DIR, 'best_model.h5'),
        monitor='val_accuracy', save_best_only=True
    )
]

# Optimizes the model using the AdamW Optimizer
optimizer = tf.keras.optimizers.AdamW(
    learning_rate=config.LEARNING_RATE,
    decay=config.WEIGHT_DECAY,
    clipnorm=1.0
)

# Compiles the model
model.compile(optimizer=optimizer,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Trains the model
history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=config.EPOCHS,
    callbacks=callbacks,
    verbose=1,  # Progress bar
    shuffle=True  # Shuffles because the model sees the data in the same order and that's bad...
)

# Evaluate on test set
model.load_weights(os.path.join(config.MODEL_DIR, 'best_model.h5'))
test_loss, test_acc = model.evaluate(test_dataset, verbose=0)
print(f"Test accuracy: {test_acc:.4f}")

# Training Curves
# Gave this part to AI because I was lazy to write all of it :))

epochs = range(1, len(history.history["loss"]) + 1)

plt.figure(figsize=(14, 5))

# Loss
plt.subplot(1, 2, 1)
plt.plot(epochs, history.history["loss"], label="Training Loss", linewidth=2)
plt.plot(epochs, history.history["val_loss"], label="Validation Loss", linewidth=2)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.grid(True)
plt.legend()

# Accuracy
plt.subplot(1, 2, 2)
plt.plot(epochs, history.history["accuracy"], label="Training Accuracy", linewidth=2)
plt.plot(epochs, history.history["val_accuracy"], label="Validation Accuracy", linewidth=2)

# Test accuracy (constant line)
plt.axhline(
    y=test_acc,
    color="red",
    linestyle="--",
    linewidth=2,
    label=f"Test Accuracy = {test_acc:.4f}"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training, Validation and Test Accuracy")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(config.MODEL_DIR, "training_curves.png"), dpi=300)
plt.tight_layout()
plt.savefig("training_results.png", dpi=300)
plt.close()

# Save final model
model.save(os.path.join(config.MODEL_DIR, config.KERAS_MODEL_NAME))
print("Training complete.")
