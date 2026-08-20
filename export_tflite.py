import tensorflow as tf
import numpy as np
import os
import config
import json

# Loads samples
with open(os.path.join(config.PROCESSED_DIR, "samples.json")) as f:
    samples = json.load(f)

# This is what you should do that makes quantization possible
# you take a small chunk of data and feed it into the quantizer.
rep_samples = samples[:200]
def representative_dataset_gen():
    for s in rep_samples:
        sequence = np.load(os.path.join(config.PROCESSED_DIR, s["file"])).astype(np.float32)
        if len(sequence) > config.SEQUENCE_LENGTH:
            sequence = sequence[:config.SEQUENCE_LENGTH]
        else:
            pad_length = config.SEQUENCE_LENGTH - len(sequence)
            sequence = np.pad(sequence, ((0, pad_length), (0, 0)), mode='constant')
        sequence = np.expand_dims(sequence, axis=0).astype(np.float32)
        yield [sequence]


# Load trained Keras model
model = tf.keras.models.load_model(os.path.join(config.MODEL_DIR, 'final_model.keras'))

# Converts the model to tflite for Android compatibility
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset_gen
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
    tf.lite.OpsSet.SELECT_TF_OPS
]
converter.inference_input_type = tf.float32
converter.inference_output_type = tf.float32

tflite_model = converter.convert()
tflite_path = os.path.join(config.MODEL_DIR, config.TFLITE_MODEL_NAME)
with open(tflite_path, 'wb') as f:
    f.write(tflite_model)
print(f"TFLite model saved to {tflite_path}")
