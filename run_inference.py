import tensorflow as tf
import numpy as np

# Load the saved Keras model (h5 format)
model = tf.keras.models.load_model("/mnt/raiddrive/AI_models/text_classification_dense_input.h5")

# Define sample examples to classify
examples = [
    "The movie was great!",
    "The movie was okay.",
    "The movie was terrible..."
]

# Function to preprocess text into numerical features or indices
def text_to_feature(text):
    # Simple keyword-based mapping for demonstration
    if "great" in text.lower():
        return 0.8
    elif "okay" in text.lower():
        return 0.5
    elif "terrible" in text.lower():
        return 0.2
    else:
        return 0.5  # Default value if no keywords match

# Preprocess each example and convert it into a format suitable for the new model
features = np.array([[text_to_feature(example)] for example in examples], dtype=np.float32)

# Print the preprocessed features to verify
print("Preprocessed Features:\n", features)

# Run predictions on the processed input data
predictions = model.predict(features)

# Print the results
for i, prediction in enumerate(predictions):
    print(f"Example: {examples[i]}")
    print(f"Prediction: {prediction[0]}\n")
