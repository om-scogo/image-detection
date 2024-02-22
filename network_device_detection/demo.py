import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from keras.models import load_model

# Load the saved model
model_path = 'network_device_detection/mango_and_apple.h5'
model = load_model(model_path)

# Define the classes
classes = ["mango", "apple"]

# Load and preprocess the image
def preprocess_image(image_path, target_size):
    img = Image.open(image_path).convert("RGB").resize(target_size, Image.LANCZOS)
    img_array = np.array(img)
    img_array = img_array / 255.0  # Normalize pixel values
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Path to the unseen image
image_path = "network_device_detection/imgs/img4.jpeg"

# Preprocess the image
input_image = preprocess_image(image_path, (128, 128))

# Make predictions
predicted_probabilities = model.predict(input_image)[0]
predicted_class_index = np.argmax(predicted_probabilities)
predicted_class = classes[predicted_class_index]

print(f"The predicted class is: {predicted_class} with probability {predicted_probabilities[predicted_class_index]}")
