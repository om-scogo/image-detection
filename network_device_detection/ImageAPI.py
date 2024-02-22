from flask import Flask, request, jsonify
import os
import numpy as np
import pandas as pd
from keras.models import load_model
from PIL import Image
import pytesseract
import google.generativeai as genai
import ai21

app = Flask(__name__)

gemini_api_key = "..."
genai.configure(api_key = gemini_api_key)

# Constants
classes = ["device", "rack"]
IMAGE_WIDTH = 128
IMAGE_HEIGHT = 128

# Load the saved model
model_path = '/Users/omachrekar/Desktop/SCOGO/network_device_detection/model_1.h5'
model = load_model(model_path)

def predict_image_class(image_path):
    fileImage = Image.open(image_path).convert("RGB").resize([IMAGE_WIDTH, IMAGE_HEIGHT], Image.LANCZOS)
    image = np.array(fileImage)
    myimage = image.reshape(1, IMAGE_WIDTH, IMAGE_HEIGHT, 3)

    # Make predictions
    my_predicted_image = model.predict(myimage)

    # Assuming threshold of 0.5 for binary classification
    predicted_class = classes[int(my_predicted_image > 0.5)]
    return predicted_class

def extract_text_from_image(image_path):
    try:
        img = Image.open(image_path)
        text = pytesseract.image_to_string(img)
        prompt = f"""
        You are an agent which only goal is to extract the MAC address and device address from the given text. The text
         can be vague becuase it is extracted from image using OCR so you should be careful which is correct MAC address. 
         If no MAC address is found then return not found. Here is the text from which you have to extract MAC address : {text}.
        """

        model = genai.GenerativeModel('gemini-pro')
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error: {str(e)}"

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file:
        image_path = file.filename
        file.save(image_path)
        predicted_class = predict_image_class(image_path)
        if predicted_class == "device":
            text = extract_text_from_image(image_path)
            return jsonify({'predicted_class': predicted_class, 'MAC Address': text})
        os.remove(image_path)  
        return jsonify({'predicted_class': predicted_class})

if __name__ == '__main__':
    app.run(debug=True)
