# Import necessary libraries
from flask import Flask, request, render_template, redirect, url_for
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
import cv2
from werkzeug.utils import secure_filename
import os

# Initialize the Flask application
app = Flask(__name__)

# Define paths
UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the trained models
binary_model = load_model(r'B-DR-MobileNet2V.h5')
multi_model = load_model(r'M-DR-MobileNet2V.h5')

# Define class names
binary_class_names = np.array(['DR', 'No_DR'])
multi_class_names = np.array(['Mild', 'Moderate', 'Proliferate_Dr', 'Severe'])

# Route to handle the home page
@app.route('/')
def index():
    return render_template('index.html')

# Helper function to process the image
def process_image(file):
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    # Preprocess the image
    image = cv2.imread(file_path)
    image_resized = cv2.resize(image, (224, 224))  # Resize to (224, 224)
    image = np.expand_dims(image_resized, axis=0)  # Expand dimensions for model

    return image, file_path

# Route to handle binary prediction
@app.route('/predict_binary', methods=['POST'])
def predict_binary():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']

    if file.filename == '':
        return redirect(request.url)

    if file:
        image, file_path = process_image(file)
        prediction = binary_model.predict(image)
        predicted_class = binary_class_names[np.argmax(prediction)]

        show_multi_class = predicted_class == 'DR'

        return render_template('index.html', 
                               binary_prediction=predicted_class, 
                               binary_image_path=file_path,
                               show_multi_class=show_multi_class)

# Route to handle multi-class prediction
@app.route('/predict_multi', methods=['POST'])
def predict_multi():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']

    if file.filename == '':
        return redirect(request.url)

    if file:
        image, file_path = process_image(file)
        prediction = multi_model.predict(image)
        predicted_class = multi_class_names[np.argmax(prediction)]

        return render_template('index.html', 
                               binary_prediction='DR',  # We know it's DR if we're here
                               binary_image_path=file_path,
                               show_multi_class=True,
                               multi_prediction=predicted_class, 
                               multi_image_path=file_path)

# Run the app
if __name__ == '__main__':
    app.run(debug=True)