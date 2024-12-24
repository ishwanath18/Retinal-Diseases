from flask import Flask, render_template, request, jsonify
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import os

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
model = load_model('D:\\mega project work\\retinalOCT_model.h5')  # Update path if needed

# Function to preprocess the uploaded image
def preprocess_image(image_path, target_size=(224, 224)):
    img = Image.open(image_path)
    img_resized = img.resize(target_size)
    img_array = np.array(img_resized) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Define the upload folder
UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Max file size 16MB

# Class labels (update as per your model's classes)
class_labels = {0: 'CNV', 1: 'DME', 2: 'DRUSEN', 3: 'NORMAL'}

# Home route that displays the HTML form
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle image upload and prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    if file:
        # Save the uploaded image
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(image_path)
        
        # Preprocess the image
        preprocessed_image = preprocess_image(image_path)
        
        # Make predictions using the trained model
        predictions = model.predict(preprocessed_image)
        print(predictions)  # Debugging line to check the raw output
        
        predicted_class = np.argmax(predictions, axis=1)
        
        # Return the prediction result
        predicted_label = class_labels.get(predicted_class[0], 'Unknown Class')
        return jsonify({'prediction': predicted_label, 'image_path': image_path})


if __name__ == '__main__':
    app.run(debug=True)
