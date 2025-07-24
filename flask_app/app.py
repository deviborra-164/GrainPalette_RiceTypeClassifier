import os
import numpy as np
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Flask app setup
app = Flask(__name__)
UPLOAD_FOLDER = os.path.join('static', 'uploaded')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load trained model
MODEL_PATH = os.path.join('..', 'saved_models', 'mobilenetv2_rice_subset_best.h5')
model = load_model(MODEL_PATH)

# Define rice class labels
class_names = ['Arborio', 'Basmati', 'Ipsala', 'Jasmine', 'Karacadag']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return render_template('index.html', prediction="❌ No file uploaded.")

    img = request.files['image']
    if img.filename == '':
        return render_template('index.html', prediction="❌ No file selected.")

    # Save uploaded image
    filename = secure_filename(img.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    img.save(filepath)

    # Preprocess image
    img_loaded = load_img(filepath, target_size=(224, 224))
    img_array = img_to_array(img_loaded) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)[0]
    confidence = np.max(prediction)
    predicted_class = class_names[np.argmax(prediction)]

    if confidence < 0.7:
        result = f"❗ This doesn't seem to be a rice grain image. (Confidence: {confidence:.2f})"
    else:
        result = f"✅ Predicted Rice Type: {predicted_class} (Confidence: {confidence:.2f})"

    return render_template('index.html', prediction=result, image_path=filepath)

if __name__ == '__main__':
    app.run(debug=True)

