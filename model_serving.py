from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import io
from PIL import Image

app = Flask(__name__)

# Load the trained model
model = load_model('model _training _service')

# Define the class labels
class_labels = {v: k for k, v in train_generator.class_indices.items()}

def prepare_image(img):
    img = img.resize((150, 150))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0
    return img

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    img = Image.open(io.BytesIO(file.read()))
    img = prepare_image(img)
    
    pred = model.predict(img)
    predicted_class = class_labels[np.argmax(pred[0])]
    confidence = np.max(pred[0])
    
    return jsonify({'category': predicted_class, 'confidence': float(confidence)})

if __name__ == '__main__':
    app.run(debug=True)
