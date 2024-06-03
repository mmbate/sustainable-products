import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import pandas as pd
from PIL import Image

# Load the model
model = load_model('model _training _service')

# Define the class labels
class_labels = {v: k for k, v in train_generator.class_indices.items()}

def prepare_image(img_path):
    img = Image.open(img_path)
    img = img.resize((150, 150))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0
    return img

def batch_predict(image_folder, output_file):
    results = []
    for img_name in os.listdir(image_folder):
        img_path = os.path.join(image_folder, img_name)
        img = prepare_image(img_path)
        pred = model.predict(img)
        predicted_class = class_labels[np.argmax(pred[0])]
        confidence = np.max(pred[0])
        results.append({'image': img_name, 'category': predicted_class, 'confidence': float(confidence)})
    
    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)

# Define paths
image_folder = 'path_to_new_images'
output_file = 'path_to_output_file.csv'

# Run batch prediction
batch_predict(image_folder, output_file)
