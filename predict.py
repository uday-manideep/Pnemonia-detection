import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import os

# Define the class labels
class_labels = ['NORMAL', 'PNEUMONIA']

# Predict function
def predict_image(image_path, model, img_size):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")

    img = Image.open(image_path).convert('RGB')
    img = img.resize(img_size)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    prediction = model.predict(img_array)
    predicted_class = class_labels[np.argmax(prediction)]
    
    plt.imshow(img)
    plt.title(f"Predicted: {predicted_class}")
    plt.axis('off')
    plt.show()
    
    return predicted_class

if __name__ == "__main__":
    model_path = os.path.abspath("pneumonia_detection_model.h5")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    try:
        model = load_model(model_path)
        print("Model loaded successfully.")
    except OSError as e:
        print(f"Error loading model: {e}")
        exit(1)

    # Predict an example image
    img_size = (224, 224)  # Adjust if your model input size differs
    image_path = 'chest_xray/test/NORMAL/NORMAL2-IM-0058-0001.jpeg'

    try:
        predicted_class = predict_image(image_path, model, img_size)
        print(f"The predicted class is: {predicted_class}")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
