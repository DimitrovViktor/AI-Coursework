import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Load model
model_path = 'food_recognition_model_2.h5'
model = tf.keras.models.load_model(model_path)

# Directory structure
class_labels = {
    'apple_pie': 0,
    'bibimbap': 1,
    'cannoli': 2,
    'edamame': 3,
    'falafel': 4,
    'french_toast': 5,
    'ice_cream': 6,
    'ramen': 7,
    'sushi': 8,
    'tiramisu': 9
}

# Reverse dictionary to map indices to labels
index_to_class_labels = {v: k for k, v in class_labels.items()}

# Preprocessing function
def preprocess_image(img_path, target_size=(224, 224)):
    img = Image.open(img_path)
    img = img.resize(target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  
    return img_array

# Path test image
test_image_path = 'ramen.jpg'

# Preprocess image
preprocessed_image = preprocess_image(test_image_path)

# Predict class
predictions = model.predict(preprocessed_image)
predicted_class_index = np.argmax(predictions, axis=1)[0]
predicted_class_label = index_to_class_labels[predicted_class_index]

print(f"Predicted class: {predicted_class_label}")
