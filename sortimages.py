import os
import shutil

from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing import image
import numpy as np

# Replace with the path to your folder of images
folder_path = 'p'

# Load the VGG16 model with weights pre-trained on ImageNet
model = VGG16(weights='imagenet')

# Function to load and preprocess an image
def load_image(image_path):
    # Load the image and resize it to the input size of the VGG16 model
    img = image.load_img(image_path, target_size=(224, 224))

    # Convert the image to a NumPy array
    x = image.img_to_array(img)

    # Add a batch dimension to the array
    x = np.expand_dims(x, axis=0)

    # Preprocess the image for the VGG16 model
    x = model.preprocess_input(x)


    return x

# Create a dictionary to map class labels to subfolder names
label_to_subfolder = {
    0: 'animals',
    1: 'plants',
    2: 'cars',
    3: 'scenes'
}

# Loop through all of the images in the folder
for filename in os.listdir(folder_path):
    # Load and classify the image
    image = load_image(os.path.join(folder_path, filename))
    prediction = model.predict(image)

    # Determine the most likely class and the corresponding subfolder name
    label = np.argmax(prediction)
    subfolder = label_to_subfolder[label]
    
    # Create the subfolder if it doesn't already exist
    subfolder_path = os.path.join(folder_path, subfolder)
    if not os.path.exists(subfolder_path):
        os.makedirs(subfolder_path)
    
    # Move the image to the subfolder
    shutil.move(os.path.join(folder_path, filename), os.path.join(subfolder_path, filename))
