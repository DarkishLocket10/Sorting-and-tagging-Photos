import cv2
import numpy as np
import tensorflow as tf
import os
import re

# Load the MobileNet model
model = tf.keras.applications.MobileNet()

# Load the imagenet labels file
with open('imagenet_labels.txt') as f:
    imagenet_labels = [line.strip() for line in f.readlines()]

# Get a list of all the image files in the folder
image_files = [f for f in os.listdir('photos') if f.endswith('.jpg')]

# Loop through the image files
for image_file in image_files:
    # Read the image file
    image = cv2.imread(os.path.join('photos', image_file))

    # Preprocess the image for the MobileNet model
    image = cv2.resize(image, (224, 224))
    image = np.expand_dims(image, axis=0)
    image = tf.keras.applications.mobilenet.preprocess_input(image)

    # Use the MobileNet model to classify the image
    predictions = model.predict(image)

    # Get the top prediction
    top_prediction = np.argmax(predictions[0])

    # Get the label for the top prediction
    label = imagenet_labels[top_prediction]

    # Replace any invalid characters in the label with an underscore
    label = re.sub(r'[^\w\s]', '_', label)

    # Rename the image file with the label
    new_name = '{}_{}.jpg'.format(os.path.splitext(image_file)[0], label)
    os.rename(os.path.join('photos', image_file), os.path.join('photos', new_name))
