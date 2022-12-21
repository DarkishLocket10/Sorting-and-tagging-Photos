import os
import tensorflow as tf

from PIL import Image

def add_tag_to_photo(filepath, tag):
    # Open the photo using the Pillow library
    image = Image.open(filepath)
    
    # Add the tag to the photo's metadata
    image.info['Keywords'] = tag
    
    # Save the modified photo
    image.save(filepath)

def main():
    # Load the MNIST dataset
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # Preprocess the data
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255

    # Build the CNN model
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    # Compile the model
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    # Train the model
    model.fit(x_train, y_train, epochs=5)

    # Evaluate the model on the test dataset
    test_loss, test_acc = model.evaluate(x_test, y_test)

    # Print the test accuracy
    print('Test accuracy:', test_acc)

    # Define a function to rename and tag a photo based on its predicted label
    def rename_and_tag_photo(photo, label):
        # Extract the filename and extension from the full path
        filepath, extension = os.path.splitext(photo)
        filename = os.path.basename(filepath)

        # Rename the photo using the predicted label
        new_filename = f'{label}{extension}'
        new_filepath = filepath.replace(filename, new_filename)
        os.rename(photo, new_filepath)

        # Add the predicted label as a tag to the photo
        # (Assumes you are using a software that supports tags)
        add_tag_to_photo(new_filepath, label)

    # Predict the labels for new photos
    new_photos = [ 
        'P1344375.jpg'
     ] # List of new photos
    predictions = model.predict(new_photos)

    # Rename and tag the photos using the predictions
    for i, photo in enumerate(new_photos):
        label = predictions[i]
        rename_and_tag_photo(photo, label)

