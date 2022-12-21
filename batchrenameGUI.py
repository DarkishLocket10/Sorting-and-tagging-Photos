import cv2
import numpy as np
import tensorflow as tf
import os
import re
from PyQt5 import QtWidgets, QtGui
from PyQt5.QtWidgets import QMessageBox

class MainWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()

        # Load the MobileNet model
        self.model = tf.keras.applications.MobileNet()

        # Load the imagenet labels file
        with open('imagenet_labels.txt') as f:
            self.imagenet_labels = [line.strip() for line in f.readlines()]

        # Create a button to select the folder
        self.folder_button = QtWidgets.QPushButton('Select Folder', self)
        self.folder_button.clicked.connect(self.select_folder)

        # Create a start button
        self.start_button = QtWidgets.QPushButton('Start', self)
        self.start_button.clicked.connect(self.start)

        # Create a label to display the selected folder
        self.folder_label = QtWidgets.QLabel(self)

        # Set the background of both buttons to light blue
        self.folder_button.setStyleSheet('background-color: #cfe2f3')
        self.start_button.setStyleSheet('background-color: #cfe2f3')

        # Set the Font of everything to Lato
        font = QtGui.QFont()
        font.setFamily('Lato')
        font.setPointSize(14)
        self.setFont(font)
        

        # Create a layout to hold the widgets
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.folder_button)
        layout.addWidget(self.folder_label)
        layout.addWidget(self.start_button)

    def select_folder(self):
        # Open a file dialog to select the folder
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, 'Select Folder')

        # Update the label to display the selected folder
        self.folder_label.setText(folder)

    def start(self):
        # Get the selected folder
        folder = self.folder_label.text()

        # Get a list of all the image files in the folder
        image_files = [f for f in os.listdir(folder) if f.endswith('.jpg')]

        # Loop through the image files
        for image_file in image_files:
            # Read the image file
            image = cv2.imread(os.path.join(folder, image_file))

            # Preprocess the image for the MobileNet model
            image = cv2.resize(image, (224, 224))
            image = np.expand_dims(image, axis=0)
            image = tf.keras.applications.mobilenet.preprocess_input(image)

            # Use the MobileNet model to classify the image
            predictions = self.model.predict(image)

            # Get the top prediction
            top_prediction = np.argmax(predictions[0])

            # Get the label for the top prediction
            label = self.imagenet_labels[top_prediction]

            # Replace any invalid characters in the label with an underscore
            label = re.sub(r'[^\w\s]', '_', label)

            # Get the file extension of the image file
            file_extension = os.path.splitext(image_file)[1]

            # Generate the new file name
            new_name = '{}{}'.format(label, file_extension)

            # If the new file name already exists, add an incremented number to the file name
            i = 1
            while os.path.exists(os.path.join(folder, new_name)):
                new_name = '{}_{}{}'.format(label, i, file_extension)
                i += 1

            # Rename the image file
            os.rename(os.path.join(folder, image_file), os.path.join(folder, new_name))

        
        # Show a message box to indicate that the batch process has finished
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Information)
        msg.setText("Batch process has finished")
        msg.setWindowTitle("Batch Process")
        msg.setStandardButtons(QMessageBox.Ok)
        msg.exec_()

if __name__ == '__main__':
    # Create an instance of QApplication
    app = QtWidgets.QApplication([])

    # Create an instance of the MainWindow class
    window = MainWindow()

    # Show the window
    window.show()

    # Start the event loop
    app.exec_()
