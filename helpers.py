#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
File Name: helpers.py
Author: Alexandre Donciu-Julin
Date: 2024-09-25
Description: Helper methods to interact with the dataset quickly.
"""

# Import statements
import os
import pickle
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import cv2

from keras.datasets import cifar10
from keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.image import resize

from sklearn.metrics import confusion_matrix

from multiprocessing import Pool, cpu_count

# define pickling paths
PICKLE_DIR = 'pickle'
os.makedirs(PICKLE_DIR, exist_ok=True)

x_train_pickle = os.path.join(PICKLE_DIR, 'x_train.pkl')
y_train_pickle = os.path.join(PICKLE_DIR, 'y_train.pkl')
x_test_pickle = os.path.join(PICKLE_DIR, 'x_test.pkl')
y_test_pickle = os.path.join(PICKLE_DIR, 'y_test.pkl')


# Resize function that takes in a batch of images
def resize_images_batch(images, target_size):
    return np.array([cv2.resize(image, target_size) for image in images])


# Function to save dataset to disk using pickle
def save_to_pickle(filename, data):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)


# Function to load dataset from pickle file
def load_from_pickle(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


# Function to check if pickled data exists
def pickled_data_exists(filenames):
    return all(os.path.exists(filename) for filename in filenames)


# Function to load and augment dataset
def load_dataset(target_size=None):

    # Load pickled data if it exists
    if target_size and pickled_data_exists([x_train_pickle, y_train_pickle, x_test_pickle, y_test_pickle]):
        # Load pickled data
        print("Loading resized images from pickle files")
        x_train = load_from_pickle(x_train_pickle)
        y_train = load_from_pickle(y_train_pickle)
        x_test = load_from_pickle(x_test_pickle)
        y_test = load_from_pickle(y_test_pickle)

    else:
        # Load CIFAR-10 data
        print("Loading CIFAR10 dataset")
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()

        # resize images
        if target_size is not None:
            print("Resizing, please wait")
            # Use multiprocessing to speed up resizing
            with Pool(cpu_count()) as pool:
                x_train = np.concatenate(
                    pool.starmap(resize_images_batch, [(batch, target_size) for batch in np.array_split(x_train, cpu_count())])
                )
                x_test = np.concatenate(
                    pool.starmap(resize_images_batch, [(batch, target_size) for batch in np.array_split(x_test, cpu_count())])
                )
            # Convert lists back to numpy arrays if not already in that format
            x_train = np.array(x_train)
            x_test = np.array(x_test)

        # Normalize features
        # x_train = x_train.astype("float32") / 255
        # x_test = x_test.astype("float32") / 255
        x_train = preprocess_input(x_train)
        x_test = preprocess_input(x_test)

        # One-hot encoding labels
        y_train = to_categorical(y_train, num_classes=10)
        y_test = to_categorical(y_test, num_classes=10)

        # Save resized data to pickle files
        save_to_pickle(x_train_pickle, x_train)
        save_to_pickle(y_train_pickle, y_train)
        save_to_pickle(x_test_pickle, x_test)
        save_to_pickle(y_test_pickle, y_test)

    return x_train, y_train, x_test, y_test


# Set up image augmentation
def data_augmentation(dataset):
    datagen = ImageDataGenerator(
        rotation_range=15,
        horizontal_flip=True,
        width_shift_range=0.1,
        height_shift_range=0.1
    )

    datagen.fit(dataset)

    return datagen

def evaluate_model(model, x_test, y_test):

    # evaluate model
    test_loss, test_acc = model.evaluate(x_test, y_test)
    print('Model Loss:', test_loss)
    print('Model Accuracy:', test_acc)


def plot_model_history(history):

    # plot loss
    plt.subplot(211)
    plt.title('Cross Entropy Loss')
    plt.plot(history.history['loss'], color='blue', label='train')
    plt.plot(history.history['val_loss'], color='red', label='val')

    # plot accuracy
    plt.subplot(212)
    plt.title('Classification Accuracy')
    plt.plot(history.history['accuracy'], color='green', label='train')
    plt.plot(history.history['val_accuracy'], color='red', label='val')
    plt.show()


def plot_confusion_matrix(model, x_test, y_test):

    # make predictions
    y_pred = np.argmax(model.predict(x_test), axis=1)
    y_test = np.argmax(y_test, axis=1)

    conf_mat = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(6, 6))
    sns.heatmap(conf_mat, annot=True, fmt='d', cmap='viridis', cbar=False)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()


def preprocess_image(image_path, target_size):
    # Load the image
    img = load_img(image_path, target_size=target_size)
    # Convert the image to a numpy array
    img_array = img_to_array(img)
    # resize image
    img_resized = resize(img_array, target_size)
    # Expand dimensions to match the model's input shape (1, height, width, channels)
    img_resized = np.expand_dims(img_array, axis=0)
    # Normalize the image (if needed, adjust according to your model's preprocessing requirements)
    img_resized = preprocess_input(img_resized)
    return img_resized


def predict_and_visualize(model, folder_path, target_size, class_mapping, rows=2, cols=5):
    # Get a list of all image files in the folder
    image_files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

    # Set up the subplot grid
    fig, axes = plt.subplots(rows, cols, figsize=(15, 6))
    axes = axes.flatten()  # Flatten the axes array for easy indexing

    # Iterate through the images and axes
    for i, ax in enumerate(axes):
        if i < len(image_files):
            image_path = os.path.join(folder_path, image_files[i])
            # Preprocess the image
            image = preprocess_image(image_path, target_size)
            # Make a prediction
            prediction = model.predict(image)
            predicted_class = np.argmax(prediction, axis=1)[0]
            # Get the class name if class_mapping is available
            predicted_label = class_mapping.get(predicted_class, f'Class {predicted_class}')

            # Display the image
            img = load_img(image_path, target_size=target_size)
            ax.imshow(img)
            ax.axis('off')  # Turn off axis lines and labels
            # Set the title as the predicted label
            ax.set_title(predicted_label)
        else:
            ax.axis('off')  # Hide empty subplots

    # Adjust layout and show the plot
    plt.tight_layout()
    plt.show()