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
import cv2
import pickle
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from keras.datasets import cifar10
from keras.utils import to_categorical
from keras.applications.resnet50 import preprocess_input

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.image import resize

from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix

from multiprocessing import Pool, cpu_count


def resize_images_batch(images: list, target_size: tuple) -> np.ndarray:
    """resize a list of images to a target size
    I used this method to resize images in parallel using multiprocessing at first
    before implementing and using the Lambda layer in the model and the resize method from TensorFlow

    Args:
        images (list): list of numpy images
        target_size (tuple): target size to resize the images to

    Returns:
        np.ndarray: numpy array of resized images
    """
    return np.array([cv2.resize(image, target_size) for image in images])


def save_to_pickle(filename: str, data: np.ndarray) -> None:
    """Pickle a dataset to disk for future use

    Args:
        filename (str): file path to save the data
        data (np.ndarray): dataset to save
    """
    with open(filename, 'wb') as f:
        pickle.dump(data, f)


def load_from_pickle(filename: str) -> np.ndarray:
    """Loads a pickled dataset from disk to avoid re-importing it

    Args:
        filename (str): file path to load the data from

    Returns:
        np.ndarray: dataset loaded from the pickle file
    """
    with open(filename, 'rb') as f:
        return pickle.load(f)


def pickled_data_exists(filenames: list) -> bool:
    """Checks if all the files in the list exist

    Args:
        filenames (list): list of file paths to check

    Returns:
        bool: True if all files exist, False otherwise
    """
    return all(os.path.exists(filename) for filename in filenames)


def load_dataset(model: str = None, target_size: tuple = None) -> tuple:
    """Load the CIFAR-10 dataset and resize it if needed

    Args:
        model (str, optional): Specify 'ResNet50' to use the keras preprocessing method for this dataset. Defaults to None.
        target_size (tuple, optional): Target size to resize the images to. Used before the Lambda resize layer. Defaults to None.

    Returns:
        tuple: training and testing data and labels
    """

    # defines pickle paths
    pickle_dir = 'pickle'
    os.makedirs(pickle_dir, exist_ok=True)

    x_train_pickle = os.path.join(pickle_dir, 'x_train.pkl')
    y_train_pickle = os.path.join(pickle_dir, 'y_train.pkl')
    x_test_pickle = os.path.join(pickle_dir, 'x_test.pkl')
    y_test_pickle = os.path.join(pickle_dir, 'y_test.pkl')

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
        if model == 'ResNet50':
            # keras offers a preprocessing method for ResNet50
            x_train = preprocess_input(x_train)
            x_test = preprocess_input(x_test)
        else:
            # normalise RGB colors from 0 to 1
            x_train = x_train.astype("float32") / 255
            x_test = x_test.astype("float32") / 255

        # One-hot encoding labels
        y_train = to_categorical(y_train, num_classes=10)
        y_test = to_categorical(y_test, num_classes=10)

        # Save resized data to pickle files
        save_to_pickle(x_train_pickle, x_train)
        save_to_pickle(y_train_pickle, y_train)
        save_to_pickle(x_test_pickle, x_test)
        save_to_pickle(y_test_pickle, y_test)

    return x_train, y_train, x_test, y_test


def data_augmentation(dataset: np.ndarray) -> ImageDataGenerator:
    """Initiate and returns an ImageDataGenerator object for data augmentation

    Args:
        dataset (np.ndarray): dataset to augment

    Returns:
        ImageDataGenerator: ImageDataGenerator object for data augmentation
    """
    datagen = ImageDataGenerator(
        rotation_range=15,
        horizontal_flip=True,
        width_shift_range=0.1,
        height_shift_range=0.1
    )

    datagen.fit(dataset)

    return datagen


def evaluate_model(model, x_test, y_test) -> None:
    """Evaluate the model on the test dataset and prints the results

    Args:
        model (keras sequential model): model to evaluate
        x_test (np.ndarray): test dataset
        y_test (np.ndarray): test labels
    """

    # evaluate model
    test_loss, test_acc = model.evaluate(x_test, y_test)

    # make predictions
    y_pred = np.argmax(model.predict(x_test), axis=1)
    y_test = np.argmax(y_test, axis=1)

    # calculate metrics
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    accuracy = accuracy_score(y_test, y_pred)

    # print results
    print(f'Model Loss: {test_loss}')
    print(f'Model Accuracy: {test_acc}')
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1-Score: {f1}")
    print(f"Accuracy Score: {accuracy}")


def plot_model_history(history: dict) -> None:
    """Plot the model's training history

    Args:
        history (dict): model's training history
    """

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


def plot_confusion_matrix(model, x_test, y_test) -> None:
    """Plot the confusion matrix for the model's predictions

    Args:
        model (keras sequential model): model to evaluate
        x_test (np.ndarray): test dataset
        y_test (np.ndarray): test labels
    """

    # make predictions
    y_pred = np.argmax(model.predict(x_test), axis=1)
    y_test = np.argmax(y_test, axis=1)

    # calculate confusion matrix
    conf_mat = confusion_matrix(y_test, y_pred)

    # plot confusion matrix
    plt.figure(figsize=(6, 6))
    sns.heatmap(conf_mat, annot=True, fmt='d', cmap='viridis', cbar=False)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()


def preprocess_image(image_path: str, target_size: tuple) -> np.ndarray:
    """Preprocess an image for prediction by a model

    Args:
        image_path (str): path to the image file
        target_size (tuple): target size to resize the image to

    Returns:
        np.ndarray: preprocessed image
    """
    # Load the image
    img = load_img(image_path, target_size=target_size)

    # Convert the image to a numpy array
    img_array = img_to_array(img)

    # resize image
    img_resized = resize(img_array, target_size)

    # Expand dimensions to match the model's input shape
    img_resized = np.expand_dims(img_array, axis=0)

    # Normalize the image
    img_resized = preprocess_input(img_resized)  # for ResNet50
    # x_train = x_train.astype("float32") / 255  # for other models

    return img_resized


def predict_and_visualize(model, folder_path: str, target_size: tuple, class_mapping: dict, cols=5) -> None:
    """Predict and visualize images in a folder using the model

    Args:
        model (keras sequential model): model to use for prediction
        folder_path (str): path to the folder containing the images
        target_size (tuple): target size to resize the images to
        class_mapping (dict): mapping of class indices to class names
        cols (int, optional): number of columns for the plot. Rows will be calculated from it. Defaults to 5.
    """

    # Get a list of all image files in the folder
    image_files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

    # calculate the number of rows needed
    rows = len(image_files) // cols + 1

    # Set up the subplot grid
    _, axes = plt.subplots(rows, cols, figsize=(15, 6))
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
