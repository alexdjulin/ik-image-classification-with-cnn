#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
File Name: helpers.py
Author: Alexandre Donciu-Julin
Date: 2024-09-25
Description: Helper methods to interact with the dataset quickly.
"""

# Import statements
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from keras.datasets import cifar10
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator

from sklearn.metrics import confusion_matrix


def load_dataset():

    # load cifar10
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # normalise features
    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255

    # one-hot-encoding labels
    y_train = to_categorical(y_train, num_classes=10)
    y_test = to_categorical(y_test, num_classes=10)

    return x_train, y_train, x_test, y_test


def load_augmented_dataset():

    # load cifar10
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # set up image augmentation
    datagen = ImageDataGenerator(
        rotation_range=15,
        horizontal_flip=True,
        width_shift_range=0.1,
        height_shift_range=0.1
    )

    datagen.fit(x_train)

    # normalise features
    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255

    # one-hot-encoding labels
    y_train = to_categorical(y_train, num_classes=10)
    y_test = to_categorical(y_test, num_classes=10)

    return x_train, y_train, x_test, y_test, datagen


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
