
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers, optimizers
from keras.applications import MobileNetV2
from keras.models import Sequential
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
IMAGESIZE = 128

def minimalmodel(image_size=128, num_classes=16):
    mobilenet =tf.keras.applications.MobileNetV3Small(
        input_shape=(IMAGESIZE,IMAGESIZE,3),
        minimalistic=True,
        alpha=1.0,
        include_top=False,
        include_preprocessing=False,
        weights='imagenet'
    )
    mobilenet._name = "mnet"
    for layer in mobilenet.layers:
        layer._name = layer._name.lower()
    model = Sequential()
    model.add(mobilenet)
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dropout(0.35))
    model.add(layers.Dense(num_classes, activation='softmax'))
    
    model.compile(
        loss='categorical_crossentropy',
        optimizer=optimizers.Adam(learning_rate=0.00002),
        metrics=['accuracy']
    )
    
    return model

def deep_tiny():
    model = Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMAGESIZE,IMAGESIZE,3)))
    model.add(layers.MaxPooling2D((2, 2),strides=(2, 2)))

    model.add(layers.Conv2D(16, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2),strides=(2, 2)))

    model.add(layers.Conv2D(8, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2),strides=(2, 2)))

    model.add(layers.Flatten())

    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dropout(0.15))

    model.add(layers.Dense(24, activation='relu'))
    model.add(layers.Dropout(0.15))

    model.add(layers.Dense(16, activation='softmax'))
    model.compile(
          loss='categorical_crossentropy',
          optimizer=optimizers.Adam(learning_rate=0.00001),
          metrics=['accuracy']
      )
    return model

def deep_tiny_x5():
    model = Sequential()
    model.add(layers.Conv2D(32, (5, 5), activation='relu', input_shape=(IMAGESIZE,IMAGESIZE,3)))
    model.add(layers.MaxPooling2D((2, 2),strides=(2, 2)))

    model.add(layers.Conv2D(16, (5, 5), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2),strides=(2, 2)))

    model.add(layers.Conv2D(8, (5, 5), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2),strides=(2, 2)))

    model.add(layers.Flatten())

    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dropout(0.15))

    model.add(layers.Dense(24, activation='relu'))
    model.add(layers.Dropout(0.15))

    model.add(layers.Dense(16, activation='softmax'))
    model.compile(
          loss='categorical_crossentropy',
          optimizer=optimizers.Adam(learning_rate=0.00001),
          metrics=['accuracy']
      )
    return model

def deep_tiny_x5_xl():
    model = Sequential()

    model.add(layers.Conv2D(32, (5, 5), activation='relu', input_shape=(IMAGESIZE,IMAGESIZE,3)))
    model.add(layers.MaxPooling2D((2, 2),strides=(2, 2)))

    model.add(layers.Conv2D(32, (5, 5), activation='relu', input_shape=(IMAGESIZE,IMAGESIZE,3)))
    model.add(layers.MaxPooling2D((2, 2),strides=(2, 2)))

    model.add(layers.Conv2D(32, (5, 5), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2),strides=(2, 2)))

    model.add(layers.Conv2D(16, (5, 5), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2),strides=(2, 2)))

    model.add(layers.Flatten())

    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dropout(0.15))

    model.add(layers.Dense(24, activation='relu'))
    model.add(layers.Dropout(0.15))

    model.add(layers.Dense(8, activation='softmax'))
    model.compile(
          loss='categorical_crossentropy',
          optimizer=optimizers.Adam(learning_rate=0.00001),
          metrics=['accuracy']
      )
    return model

def teensy_model():
    model = Sequential()
    model.add(layers.Conv2D(48, (3, 3), activation='relu', input_shape=(IMAGESIZE,IMAGESIZE,3)))
    model.add(layers.MaxPooling2D((2, 2),strides=(2, 2)))

    model.add(layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((4, 4),strides=(4, 4)))

    model.add(layers.Flatten())

    model.add(layers.Dense(24, activation='relu'))
    model.add(layers.Dropout(0.15))

    model.add(layers.Dense(16, activation='softmax'))
    model.compile(
          loss='categorical_crossentropy',
          optimizer=optimizers.Adam(learning_rate=0.00001),
          metrics=['accuracy']
      )
    return model

def vanilla_mobilenet():
    model = tf.keras.applications.mobilenet_v2.MobileNetV2(
      input_shape=(IMAGESIZE,IMAGESIZE,3),
      alpha=1.0,
      include_top=True,
      weights=None,
      classes=30,
      classifier_activation='softmax'
      )
    model.compile(
          loss='categorical_crossentropy',
          optimizer=optimizers.Adam(learning_rate=0.00002),
          metrics=['accuracy']
      )
    return model


def le_net_model():
    model = Sequential()
    model.add(layers.Conv2D(6, 5, activation= 'tanh', input_shape=(IMAGESIZE,IMAGESIZE,3)))
    model.add(layers.AveragePooling2D (2))
    model.add(layers.Activation('sigmoid'))
    model.add(layers.Conv2D(16, 5, activation= 'tanh')) 
    model.add(layers.AveragePooling2D(2))
    model.add(layers.Activation('sigmoid'))
    model.add(layers.Conv2D(120, 5, activation= 'tanh'))
    model.add(layers.Flatten())
    model.add(layers.Dense(84, activation= 'tanh')) 
    model.add(layers.Dense(30, activation='softmax'))

    model.compile(
        loss='categorical_crossentropy',
        optimizer=optimizers.Adam(learning_rate=0.00001),
        metrics=['accuracy']
    )
    return model