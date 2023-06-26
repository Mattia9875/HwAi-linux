
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers, optimizers
from keras.applications import MobileNetV2
from keras.models import Sequential
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
IMAGESIZE = 128

def CustomSepConv(model, channels, stride, block_number):
    name = "CBlock_"+str(block_number)
    model.add(layers.SeparableConv2D(np.round(channels), (3, 3),strides=(stride,stride),padding="same",name = name+"_SepConv2D"))
    model.add(layers.BatchNormalization(name=name+"_BatchNorm"))
    model.add(layers.ReLU(max_value=6.0,name=name+"_ReLu"))
    model.add(layers.Dropout(0.01))
    return

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

def JoJoBizzareModelScalable(num_classes=8,input_size=(96,96,3),alpha=1.0,beta=3, strides=2,channels=32,gain=1.0):
    """
    Args:
        num_classes: number of output classes
        input_size : input feature map size
        alpha: width multiplier
        beta: depth multiplier
        g: gain
        channels: number of starting channels
    Returns:
        A keras model
    """
    strname= "jojo_n"+str(num_classes)+"_r"+str(input_size[0])+"x"+str(input_size[1])+"_a"+str(alpha)+"_b"+str(beta)+"_g"+str(gain)+"_strides"+str(strides)
    model = Sequential(name=strname)

    #input layer
    model.add(layers.InputLayer(input_shape=input_size, name="input_layer"))
    # first layer is a standard conv2D
    model.add(layers.Conv2D(np.round(alpha*channels), (3, 3),strides=(strides,strides),padding="same",name="Conv0"))

    # Add beta layers
    for i in range(1,beta+1):
        ch=(alpha*channels*(i*gain))
        CustomSepConv(model=model, channels=ch, stride=2, block_number=i)

    # Reduce last layer dimensions
    model.add(layers.GlobalAveragePooling2D())

    # Final Classifier
    model.add(layers.Dense(num_classes, activation='softmax'))

    model.compile(
          loss='categorical_crossentropy',
          optimizer=optimizers.Adam(learning_rate=0.00005),
          metrics=['accuracy']
      )
    return model


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
