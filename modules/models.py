
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
    #print((3,3,np.round(channels)),(1,1,np.round(channels)))
    return

def MyV2Conv(model, in_c,out_c, stride, block_number,t):
    name = "MBlock_"+str(block_number)
    model.add(layers.Conv2D(np.round(t*in_c),(1, 1),padding="same",name = name+"_Expand"))
    model.add(layers.BatchNormalization(name=name+"_ExpandBatchNorm"))
    model.add(layers.Activation("relu",name=name+"_ExpandReLu"))
    model.add(layers.DepthwiseConv2D((3, 3),strides=(stride,stride),padding="same",name = name+"_DepConv2D"))
    model.add(layers.BatchNormalization(name=name+"_DepBatchNorm"))
    model.add(layers.Activation("relu",name=name+"_DepReLu"))
    model.add(layers.Conv2D(np.round(out_c),(1, 1),padding="same",name = name+"_CompressConv2D"))
    model.add(layers.BatchNormalization(name=name+"_CompressBatchNorm"))
    return
    
    

def minimalmodel(image_size=128, num_classes=16,alpha_model=1.0):
    mobilenet =tf.keras.applications.MobileNetV3Small(
        input_shape=(IMAGESIZE,IMAGESIZE,3),
        minimalistic=True,
        alpha=alpha_model,
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

    for i in range(1,beta+1):
        ch=(alpha*channels*(i*gain))
        CustomSepConv(model=model, channels=ch, stride=2, block_number=i)

    #avgpool
    model.add(layers.GlobalAveragePooling2D())

    model.add(layers.Dense(num_classes, activation='softmax'))

    model.compile(
          loss='categorical_crossentropy',
          optimizer=optimizers.Adam(learning_rate=0.00005),
          metrics=['accuracy']
      )
    return model

def JoJoBizzareModelScalableold(num_classes=8,input_size=96,alpha=1.0,beta=3,channels=32,g=1.0):
    """
    Return the estimated memory usage of a given Keras model in bytes.
    This includes the model weights and layers, but excludes the dataset.
    The model shapes are multipled by the batch size, but the weights are not.
    Args:
        num_classes: number of output classes
        input_size : input feature map size (input_size,input_size,3)
        alpha: width multiplier
        beta: depth multiplier
        g: gain
        channels: number of starting channels
    Returns:
        A keras model
    """
    strname= "jojo_s_n"+str(num_classes)+"_r"+str(input_size)+"_a"+str(alpha)+"_b"+str(beta)+"_g"+str(g)
    model = Sequential(name=strname)

    #input layer
    model.add(layers.InputLayer(input_shape=(input_size,input_size,3), name="input_layer"))
    # first layer is a standard conv2D
    model.add(layers.Conv2D(np.round(alpha*channels), (3, 3),strides=(2,2),padding="same",name="Conv0"))
    for i in range(1,beta+1):
        # The second part of the block reduces the size of the feature map
        if i % 2 == 0:
            s = 2
        else:
            s = 1 

        #first block
        name = "Block"+str(i)
        model.add(layers.SeparableConv2D(np.round(alpha*channels*(i**g)), (3, 3),strides=(s,s),padding="same",name = name+"_SepConv2D"+str(1)))
        model.add(layers.BatchNormalization(name=name+"_BatchNorm"+str(1)))
        model.add(layers.Activation("relu",name=name+"_ReLu"+str(1)))

        #second block
        model.add(layers.SeparableConv2D(np.round(alpha*channels*(i**g)),(3,3),strides=(1,1),padding="same",name=name+"_SepConv2D"+str(2)))
        model.add(layers.BatchNormalization(name=name+"_BatchNorm"+str(2)))
        model.add(layers.Activation("relu",name=name+"_ReLu"+str(2)))

    #avgpool
    model.add(layers.GlobalAveragePooling2D())

    model.add(layers.Dense(24, activation='relu'))
    model.add(layers.Dropout(0.15))

    model.add(layers.Dense(num_classes, activation='softmax'))

    model.compile(
          loss='categorical_crossentropy',
          optimizer=optimizers.Adam(learning_rate=0.00001),
          metrics=['accuracy']
      )
    return model

def JoJoBizzareModel(num_classes=8,input_size=128,alpha=1.0):

    strname= "jojo_n"+str(num_classes)+"_r"+str(input_size)+"_a"+str(alpha)
    model = Sequential(name=strname)

    #input layer
    model.add(layers.InputLayer(input_shape=(input_size,input_size,3), name="input_layer"))

    # first layer is a standard conv2D
    model.add(layers.Conv2D(np.round(alpha*32), (3, 3)))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))

    #dp1
    model.add(layers.SeparableConv2D(np.round(alpha*32),(3,3),strides=(1,1),name="dws1"))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))

    #dp2
    model.add(layers.Conv2D(np.round(alpha*32),(3,3),strides=(1,1),name="dws2"))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))

    #dp3
    model.add(layers.SeparableConv2D(np.round(alpha*64),(3,3),strides=(2,2),name="dws3"))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))

    #dp4
    model.add(layers.Conv2D(np.round(alpha*64),(3,3),strides=(1,1),name="dws4"))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))

    #dp5
    model.add(layers.SeparableConv2D(np.round(alpha*128),(3,3),strides=(2,2),name="dws5"))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))

    #dp6
    model.add(layers.Conv2D(np.round(alpha*128),(3,3),strides=(1,1),name="dws6"))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))

    #avgpool
    model.add(layers.GlobalAveragePooling2D())

    model.add(layers.Dense(24, activation='relu'))
    model.add(layers.Dropout(0.15))

    model.add(layers.Dense(num_classes, activation='softmax'))

    model.compile(
          loss='categorical_crossentropy',
          optimizer=optimizers.Adam(learning_rate=0.00001),
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
