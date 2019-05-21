# imports
import matplotlib.pyplot as plt
import PIL
import tensorflow as tf
import numpy as np
import os
import json
from tensorflow.python.keras.models import Model, Sequential, load_model
from tensorflow.python.keras.layers import Dense, Flatten, Dropout,GlobalAveragePooling2D,Input
from tensorflow.python.keras.applications import VGG16
#from tensorflow.keras.applications.resnet50 import ResNet50
#from tensorflow.python.keras.applications import preprocess_input, decode_predictions
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.optimizers import Adam, RMSprop, Adadelta


def finetune_vgg16_model(base_model, transfer_layer, x_trainable, dropout, fc_layers, num_classes):
    for layer in base_model.layers[:-x_trainable]:
        layer.trainable = False

    x = transfer_layer.output
    #x = Dense(num_classes, activation='softmax')
    x = Flatten()(x)
    for fc in fc_layers:
        # New FC layer, random init
        x = Dense(fc, activation='relu')(x) 
        x = Dropout(dropout)(x)
    # New softmax layer
    predictions = Dense(num_classes, activation='softmax')(x)  
    finetune_model = Model(inputs=base_model.input, outputs=predictions)
    return finetune_model
# return model


if __name__ == "__main__":
    base_model = VGG16(weights='imagenet', 
                      include_top=False, input_shape=(224,224,3))
    input_shape = base_model.layers[0].output_shape[1:3]
    transfer_layer = base_model.get_layer(index=-1)
    new_model = finetune_vgg16_model(base_model, transfer_layer, 5, 0.5, [1024, 1024], 196)
    optimizer = Adam(lr=0.000001)
    new_model.compile(optimizer, loss='categorical_crossentropy', metrics=['accuracy'])