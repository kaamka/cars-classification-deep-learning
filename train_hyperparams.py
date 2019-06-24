# -*- coding: utf-8 -*-
from create_model import finetune_vgg16_model, finetune_resnet50_model
from create_model import finetune_inceptionv3
from data_preprocessing import create_data_generators
import json, codecs
import numpy as np
import datetime
import shutil, os, pickle
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.python.keras.models import Model, Sequential, load_model
from tensorflow.python.keras.layers import Dense, Flatten, Dropout,GlobalAveragePooling2D,Input
from tensorflow.python.keras.applications import VGG16
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.optimizers import Adam, RMSprop, Adadelta
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import PIL
import tensorflow as tf
import numpy as np
import os
import json
from tensorflow.python.keras.models import Model, Sequential, load_model
from tensorflow.python.keras.layers import Dense, Flatten, Dropout,GlobalAveragePooling2D,Input
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.resnet50 import ResNet50
#from tensorflow.python.keras.applications import preprocess_input, decode_predictions
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.optimizers import Adam, RMSprop, Adadelta

#import tensorflow as tf
#tf.test.gpu_device_name()
from __future__ import print_function

from hyperopt import Trials, STATUS_OK, tpe, rand
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential
from keras.utils import np_utils
from sklearn.metrics import accuracy_score
from hyperas import optim
from hyperas.distributions import choice, uniform
from keras import optimizers



# def load_constants()
#HYPERPARAMS_FILE = 'hyperparams.json'

if (os.getcwd() == '/home/kalkami/translearn' or os.getcwd() == '/home/kalkami/translearn_cpu'):
    #lhcpgpu1
    TRAIN_DIR = '/data/IntelliGate/kalkami/DATASETS/carsStanford_all/train'
    TEST_DIR = '/data/IntelliGate/kalkami/DATASETS/carsStanford_all/test'
else:
    #local
    TRAIN_DIR = '/media/kamila/System/Users/Kama/Documents/DATASETS/carsStanford_s/train'
    TEST_DIR = '/media/kamila/System/Users/Kama/Documents/DATASETS/carsStanford_s/test'

def create_folder_with_results():
    now = datetime.datetime.now()
    training_time = now.strftime("%Y%m%d_%H%M")
    # name of dir due to today date
    TRAINING_TIME_PATH = "../saved_models/" + 'serch_hypers_'+ training_time
    access_rights = 0o755
    try:  
        os.makedirs(TRAINING_TIME_PATH, access_rights)
    except OSError:  
        print ("Creation of the directory %s failed" % TRAINING_TIME_PATH)
    else:  
        print ("Successfully created the directory %s" % TRAINING_TIME_PATH)
        return TRAINING_TIME_PATH

def saveHist(path,history):
    new_hist = {}
    for key in list(history.history.keys()):
        if type(history.history[key]) == np.ndarray:
            new_hist[key] == history.history[key].tolist()
        elif type(history.history[key]) == list:
           if type(history.history[key][0]) == np.float64:
               new_hist[key] = list(map(float, history.history[key]))

    print(new_hist)
    with codecs.open(path, 'w', encoding='utf-8') as f:
        json.dump(new_hist, f, separators=(',', ':'), sort_keys=True, indent=4) 


def loadHist(path):
    with codecs.open(path, 'r', encoding='utf-8') as f:
        n = json.loads(f.read())
    return n

def plot_training(history, path_acc, path_loss):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(len(acc))

    plt.figure()
    plt.plot(epochs, acc, 'b', label="Training set accuracy")
    plt.plot(epochs, val_acc, 'r', label="Test set accuracy")
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.savefig(path_acc)

    plt.figure()
    plt.plot(epochs, loss, 'b', label="Training set loss")
    plt.plot(epochs, val_loss, 'r', label="Test set loss")
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()
    plt.savefig(path_loss)

    


def plot_training_history(history, path):
    # Get the classification accuracy and loss-value
    # for the training-set.
    acc = history.history['categorical_accuracy']
    loss = history.history['loss']

    # Get it for the validation-set (we only use the test-set).
    val_acc = history.history['val_categorical_accuracy']
    val_loss = history.history['val_loss']

    # Plot the accuracy and loss-values for the training-set.
    plt.plot(acc, linestyle='-', color='b', label='Training Acc.')
    plt.plot(loss, 'o', color='b', label='Training Loss')
    
    # Plot it for the test-set.
    plt.plot(val_acc, linestyle='--', color='r', label='Test Acc.')
    plt.plot(val_loss, 'o', color='r', label='Test Loss')

    # Plot title and legend.
    plt.title('Training and Test Accuracy')
    plt.legend()

    # Save the image
    plt.savefig(path)

    # Ensure the plot shows correctly.
    plt.show()



'''
cls_train = generator_train.classes
cls_test = generator_test.classes
class_names = list(generator_train.class_indices.keys())
num_classes = generator_train.num_classes
class_weight = compute_class_weight(class_weight='balanced',
                                    classes=np.unique(cls_train),
                                    y=cls_train)

steps_per_epoch = generator_train.n / batch_size 
steps_test = generator_test.n / batch_size
'''

def data(input_shape, batch_size, train_dir, test_dir):
    generator_train, generator_test = create_data_generators(input_shape, batch_size, 
                            train_dir, test_dir)
    return generator_train, generator_test


def create_model(train_generator, test_generator):
    base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(299,299,3))
    input_shape = base_model.layers[0].output_shape[1:3]
    transfer_layer = base_model.get_layer(index=-1) 
    new_weights = ""
    for layer in base_model.layers[:-60]:
        layer.trainable = False

    x = transfer_layer.output
    x = GlobalAveragePooling2D(name='avg_pool')(x)
    x = Dropout(dropout)(x)
    predictions = Dense(num_classes, activation='softmax')(x)  
    finetune_model = Model(inputs=base_model.input, outputs=predictions)
    if new_weights != "":
        finetune_model.load_weights(new_weights)
        
    # model = models.Sequential()
    # model.add(layers.Dense({{choice([np.power(2, 5), np.power(2, 6), np.power(2, 7)])}}, input_shape=(len(data.columns),)))
    # model.add(LeakyReLU(alpha={{uniform(0.5, 1)}}))
    # model.add(Dropout({{uniform(0.5, 1)}}))
    # model.add(layers.Dense({{choice([np.power(2, 3), np.power(2, 4), np.power(2, 5)])}}))
    # model.add(LeakyReLU(alpha={{uniform(0.5, 1)}}))
    # model.add(Dropout({{uniform(0.5, 1)}}))
    # model.add(layers.Dense(1, activation='sigmoid'))
    reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                  patience=5, min_lr=0.001)
    finetune_model.compile(optimizer={{choice(['rmsprop', 'adam', 'sgd'])}},
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    class_weight = compute_class_weight(class_weight='balanced',
                                    classes=np.unique(generator_train.classes),
                                    y=generator_train.classes)

    history = finetune_model.fit_generator(generator=generator_train,
                                  epochs={{choice([25, 50, 75, 100])}},
                                  steps_per_epoch=generator_train.n // BATCHSIZE,
                                  batch_size={{choice([16, 32, 64])}},
                                  class_weight=class_weight,
                                  validation_data=generator_test,
                                  validation_steps=generator_test.n // BATCHSIZE,
                                  callbacks=[reduce_lr])

    score, acc = finetune_model.evaluate(X_val, y_val, verbose=0)
    print('Test accuracy:', acc)
    return {'loss': -acc, 'status': STATUS_OK, 'model': finetune_model}


# if __name__ == '__main__':
    # TRAINING_TIME_PATH = create_folder_with_results()
    # generator_train, generator_test = create_data_generators(input_shape, BATCHSIZE, 
    #                         TRAIN_DIR, TEST_DIR, 
    #                         save_augumented=None, plot_imgs = False)
    # class_names = list(generator_train.class_indices.keys())
    # with open(TRAINING_TIME_PATH+'/class_names.txt', 'w') as filehandle:  
    #         for listitem in class_names:
    #             filehandle.write('%s\n' % listitem)

#     best_run, best_model = optim.minimize(model=create_model,
#                                           data=data,
#                                           algo=tpe.suggest,
#                                           max_evals=15,
#                                           trials=Trials())
#     X_train, X_val, X_test, y_train, y_val, y_test = data()
#     print("Evalutation of best performing model:")
#     print(best_model.evaluate(X_test, y_test))
#     print("Best performing model chosen hyper-parameters:")
#     print(best_run)

#     best_model.save('breast_cancer_model.h5')