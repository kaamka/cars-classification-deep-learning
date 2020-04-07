from create_model import finetune_vgg16_model, finetune_resnet50_model
from create_model import finetune_inceptionv3
from create_datagenerators import create_data_generators
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
#from tensorflow.keras.applications.resnet50 import ResNet50
#from tensorflow.python.keras.applications import preprocess_input, decode_predictions
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.optimizers import Adam, RMSprop, Adadelta
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping
#import tensorflow as tf
#tf.test.gpu_device_name()


## DEFAULT PATHS
HYPERPARAMS_FILE = 'hyperparams.json'
TRAIN_DIR = 'DATASETS/Stanford_Dataset_sorted/train'
TEST_DIR = 'DATASETS/Stanford_Dataset_sorted/test'
SAVE_RESULRS_DIR = 'saved_models/'


#20190612_1048
if (os.getcwd() == '/home/kalkami/translearn'or os.getcwd() == '/home/kalkami/translearn_cpu'):
    #lhcpgpu1
    TRAIN_DIR = '/data/IntelliGate/kalkami/DATASETS/carsStanford_all/train'
    TEST_DIR = '/data/IntelliGate/kalkami/DATASETS/carsStanford_all/test'
elif (os.getcwd() == '/home/kamila/Desktop/InteliGate/CLASSIFICATION/VMMR/cars-classification-deep-learning'):
    #local
    TRAIN_DIR = '/media/kamila/System/Users/Kama/Documents/DATASETS/carsStanford_s/train'
    TEST_DIR = '/media/kamila/System/Users/Kama/Documents/DATASETS/carsStanford_s/test'

else:
    # default
    TRAIN_DIR = 'DATASETS/Stanford_Dataset_sorted/train'
    TEST_DIR ='DATASETS/Stanford_Dataset_sorted/test'


with open(HYPERPARAMS_FILE, "r") as read_file:
    data = json.load(read_file)

HYPERPARAMS = data['hyperparameters'][0]
EPOCHS = HYPERPARAMS['EPOCHS']
FC_LAYERS = HYPERPARAMS['FC_LAYERS']
EPOCHS = HYPERPARAMS['DROPOUT']
WEIGHTS = HYPERPARAMS['WEIGHTS']
TRAIN_LAYERS = HYPERPARAMS['TRAIN_LAYERS']
BATCHSIZE = HYPERPARAMS['BATCHSIZE']
DROPOUT = HYPERPARAMS['DROPOUT']
NEW_WEIGHTS = HYPERPARAMS['NEW_WEIGHTS'] 


def create_folder_with_results():
    now = datetime.datetime.now()
    training_time = now.strftime("%Y%m%d_%H%M")
    # name of dir due to today date
    TRAINING_TIME_PATH = "../saved_models/" + training_time
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

if __name__ == "__main__":
    # load_constants()
    TRAINING_TIME_PATH = create_folder_with_results()
    # base_model = ResNet50(weights=WEIGHTS, 
    #                   include_top=False, input_shape=(224,224,3))
    # save hyperparams file into result folder
    shutil.copy2(HYPERPARAMS_FILE, TRAINING_TIME_PATH)
    base_model = InceptionV3(weights=WEIGHTS, include_top=False, input_shape=(299,299,3))
    input_shape = base_model.layers[0].output_shape[1:3]
    transfer_layer = base_model.get_layer(index=-1)
    generator_train, generator_test = create_data_generators(input_shape, BATCHSIZE, 
                            TRAIN_DIR, TEST_DIR, 
                            save_augumented=None, plot_imgs = False)
    class_names = list(generator_train.class_indices.keys())
    with open(TRAINING_TIME_PATH+'/class_names.txt', 'w') as filehandle:  
            for listitem in class_names:
                filehandle.write('%s\n' % listitem)
    finetune_model = finetune_inceptionv3(base_model, transfer_layer, TRAIN_LAYERS, 
                                      dropout=DROPOUT, 
                                      fc_layers=FC_LAYERS, 
                                      num_classes= generator_train.num_classes,
                                      new_weights=NEW_WEIGHTS)
    # load weights from last best training by new weights

    #compile
    optimizer = Adam(lr=HYPERPARAMS['LEARN_RATE'])
    finetune_model.compile(optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    class_weight = compute_class_weight(class_weight='balanced',
                                    classes=np.unique(generator_train.classes),
                                    y=generator_train.classes)

    # CHECKPOINTS
    NEW_MODEL_PATH_STRUCTURE = TRAINING_TIME_PATH+'/weights.best.hdf5'
    checkpoint = ModelCheckpoint(NEW_MODEL_PATH_STRUCTURE, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]

    # generator_train.n // BATCHSIZE

    # Fit the model - train
    epochs = HYPERPARAMS['EPOCHS']
    #save model architecture
    NEW_MODEL_PATH_STRUCTURE = TRAINING_TIME_PATH+'/model.json'
    # serialize model to JSON
    model_json = finetune_model.to_json()
    with open(NEW_MODEL_PATH_STRUCTURE, "w") as json_file:
        json_file.write(model_json)
    history = finetune_model.fit_generator(generator=generator_train,
                                  epochs=epochs,
                                  steps_per_epoch=generator_train.n // BATCHSIZE,
                                  class_weight=class_weight,
                                  validation_data=generator_test,
                                  validation_steps=generator_test.n // BATCHSIZE,
                                  callbacks=callbacks_list, verbose=0)

    #finetune_model.save(NEW_MODEL_PATH)
    with open(TRAINING_TIME_PATH +'/history.txt', 'w') as f:  
        f.write(str(history.history))

    with open(TRAINING_TIME_PATH +'/model_summary.txt', 'w') as fh:
    # Pass the file handle in as a lambda function to make it callable
        finetune_model.summary(print_fn=lambda x: fh.write(x + '\n'))
        
    plot_training(history, 
        TRAINING_TIME_PATH +'/acc_vs_epochs.png', 
        TRAINING_TIME_PATH +'/loss_vs_epochs.png')
    #print(str(history.history))
