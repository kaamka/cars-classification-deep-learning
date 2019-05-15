from create_model import finetune_vgg16_model
import json, codecs
import datetime
import os
import matplotlib as plt
from tensorflow.python.keras.models import Model, Sequential, load_model
from tensorflow.python.keras.layers import Dense, Flatten, Dropout,GlobalAveragePooling2D,Input
from tensorflow.python.keras.applications import VGG16
#from tensorflow.keras.applications.resnet50 import ResNet50
#from tensorflow.python.keras.applications import preprocess_input, decode_predictions
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.optimizers import Adam, RMSprop, Adadelta

EPOCHS = ''
FC_LAYERS = ''
DROPOUT = ''
HYPERPARAMS_FILE = 'hyperprams.json'
TRAIN_DIR = '../DATASETS/carsStanford_all/train'
TEST_DIR = '../DATASETS/carsStanford_all/test'

with open(HYPERPARAMS_FILE, "r") as read_file:
    data = json.load(read_file)

HYPERPARAMS = data['hyperparameters'][0]
EPOCHS = HYPERPARAMS['EPOCHS']
FC_LAYERS = HYPERPARAMS['FC_LAYERS']
EPOCHS = HYPERPARAMS['DROPOUT']
WEIGHTS = HYPERPARAMS['WEIGHTS']

def create_folder_with_results():
    now = datetime.datetime.now()
    training_time = now.strftime("%Y%m%d_%H%M")
    # name of dir due to today date
    TRAINING_TIME_PATH = "saved_models/" + training_time
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

def plot_training(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(len(acc))

    plt.plot(epochs, acc, 'b')
    plt.plot(epochs, val_acc, 'r')
    plt.title('Training and validation accuracy')

    plt.figure()
    plt.plot(epochs, loss, 'b')
    plt.plot(epochs, val_loss, 'r')
    plt.title('Training and validation loss')
    plt.show()
    plt.savefig('acc_vs_epochs.png')


if __name__ == "__main__":
    TRAINING_TIME_PATH = create_folder_with_results()

    base_model = VGG16(weights=WEIGHTS, 
                      include_top=False, input_shape=(224,224,3))
    input_shape = base_model.layers[0].output_shape[1:3]
    finetune_model = finetune_vgg16_model(base_model, transfer_layer, HYPERPARAMS['TRAIN_LAYERS'], 
                                      dropout=DROPOUT, 
                                      fc_layers=FC_LAYERS, 
                                      num_classes= num_classes)
    
    #compile
    optimizer = Adam(lr=HYPERPARAMS['LEARN_RATE'])
    finetune_model.compile(optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    #train
    history = finetune_model.fit_generator(generator=generator_train,
                                  epochs=epochs,
                                  steps_per_epoch=steps_per_epoch,
                                  class_weight=class_weight,
                                  validation_data=generator_test,
                                  validation_steps=steps_test)
    #save
    NEW_MODEL_PATH = TRAINING_TIME_PATH+'/newmodel.h5'
    finetune_model.save(NEW_MODEL_PATH)

    with open(TRAINING_TIME_PATH +'/history.txt', 'w') as f:  
        f.write(str(history.history))

    with open(TRAINING_TIME_PATH +'/model_summary', 'w') as f:
        f.write(str(finetune_model.summary()))
        
    plot_training(history)
    print(str(history.history))
