from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import model_from_json
import matplotlib.pyplot as plt
import numpy as np
import os, random
import os, random
from data_preprocessing import create_data_generators
import json

# def load_constants()
RESULTS_FOLDER = "../saved_models/20190612_1048"
HYPERPARAMS_FILE =  RESULTS_FOLDER+ '/hyperparams.json'

if (os.getcwd() == '/home/kalkami/translearn'):
    #lhcpgpu1
    TRAIN_DIR = '/data/IntelliGate/kalkami/DATASETS/carsStanford_all/train'
    TEST_DIR = '/data/IntelliGate/kalkami/DATASETS/carsStanford_all/test'
    TRAIN_DIR_TST = TRAIN_DIR
    TEST_DIR_TST = TEST_DIR
else:
    #local
    TRAIN_DIR = '/media/kamila/System/Users/Kama/Documents/DATASETS/carsStanford_s/train'
    TEST_DIR = '/media/kamila/System/Users/Kama/Documents/DATASETS/carsStanford_s/test'
    TRAIN_DIR_TST = '/media/kamila/System/Users/Kama/Documents/DATASETS/carsStanford_all/train'
    TEST_DIR_TST = '/media/kamila/System/Users/Kama/Documents/DATASETS/carsStanford_all/test'

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
INPUT_SHAPE = (299,299,3) # try to take it from model
#input_shape = model.layers[0].output_shape[1:3]

def load_image(img_path, input_shape, show=True):

    img = image.load_img(img_path, target_size=input_shape)
    img_tensor = image.img_to_array(img)                    # (height, width, channels)
    img_tensor = np.expand_dims(img_tensor, axis=0)         # (1, height, width, channels), add a dimension because the model expects this shape: (batch_size, height, width, channels)
    img_tensor /= 255.                                      # imshow expects values in the range [0, 1]

    if show:
        plt.imshow(img_tensor[0])                           
        plt.axis('off')
        plt.show()

    return img_tensor

def decode_predictions(preds, class_names, top=3):
    results = []
    for pred in preds:
        top_indices = pred.argsort()[-top:][::-1]
        result = [(class_names[i], pred[i]) for i in top_indices]
        result.sort(key=lambda x: x[1], reverse=True)
        results.append(result)
    return results

def predict(img_path, model, input_shape, class_names):
    img_array = load_image(img_path, input_shape)
    preds = model.predict(img_array)
    print(decode_predictions(preds, class_names))
    # Decode the output of the VGG16 model.
    #pred_decoded = decode_predictions(pred)[0]
    ## Print the predictions.
    #for code, name, score in pred_decoded:
        #print("{0:>6.2%} : {1}".format(score, name))

io = True
#if __name__ == "__main__":
if io:
    generator_train, generator_test = create_data_generators(INPUT_SHAPE, BATCHSIZE, 
                                                                TRAIN_DIR, TEST_DIR, 
                                                                save_augumented=None, 
                                                                plot_imgs = False)
    # randomly select an image from defined class
    car_class = 'Audi A5 Coupe 2012'                                                           
    class_names = list(generator_train.class_indices.keys())        
    test_dir = TEST_DIR_TST + '/' + car_class
    test_img = test_dir + '/' + random.choice(os.listdir(test_dir))

    # load json and create model
    json_file = open(RESULTS_FOLDER + '/model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(RESULTS_FOLDER + "/weights.best.hdf5")
    input_shape = loaded_model.layers[0].output_shape[1:3]
    print("Loaded model from disk")
    predict(test_img, loaded_model, input_shape, class_names)
    
    # evaluate loaded model on test data
    # loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    # score = loaded_model.evaluate(X, Y, verbose=0)
    # print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))       
    #test_img = '../DATASETS/test_imgs/f.png'

    # load model weigths & structure simultanously
    # model_path = NEW_MODEL_PATH
    # model_path = 'saved_models/20190502_1643/vgg16_.h5'
    # model = load_model(model_path)
    # load_image(test_img)
