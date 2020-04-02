from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import model_from_json
import matplotlib.pyplot as plt
import numpy as np
import os, random, sys
from create_datagenerators import create_data_generators
import json
from sklearn.metrics import confusion_matrix
from data_preprocessing import resize_white, resize_black
from tensorflow.keras.utils import plot_model
from collections import Counter


#20190612_1048
if (os.getcwd() == '/home/kalkami/translearn'or os.getcwd() == '/home/kalkami/translearn_cpu'):
    #lhcpgpu1
    TRAIN_DIR = '/data/IntelliGate/kalkami/DATASETS/carsStanford_all/train'
    TEST_DIR = '/data/IntelliGate/kalkami/DATASETS/carsStanford_all/test'
    TRAIN_DIR_TST = TRAIN_DIR
    TEST_DIR_TST = TEST_DIR
elif (os.getcwd() == '/home/kamila/Desktop/InteliGate/CLASSIFICATION/VMMR/cars-classification-deep-learning'):
    #local
    TRAIN_DIR = '/media/kamila/System/Users/Kama/Documents/DATASETS/carsStanford_s/train'
    TEST_DIR = '/media/kamila/System/Users/Kama/Documents/DATASETS/carsStanford_s/test'
    TRAIN_DIR_TST = '/media/kamila/System/Users/Kama/Documents/DATASETS/carsStanford_all/train'
    TEST_DIR_TST = '/media/kamila/System/Users/Kama/Documents/DATASETS/carsStanford_all/test'
    TEST_DIR_TST = '/media/kamila/System/Users/Kama/Documents/DATASETS/CARS_GOOGLE_IMG/downloads'
else:
    # default
    SAVE_RESULRS_DIR = 'saved_models/'
    RESULTS_FOLDER = SAVE_RESULRS_DIR + '/20190701_1148'
    TRAIN_DIR = 'DATASETS/Stanford_Dataset_sorted/train'
    TEST_DIR ='DATASETS/Stanford_Dataset_sorted/test'


def load_image(img_path, input_shape, resize = False):
    if resize:
        img, pth = resize_black(input_shape[1], img_path, print_oldsize=False)
    else:
        img = image.load_img(img_path, target_size=input_shape)
    img_tensor = image.img_to_array(img)                    # (height, width, channels)
    img_tensor = np.expand_dims(img_tensor, axis=0)         # (1, height, width, channels), add a dimension because the model expects this shape: (batch_size, height, width, channels)
    img_tensor /= 255.                                      # imshow expects values in the range [0, 1]
    return img_tensor

def decode_predictions(preds, class_names, top=5):
    results = []
    for pred in preds:
        top_indices = pred.argsort()[-top:][::-1]
        result = [(class_names[i], pred[i]) for i in top_indices]
        result.sort(key=lambda x: x[1], reverse=True)
        results.append(result)
    return results

def predict(img_path, model, input_shape, class_names, correct_class):
    img_array = load_image(img_path, input_shape)
    preds = model.predict(img_array)
    predictions = decode_predictions(preds, class_names)
    top1_pred = predictions[0][0]
    print(predictions)
    img_org = image.load_img(img_path)
    fig, axs = plt.subplots(1,2)
    axs[0].set_title(correct_class)
    axs[0].imshow(img_org)
    #axs[0].axis('off')

    axs[1].set_title(str(top1_pred))
    axs[1].imshow(img_array[0])
    plt.show()
    # Decode the output of the VGG16 model.
    #pred_decoded = decode_predictions(pred)[0]
    ## Print the predictions.
    #for code, name, score in pred_decoded:
        #print("{0:>6.2%} : {1}".format(score, name))

def load_model(results_folder, show_accuracy=False):
    # load json and create model
    json_file = open(results_folder + '/model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(results_folder + "/weights.best.hdf5")
    input_shape = loaded_model.layers[0].output_shape[1:3]
    print("Loaded model from disk")
    return loaded_model, input_shape


def perform_pred(car_class, results_folder=RESULTS_FOLDER, test_dir=TEST_DIR_TST, img_pth=None):
    HYPERPARAMS_FILE =  results_folder+ '/hyperparams.json'

    with open(HYPERPARAMS_FILE, "r") as read_file:
        data = json.load(read_file)

    HYPERPARAMS = data['hyperparameters'][0]
    BATCHSIZE = HYPERPARAMS['BATCHSIZE'] 

    if img_pth is None:   
        # randomly select an image from defined class     
        test_dir_full = test_dir + '/' + car_class
        test_img = test_dir_full + '/' + random.choice(os.listdir(test_dir_full))
    else:
        test_img = img_pth

    loaded_model, input_shape=load_model(results_folder)
    print (input_shape[1])
    if os.path.exists(results_folder+'/class_names.txt'):
        class_names = []
        # open file and read the content in a list
        with open(results_folder+'/class_names.txt', 'r') as filehandle:  
            for line in filehandle:
                current_line = line[:-1]
                class_names.append(current_line)
    else:
        generator_train, generator_test = create_data_generators(input_shape, BATCHSIZE, 
                                                                TRAIN_DIR, TEST_DIR, 
                                                                save_augumented=None, 
                                                                plot_imgs = False) 
        class_names = list(generator_train.class_indices.keys())
        print(class_names)
        with open(results_folder+'/class_names.txt', 'w') as filehandle:  
            for listitem in class_names:
                filehandle.write('%s\n' % listitem)
            
    predict(test_img, loaded_model, input_shape, class_names, car_class)


if __name__ == "__main__":
    #img_test = '../test_imgs/f.png'
    # Audi A5 Coupe 2012
    print(sys.argv)
    if len(sys.argv) == 1:
        print('Too few arguments.')
    elif len(sys.argv) == 2:
        perform_pred(sys.argv[1])
    elif len(sys.argv) == 3:
        if str(sys.argv[2]).endswith('.jpg') or str(sys.argv[2]).endswith('.png'):
            perform_pred(sys.argv[1], img_pth=sys.argv[2])
        else:
            perform_pred(sys.argv[1], results_folder=sys.argv[2])
    elif len(sys.argv) == 4:
        if str(sys.argv[3]).endswith('.jpg') or str(sys.argv[3]).endswith('.png'):
            perform_pred(sys.argv[1], results_folder=sys.argv[2], 
                            img_pth=sys.argv[3])
        else:
            perform_pred(sys.argv[1], results_folder=sys.argv[2], 
                                    test_dir=sys.argv[3])
    else:
        print('Too much arguments.')


