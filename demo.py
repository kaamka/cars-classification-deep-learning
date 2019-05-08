from keras.preprocessing import image
import os, random
import os, random


def load_image(img_path, show=True):

    img = image.load_img(img_path, target_size=input_shape)
    img_tensor = image.img_to_array(img)                    # (height, width, channels)
    img_tensor = np.expand_dims(img_tensor, axis=0)         # (1, height, width, channels), add a dimension because the model expects this shape: (batch_size, height, width, channels)
    img_tensor /= 255.                                      # imshow expects values in the range [0, 1]

    if show:
        plt.imshow(img_tensor[0])                           
        plt.axis('off')
        plt.show()

    return img_tensor

def decode_predictions(preds, top=3):
    results = []
    for pred in preds:
        top_indices = pred.argsort()[-top:][::-1]
        result = [(class_names[i], pred[i]) for i in top_indices]
        result.sort(key=lambda x: x[1], reverse=True)
        results.append(result)
    return results

def predict(img_path, model):
    img_array = load_image(img_path)
    pred = model.predict(img_array)
    print(decode_predictions(pred))
    # Decode the output of the VGG16 model.
    #pred_decoded = decode_predictions(pred)[0]

    ## Print the predictions.
    #for code, name, score in pred_decoded:
        #print("{0:>6.2%} : {1}".format(score, name))

        
car_class = 'BMW 3 Series Sedan 2012'
test_dir = '../DATASETS/carsStanford_all/test/' + car_class
test_img = test_dir + '/' + random.choice(os.listdir(test_dir))

        
#test_img = '../DATASETS/test_imgs/f.png'
model_path = NEW_MODEL_PATH
#model_path = 'saved_models/20190502_1643/vgg16_.h5'
model = load_model(model_path)
#load_image(test_img)
predict(test_img, model)