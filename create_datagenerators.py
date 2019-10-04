from analyse_results import plot_images, load_images
from analyse_results import path_join
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import json


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

def create_data_generators(input_shape, batch_size, 
                            train_dir, test_dir, 
                            save_augumented=False, plot_imgs = False):
    datagen_train = ImageDataGenerator(
      rescale=1./255,
      rotation_range=20.,
      width_shift_range=0.1,
      height_shift_range=0.1,
      zoom_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest')

    datagen_test = ImageDataGenerator(rescale=1./255)

    if save_augumented:
        save_to_dir='augmented_images/'
    else:
        save_to_dir = None

    generator_train = datagen_train.flow_from_directory(directory=train_dir,
                                                    target_size=input_shape,
                                                    batch_size=batch_size,
                                                    shuffle=True,
                                                    save_to_dir=save_to_dir)   
                                        
    generator_test = datagen_test.flow_from_directory(directory=test_dir,
                                                  target_size=input_shape,
                                                  batch_size=batch_size,
                                                  shuffle=False)
    if plot_imgs:
        plot_some_images(train_dir, generator_train)
    return generator_train, generator_test


def plot_some_images(train_dir, generator_train):

    image_paths_train = path_join(train_dir, generator_train.filenames)
    #image_paths_test = path_join(test_dir, generator_test.filenames)

    # Load the first images from the train-set.
    images = load_images(image_paths=image_paths_train[555:564])

    # Get the true classes for those images.
    cls_train = generator_train.classes
    cls_true = cls_train[555:564]

    # Plot the images and labels using our helper-function above.
    class_names = list(generator_train.class_indices.keys())
    plot_images(images=images, cls_true=cls_true, class_names = class_names, smooth=True)


if __name__ == "__main__":
    # data
    #local
    TRAIN_DIR = '/media/kamila/System/Users/Kama/Documents/DATASETS/carsStanford_all/train'
    TEST_DIR = '/media/kamila/System/Users/Kama/Documents/DATASETS/carsStanford_all/test'
    create_data_generators((224,224,3), 20, 
                            TRAIN_DIR, TEST_DIR, 
                            save_augumented=None, plot_imgs = True)
    