import os
import matplotlib.pyplot as plt
import PIL
from sklearn.metrics import confusion_matrix
import numpy as np
#CLASS pred
'''contains'''
#make class?

def path_join(dirname, filenames):
    return [os.path.join(dirname, filename) for filename in filenames]

def plot_images(images, cls_true, class_names, cls_pred=None, smooth=True):
    '''Function used to plot at most 9 images in a 3x3 grid, 
    and writing the true and predicted classes below each image.'''
    assert len(images) == len(cls_true)

    # Create figure with sub-plots.
    fig, axes = plt.subplots(3, 3)

    # Adjust vertical spacing.
    if cls_pred is None:
        hspace = 0.9
    else:
        hspace = 2.0
    fig.subplots_adjust(hspace=hspace, wspace=0.6)

    # Interpolation type.
    if smooth:
        interpolation = 'spline16'
    else:
        interpolation = 'nearest'

    for i, ax in enumerate(axes.flat):
        # There may be less than 9 images, ensure it doesn't crash.
        if i < len(images):
            # Plot image.
            ax.imshow(images[i],
                      interpolation=interpolation)

            # Name of the true class.
            cls_true_name = class_names[cls_true[i]]

            # Show true and predicted classes.
            if cls_pred is None:
                xlabel = "True: \n{0}".format(cls_true_name)
            else:
                # Name of the predicted class.
                cls_pred_name = class_names[cls_pred[i]]

                xlabel = "True: \n{0}\n\nPred: \n{1}".format(cls_true_name, cls_pred_name)

            # Show the classes as the label on the x-axis.
            ax.set_xlabel(xlabel)
        
        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()
    
    


def print_confusion_matrix(cls_pred, cls_test, class_names):
    '''Helper-function for printing confusion matrix'''
    # cls_pred is an array of the predicted class-number for
    # all images in the test-set.

    # Get the confusion matrix using sklearn.
    cm = confusion_matrix(y_true=cls_test,  # True class for test-set.
                          y_pred=cls_pred)  # Predicted class.

    print("Confusion matrix:")
    
    # Print the confusion matrix as text.
    print(cm)
    
    # Print the class-names for easy reference.
    for i, class_name in enumerate(class_names):
        print("({0}) {1}".format(i, class_name))
        
        
def plot_example_errors(cls_test, cls_pred, class_names, image_paths_test):
    '''Function for plotting examples of images from the test-set that have been mis-classified.'''
    # cls_pred is an array of the predicted class-number for
    # all images in the test-set.

    # Boolean array whether the predicted class is incorrect.
    incorrect = (cls_pred != cls_test)

    # Get the file-paths for images that were incorrectly classified.
    image_paths = np.array(image_paths_test)[incorrect]

    # Load the first 9 images.
    images = load_images(image_paths=image_paths[0:9])
    
    # Get the predicted classes for those images.
    cls_pred = cls_pred[incorrect]

    # Get the true classes for those images.
    cls_true = cls_test[incorrect]
    
    # Plot the 9 images we have loaded and their corresponding classes.
    # We have only loaded 9 images so there is no need to slice those again.
    plot_images(images=images, class_names=class_names,
                cls_true=cls_true[0:9],
                cls_pred=cls_pred[0:9])
    
    
def example_errors(new_model, generator_test, steps_test):
    '''The Keras data-generator for the test-set must be reset
    before processing. This is because the generator will loop
    infinitely and keep an internal index into the dataset.
    So it might start in the middle of the test-set if we do
    not reset it first. This makes it impossible to match the
    predicted classes with the input images.
    If we reset the generator, then it always starts at the
    beginning so we know exactly which input-images were used.'''
    generator_test.reset()
    
    # Predict the classes for all images in the test-set.
    y_pred = new_model.predict_generator(generator_test,
                                         steps=steps_test)

    # Convert the predicted classes from arrays to integers.
    cls_pred = np.argmax(y_pred,axis=1)

    # Plot examples of mis-classified images.
    plot_example_errors(cls_pred)
    
    # Print the confusion matrix.
    print_confusion_matrix(cls_pred)
    
    
def load_images(image_paths):
    # Load the images from disk.
    images = [plt.imread(path) for path in image_paths]

    # Convert to a numpy array and return it.
    return np.asarray(images)


def print_final_acc(model, generator_test, generator_train):
    result = model.evaluate_generator(generator_test, steps=steps_test)
    result_train = model.evaluate_generator(generator_train, steps=steps_per_epoch)
    print("Train-set classification accuracy: {0:.2%}".format(result_train[1]))
    print("Test-set classification accuracy: {0:.2%}".format(result[1]))