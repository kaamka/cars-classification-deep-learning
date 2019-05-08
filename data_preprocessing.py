from tensorflow.python.keras.preprocessing.image import ImageDataGenerator

train_dir = '../DATASETS/carsStanford_all/train'
test_dir = '../DATASETS/carsStanford_all/test'
HYPERPARAMS_FILE = 'hyperparams.json'

# read the hyperparams file and load into dictionary
with open(HYPERPARAMS_FILE, "r") as read_file:
    data = json.load(read_file)

HYPERPARAMS = data['hyperparameters'][0]

datagen_train = ImageDataGenerator(
      rescale=1./255,
      rotation_range=180,
      width_shift_range=0.1,
      height_shift_range=0.1,
      shear_range=0.1,
      zoom_range=[0.9, 1.5],
      horizontal_flip=True,
      vertical_flip=True,
      fill_mode='nearest')



datagen_test = ImageDataGenerator(rescale=1./255)

batch_size = HYPERPARAMS['BATCHSIZE']

if True:
    save_to_dir = None
else:
    save_to_dir='augmented_images/'
    
generator_train = datagen_train.flow_from_directory(directory=train_dir,
                                                    target_size=input_shape,
                                                    batch_size=batch_size,
                                                    shuffle=True,
                                                    save_to_dir=save_to_dir)    
steps_per_epoch = generator_train.n / batch_size
                    
                    
generator_test = datagen_test.flow_from_directory(directory=test_dir,
                                                  target_size=input_shape,
                                                  batch_size=batch_size,
                                                  shuffle=False)

steps_test = generator_test.n / batch_size
#steps_test

image_paths_train = path_join(train_dir, generator_train.filenames)
image_paths_test = path_join(test_dir, generator_test.filenames)

cls_train = generator_train.classes
cls_test = generator_test.classes
#print(cls_train)
#print(cls_test)

class_names = list(generator_train.class_indices.keys())
print(class_names)

num_classes = generator_train.num_classes
#num_classes


from sklearn.utils.class_weight import compute_class_weight

class_weight = compute_class_weight(class_weight='balanced',
                                    classes=np.unique(cls_train),
                                    y=cls_train)
#generator.train, generator.test, cls_train, cls_test