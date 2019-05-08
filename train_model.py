from data_preprocessing import *
FC_LAYERS = [1024, 1024]
dropout = HYPERPARAMS['DROPOUT']
with open(HYPERPARAMS_FILE, "r") as read_file:
    data = json.load(read_file)

HYPERPARAMS = data['hyperparameters'][0]
epochs = HYPERPARAMS['EPOCHS']
steps_per_epoch = generator_train.n / HYPERPARAMS['BATCHSIZE']
now = datetime.datetime.now()
training_time = now.strftime("%Y%m%d_%H%M")

# name of dir due to today date
TRAINING_TIME_PATH = "saved_models/" + training_time


# define the access rights
access_rights = 0o755

try:  
    os.makedirs(TRAINING_TIME_PATH, access_rights)
except OSError:  
    print ("Creation of the directory %s failed" % TRAINING_TIME_PATH)
else:  
    print ("Successfully created the directory %s" % TRAINING_TIME_PATH)
# obczaj dane
# Load the first images from the train-set.
images = load_images(image_paths=image_paths_train[555:564])

# Get the true classes for those images.
cls_true = cls_train[555:564]

# Plot the images and labels using our helper-function above.
plot_images(images=images, cls_true=cls_true, smooth=True)

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
    #model
    finetune_model = build_finetune_model(base_model, transfer_layer, HYPERPARAMS['TRAIN_LAYERS'], 
                                      dropout=dropout, 
                                      fc_layers=FC_LAYERS, 
                                      num_classes= num_classes)
    #datagenerators
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
