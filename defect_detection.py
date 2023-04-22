import pandas as pd
import numpy as np
import glob
import imageio
import matplotlib.pyplot as plt
import sklearn
def load_file(file_path, label):

    # declare the folder name
    folder_name = file_path.split("/")[-1]
    # declare output list
    out_list = []
    # load every file that .png format
    for image_path in glob.glob(file_path + "/*.png"):
        # read image file
        image = imageio.imread(image_path)
        # declare temporary dict dtype
        temp = {}
        # set the file name
        temp["name"] = image_path.split("/")[-1]
        # set the file label, 0 for non defect. 1 for defect
        temp["label"] = label

        # There are somes images are tensor dtype
        # Thus I fix by selecting only a tensor index zero
        try:   
            temp["data"] = image[:,:,0].astype("int") 
        except:
            # normal case
            temp["data"] = image.astype("int")
        # append temp into output list
        out_list.append(temp)
    # print process status by checking size of output list
    if len(out_list) == 0:
        print("loading files from folder: {} is failed".format(folder_name))
    else:
        print("loading file from folder: {} is successful".format(folder_name))
    # convert list into numpy array dtype
    return np.array(out_list)
defect_images = load_file(file_path=defect_images_path, label=1)
non_defect_images1 = load_file(file_path=non_defect_images_path1, label=0)
non_defect_images2 = load_file(file_path=non_defect_images_path2, label=0)
non_defect_images3 = load_file(file_path=non_defect_images_path3, label=0)
non_defect_images4 = load_file(file_path=non_defect_images_path4, label=0)
non_defect_images5 = load_file(file_path=non_defect_images_path5, label=0)
non_defect_images6 = load_file(file_path=non_defect_images_path6, label=0)
non_defect_images7 = load_file(file_path=non_defect_images_path7, label=0)
mask_images = load_file(file_path=mask_images_path, label=-1)
# contribute the non defect dataset into one file
non_defect_images = np.concatenate((non_defect_images1, non_defect_images2))
non_defect_images = np.concatenate((non_defect_images, non_defect_images3))
non_defect_images = np.concatenate((non_defect_images, non_defect_images4))
non_defect_images = np.concatenate((non_defect_images, non_defect_images5))
non_defect_images = np.concatenate((non_defect_images, non_defect_images6))
non_defect_images = np.concatenate((non_defect_images, non_defect_images7))
print("defect_images.shape: {}\nnon_defect_images.shape: {}\nmask_images.shape:{} \n".format(defect_images.shape, non_defect_images.shape, mask_images.s
# we shuffle the order of defect-free and defect images
np.random.shuffle(non_defect_images)
np.random.shuffle(defect_images)
# the class size is the min length compared with defect-free and defect images
class_size = defect_images.shape[0] if defect_images.shape[0] <= non_defect_images.shape[0] else non_defect_images.shape[0]
# declare dataset by concat defect_images and non_defect_images with length 0 to class_size
dataset = np.concatenate((defect_images[:class_size], non_defect_images[:class_size]))
# create an empty matrix X with is matrix of 256x4096 and has dataset length row
X = np.empty([dataset.shape[0], 256, 4096]).astype(int)
# create vector y which has dataset length
y = np.empty(dataset.shape[0]).astype(int)
# assign the X,y one-by-one
for i in range(dataset.shape[0]):
    X[i] = dataset[i]["data"]
    y[i] = dataset[i]["label"]
# since Keras acquire the Image input is a tensor type -> we reshape X
X = X.reshape(X.shape[0], 256, 4096, 1)
# display size of the label 0 and label 1 
np.unique(y, return_counts=True)
     
from tensorflow.keras import datasets, layers, models, optimizers, regularizers, callbacks
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
def create_model(image_shape=(256, 4096, 1), print_summary=False):
    # initial model
    model = models.Sequential()

    # CONV layer: filter 16, stride 7x7
    model.add(layers.Conv2D(16, (7, 7),input_shape=image_shape))
    # Batch Normalization layer -> avoid overfitting
    model.add(layers.BatchNormalization())
    # activation layer 
    model.add(layers.Activation('relu'))
    # max pooling -> reduce image size
    model.add(layers.MaxPooling2D((2, 2)))
    # droput later -> avoid overfitting
    model.add(layers.Dropout(0.25))

    model.add(layers.Conv2D(32, (5, 5), padding="same"))
    # Batch Normalization layer -> avoid overfitting
    model.add(layers.BatchNormalization())
    # activation layer 
    model.add(layers.Activation('relu'))
    # max pooling -> reduce image size
    model.add(layers.MaxPooling2D((2, 2)))
    # droput later -> avoid overfitting
    model.add(layers.Dropout(0.25))
    
    model.add(layers.Conv2D(64, (3, 3), padding="same"))
    # Batch Normalization layer -> avoid overfitting
    model.add(layers.BatchNormalization())
    # activation layer 
    model.add(layers.Activation('relu'))
    # max pooling -> reduce image size
    model.add(layers.MaxPooling2D((2, 2)))
    # droput later -> avoid overfitting
    model.add(layers.Dropout(0.25))
    
    model.add(layers.Conv2D(64, (3, 3), padding="same"))
    # Batch Normalization layer -> avoid overfitting
    model.add(layers.BatchNormalization())
    # activation layer 
    model.add(layers.Activation('relu'))
    # max pooling -> reduce image size
    model.add(layers.MaxPooling2D((2, 2)))
    # droput later -> avoid overfitting
    model.add(layers.Dropout(0.25))
    # flatten later -> from matrix to vector
    model.add(layers.Flatten())
    
    # fully connected layer -> nn layer with 64 nodes
    model.add(layers.Dense(64))
    # Batch Normalization layer -> avoid overfitting
    model.add(layers.BatchNormalization())
    # activation layer 
    model.add(layers.Activation('relu'))
    # droput later -> avoid overfitting
    model.add(layers.Dropout(0.25))

    # fully connected layer -> nn layer with 64 nodes
    model.add(layers.Dense(64))
    # Batch Normalization layer -> avoid overfitting
    model.add(layers.BatchNormalization())
    # activation layer 
    model.add(layers.Activation('relu'))
    # droput later -> avoid overfitting
    model.add(layers.Dropout(0.25))
    
    # output layer
    model.add(layers.Dense(1, activation='sigmoid'))
    
    # set model compiler
    model.compile(optimizer='SGD', loss='binary_crossentropy', metrics=['accuracy'])
    
    # show the CNN model detail
    if print_summary:
        model.summary()
    return model
def train_model(model, xtrain, ytrain, xval, yval, n_epoch, batch_size):
    # train CNN model
    # batch size to reduce memory usage
    # set early stopping to avoid overfitting
    
    earlystopping = EarlyStopping(monitor='val_accuracy', patience=2)
    filepath = project_path + "/model/weights-improvement-{epoch:02d}-{val_accuracy:.2f}.hdf5"
    
    checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint, earlystopping]

    history = model.fit(xtrain, ytrain, epochs=n_epoch, batch_size=batch_size, validation_data=(xval, yval), callbacks=[callbacks_list])
    return history

create_model(image_shape=(256, 4096, 1), print_summary=True)
from sklearn.model_selection import StratifiedKFold
# set number of split
kfold_splits = 4
# set number of epoch
n_epoch = 10
# set batch size
batch_size = 10

# create StratifiedKFold
skf = StratifiedKFold(n_splits=kfold_splits, shuffle=True)
for index, (train_indices, val_indices) in enumerate(skf.split(X, y)):
    print("Training on fold {}/{}...".format(index+1, kfold_splits))

    # declare x train and x validate
    xtrain, xval = X[train_indices], X[val_indices]
    # declare y train and y validate
    ytrain, yval = y[train_indices], y[val_indices]

    # print number of class portion
    print("ytrain: number of samples each class: {}".format(np.unique(ytrain, return_counts=True)))
    print("yval: number of samples each class: {}".format(np.unique(yval, return_counts=True)))

    # clear the model
    model = None
    # create cnn model
    model = create_model()

    print("Training new iteration on {} training samples, {} validation samples, this may be a while...".format(xtrain.shape[0], xval.shape[0]))
    
    # train CNN model
    history = train_model(model, xtrain, ytrain, xval, yval, n_epoch, batch_size)

    print("--------------------------------------------------------------------")
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
print("y_train: number of samples each class: {}".format(np.unique(y_train, return_counts=True)))
print("y_test: number of samples each class: {}".format(np.unique(y_test, return_counts=True)))
cnn_model = None
cnn_model = create_model(image_shape=(256, 4096, 1))
earlystopping = EarlyStopping(monitor='val_accuracy', patience=2)
filepath = project_path + "/model.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint, earlystopping]
cnn_model.fit(X_train, y_train, batch_size=10, epochs=10, validation_split=0.2, callbacks=callbacks_list)
cnn_model1 = create_model()
cnn_model1.load_weights(project_path + "/model.hdf5")
score, acc = cnn_model1.evaluate(X_test, y_test, verbose=0)
     