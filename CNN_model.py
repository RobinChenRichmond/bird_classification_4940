import os,random
import math
import numpy as np
import matplotlib.pyplot as plt
import random, sys, keras
from keras.utils import np_utils
import keras.models as models
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.core import Reshape,Dense,Dropout,Activation,Flatten
from keras.layers.noise import GaussianNoise
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.regularizers import *
from keras.optimizers import adam
from scipy.misc import imresize, imread
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
config.log_device_placement = True
sess = tf.Session(config=config)
set_session(sess)  # set this TensorFlow session as the default session for Keras




## Step1: Data Pre-Processing


# initialize data and labels, and import dataset from DIR
data = []
label = []
DIR = "/Users/guanyuchen/Desktop/Github/bird_classification_4940/bird_dataset"
folders = os.listdir(DIR)
map = {}

# clean the folders to remove useless system files
index = 0
for foldername in folders:
    if foldername.startswith("."):
        folders.remove(foldername)

# extract data from folders and do the image pre-processing
for foldername in folders:
    print("working on "+foldername)
    map[foldername] = index
    index = index+1
    if foldername.startswith("."):
        continue
    files = os.listdir(DIR +"/" + foldername)
    for file in files:
        if not file.endswith(".jpg"):
            continue
        else:
            img = imread(DIR +"/" + foldername + "/" + file)
            img = imresize(img, (128, 128))
            data.append(img)
            label.append(map[foldername])

# split the data into train & test sets, and convert them into correct format for the Keras model
X_train, X_test, Y_train, Y_test = train_test_split(data, label, test_size=0.2, stratify=label, shuffle=True)
X_train = np.asarray(X_train)
X_test = np.asarray(X_test)
Y_train = np.asarray(Y_train)
Y_test = np.asarray(Y_test)
Y_train = keras.utils.to_categorical(Y_train, len(folders))
Y_test = keras.utils.to_categorical(Y_test, len(folders))




## Step2: Build the lite version VGG model using Keras primitives


# set parameters
in_shp = list(X_train.shape[1:])
classes = folders # name of different classes
nb_epoch = 100    # number of epochs to train on
batch_size = 32   # training batch size
dr_1 = 0.25       # dropout rate (%) for convolutional layers
dr_2 = 0.5        # dropout rate (%) for dense layers

# build the CNN model
model = models.Sequential()
model.add(Conv2D(32, (3,3), input_shape=X_train.shape[1:], activation="relu"))
model.add(Dropout(dr_1))
model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(32, (3,3), activation="relu"))
model.add(Dropout(dr_1))
model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(64, (3,3), activation="relu"))
model.add(Dropout(dr_1))
model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(64, (3,3), activation="relu"))
model.add(Dropout(dr_1))
model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(dr_2))
model.add(Dense(512, activation='relu'))
model.add(Dropout(dr_2))
model.add(Dense(len(classes), activation='softmax'))
model.add(Reshape([len(classes)]))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# image augmentation
train_datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True)
test_datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True)

train_datagen.fit(X_train)
test_datagen.fit(X_test)

# train the model and store the weight
filepath = '4940weight.wts.h5'
history = model.fit_generator(train_datagen.flow(X_train, Y_train, batch_size=batch_size, shuffle=True),
                    steps_per_epoch=1+len(X_train) / batch_size, 
                    epochs=nb_epoch, verbose=1, 
                    validation_data=test_datagen.flow(X_test, Y_test, batch_size=batch_size),
                    validation_steps=1+len(X_test) / batch_size,
                    callbacks = [keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=True, mode='auto'),
                    keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='auto')])




## Step3: Data Analysis


# Re-load the best weights once training is finished
model.load_weights(filepath)

# Show simple version of performance
score = model.evaluate_generator(test_datagen.flow(X_test, Y_test, batch_size=batch_size), steps = 1+len(X_test) / batch_size)
print "Validation Loss and Accuracy: ",score

# Optional: show analysis graphs
plt.figure()
plt.title('Training performance')
plt.plot(history.epoch, history.history['loss'], label='train loss+error')
plt.plot(history.epoch, history.history['val_loss'], label='val_error')
plt.legend()
plt.show()

# helper method to plot the confusion matrix
def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues, labels=[]):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

# Plot confusion matrix for the whole dataset
test_Y_hat = model.predict_generator(test_datagen.flow(X_test, Y_test, batch_size=batch_size,shuffle=False), steps = 1+len(X_test) / batch_size)
conf = np.zeros([len(classes),len(classes)])
confnorm = np.zeros([len(classes),len(classes)])
for i in range(0,X_test.shape[0]):
    j = list(Y_test[i,:]).index(1)
    k = int(np.argmax(test_Y_hat[i,:]))
    conf[j,k] = conf[j,k] + 1

for i in range(0,len(classes)):
    confnorm[i,:] = conf[i,:] / np.sum(conf[i,:])
plot_confusion_matrix(confnorm, labels=classes)