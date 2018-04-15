import os,random
os.environ["KERAS_BACKEND"] = "tensorflow"
import math
import numpy as np
import matplotlib.pyplot as plt
import random, sys, keras
from keras.utils import np_utils
import keras.models as models
from keras.layers.core import Reshape,Dense,Dropout,Activation,Flatten
from keras.layers.noise import GaussianNoise
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.regularizers import *
from keras.optimizers import adam
from scipy.misc import imresize, imread
from sklearn.model_selection import train_test_split

data = []
label = []

DIR = "/Users/guanyuchen/Desktop/Github/bird_classification_4940/bird_dataset"

folders = os.listdir(DIR)

map = {}
index = 0
for foldername in folders:
    if foldername.startswith("."):
        folders.remove(foldername)

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
            img = imresize(img, (140, 140))
            data.append(img)
            label.append(map[foldername])

print(len(data))
print(len(label))

X_train, X_test, Y_train, Y_test = train_test_split(data, label, test_size=0.2, shuffle=True)

X_train = np.asarray(X_train)
X_test = np.asarray(X_test)
Y_train = np.asarray(Y_train)
Y_test = np.asarray(Y_test)

Y_train = keras.utils.to_categorical(Y_train, len(folders))
Y_test = keras.utils.to_categorical(Y_test, len(folders))

print(len(X_train))
print(len(X_test))
print(len(Y_train))
print(len(Y_test))

in_shp = list(X_train.shape[1:])
print X_train.shape, X_train.shape[1:], in_shp
classes = folders




# Build the lite version VGG model using Keras primitives --
#  - Reshape [N,2,128] to [N,1,2,128] on input
#  - Pass through 4 2DConv/ReLu layers
#  - Pass through 2 Dense layers (ReLu and Softmax)
#  - Perform categorical cross entropy optimization

# Set up some params
nb_epoch = 100     # number of epochs to train on
batch_size = 32  # training batch size
dr = 0.5 # dropout rate (%)

# build the CNN model
model = models.Sequential()
model.add(Conv2D(32, (3,3), input_shape=X_train.shape[1:], activation="relu"))
model.add(Dropout(dr))
model.add(ZeroPadding2D((2,2)))
model.add(Conv2D(64, (3,3), activation="relu"))
model.add(Dropout(dr))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(dr))
model.add(Dense(len(classes), activation='softmax'))
model.add(Reshape([len(classes)]))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# Train the dataset
#  and store the weights
filepath = '4940weight.wts.h5'
history = model.fit(X_train,Y_train,batch_size=batch_size,
    nb_epoch=nb_epoch,
    verbose=1,
    validation_data=(X_test, Y_test),
    callbacks = [
        keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=True, mode='auto'),
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=0, mode='auto')]
    )


# Re-load the best weights once training is finished
# model.load_weights(filepath)


# Show simple version of performance
score = model.evaluate(X_test, Y_test, verbose=0, batch_size=batch_size)
print "Validation Loss and Accuracy: ",score


# Optional: show analysis graphs
plt.figure()
plt.title('Training performance')
plt.plot(history.epoch, history.history['loss'], label='train loss+error')
plt.plot(history.epoch, history.history['val_loss'], label='val_error')
plt.legend()
plt.show()
'''

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
test_Y_hat = model.predict(X_test, batch_size=batch_size)
conf = np.zeros([len(classes),len(classes)])
confnorm = np.zeros([len(classes),len(classes)])
for i in range(0,X_test.shape[0]):
    j = list(Y_test[i,:]).index(1)
    k = int(np.argmax(test_Y_hat[i,:]))
    conf[j,k] = conf[j,k] + 1

for i in range(0,len(classes)):
    confnorm[i,:] = conf[i,:] / np.sum(conf[i,:])
plot_confusion_matrix(confnorm, labels=classes)



# Accuracy and confusion matrix for data with each SNR
acc = {}
for snr in snrs:
  # extract classes @ SNR
  test_SNRs = map(lambda x: lbl[x][1], test_idx)
  test_X_i = X_test[np.where(np.array(test_SNRs)==snr)]
  test_Y_i = Y_test[np.where(np.array(test_SNRs)==snr)]

  # estimate classes
  test_Y_i_hat = model.predict(test_X_i)
  conf = np.zeros([len(classes),len(classes)])
  confnorm = np.zeros([len(classes),len(classes)])

  for i in range(0,test_X_i.shape[0]):
    j = list(test_Y_i[i,:]).index(1)
    k = int(np.argmax(test_Y_i_hat[i,:]))
    conf[j,k] = conf[j,k] + 1

  for i in range(0,len(classes)):
    confnorm[i,:] = conf[i,:] / np.sum(conf[i,:])

  #plt.figure()
  #plot_confusion_matrix(confnorm, labels=classes, title="ConvNet Confusion Matrix (SNR=%d)"%(snr))
  cor = np.sum(np.diag(conf))
  ncor = np.sum(conf) - cor
  print "SNR: ",snr, " Overall Accuracy: ", cor / (cor + ncor)
  acc[snr] = 1.0 * cor / (cor + ncor)

# Save results to a pickle file for plotting later
print acc
fd = open('results_cnn_d0.5.dat','wb')
cPickle.dump( ("CNN", 0.5, acc) , fd )

# Plot accuracy curve
plt.plot(snrs, map(lambda x: acc[x], snrs))
plt.xlabel("Signal to Noise Ratio")
plt.ylabel("Classification Accuracy")
plt.title("New Model Classification Accuracy on RadioML 2016.10 Alpha")
plt.show()
'''