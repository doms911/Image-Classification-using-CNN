import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from keras.datasets import cifar10
from keras.utils import to_categorical

from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout, BatchNormalization
from keras.callbacks import EarlyStopping
from keras.preprocessing.image import ImageDataGenerator

from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import classification_report, confusion_matrix

# load  the data
# X = images, y = labels

(X_train, y_train), (X_test, y_test) = cifar10.load_data()
print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_test shape: {y_test.shape}")

# visualization of the data

# define the labels od the dataset 
labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# images displaying in grid format
# define the dimensions of the grid
width_grid = 10
length_grid = 10

# subplot returns the figure object and axes object
# can use axes object to plot specific figures at various locations
fig, axes = plt.subplots(length_grid, width_grid, figsize=(17, 17))

axes = axes.ravel() # flatten the 15 x 15 matrix into 225 array

n_train = len(X_train) # get the length of the train dataset

# select the random number from 0 to n_train
for i in np.arange(0, width_grid * length_grid):
    
    index = np.random.randint(0, n_train) # select a random number
    # read and display an image with the selected index
    axes[i].imshow(X_train[index,1:]) # accessing the index row and slicing the rest of the columns thus selecting a picture
    label_index = int(y_train[index])
    axes[i].set_title(labels[label_index], fontsize = 8)
    axes[i].axis('off')
    
plt.subplots_adjust(hspace=0.4)

plt.show()

plt.figure(2)
classes_name = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

classes, count = np.unique(y_train, return_counts=True)
plt.barh(classes_name, count)
plt.title('Class distribution in training set')
plt.figure(2)

plt.show()

classes, counts = np.unique(y_test, return_counts=True)
plt.barh(classes_name, count)
plt.title('Class distribution in test set')

plt.show()

# data preprocessing 

# scaling the date
X_train = X_train / 255.0
X_test = X_test / 255.0

# transform target variable into one-hotenconding

y_cat_train = to_categorical(y_train, 10)
y_cat_test = to_categorical(y_test, 10)

print(y_cat_train)

# MODEL BUILDING

INPUT_SHAPE = (32, 32, 3)
KERNEL_SIZE = (3, 3)
model = Sequential()

# convolutional layer
model.add(Conv2D(filters=32, kernel_size=KERNEL_SIZE, input_shape=INPUT_SHAPE, activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(filters=32, kernel_size=KERNEL_SIZE, input_shape=INPUT_SHAPE, activation='relu', padding='same'))
model.add(BatchNormalization())

# pooling layer
model.add(MaxPool2D(pool_size=(2, 2)))

# dropout layers
model.add(Dropout(0.25))

model.add(Conv2D(filters=64, kernel_size=KERNEL_SIZE, input_shape=INPUT_SHAPE, activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(filters=64, kernel_size=KERNEL_SIZE, input_shape=INPUT_SHAPE, activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(filters=128, kernel_size=KERNEL_SIZE, input_shape=INPUT_SHAPE, activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(filters=128, kernel_size=KERNEL_SIZE, input_shape=INPUT_SHAPE, activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
# model.add(Dropout(0.2))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(10, activation='softmax'))

METRICS = [
    'accuracy',
    tf.keras.metrics.Precision(name='precision'),
    tf.keras.metrics.Recall(name='recall')
]
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=METRICS)
model.summary()

# early stopping
early_stop = EarlyStopping(monitor='val-loss', patience=2)

# data augmentations
