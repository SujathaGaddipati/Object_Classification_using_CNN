# VGG CNN MODEL as feature extractor with augmentation
import cv2
import numpy as np
import pandas as pd
import os
from random import shuffle
from tqdm import tqdm
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score


# directory path for images
Class1 = r'<DIRECTORY_PATH>'
Class2 = r'<DIRECTORY_PATH>'
Class3 = r'<DIRECTORY_PATH>'
Class4 = r'<DIRECTORY_PATH>'


# Labelling function
def Class1_label(x):
    images = []
    for i in tqdm(os.listdir(x)):
            path = os.path.join(x,i)
            img = cv2.imread(path)
            img = cv2.resize(img, (64,64))
            images.append([np.array(img), np.array([1,0,0,0])])
    return(images)

def Class2_label(x):
    images = []
    for i in tqdm(os.listdir(x)):
            path = os.path.join(x,i)
            img = cv2.imread(path)
            img = cv2.resize(img, (64,64))
            images.append([np.array(img), np.array([0,1,0,0])])
    return(images)

def Class3_label(x):
    images = []
    for i in tqdm(os.listdir(x)):
            path = os.path.join(x,i)
            img = cv2.imread(path)
            img = cv2.resize(img, (64,64))
            images.append([np.array(img), np.array([0,0,1,0])])
    return(images)

def Class4_label(x):
    images = []
    for i in tqdm(os.listdir(x)):
            path = os.path.join(x,i)
            img = cv2.imread(path)
            img = cv2.resize(img, (64,64))
            images.append([np.array(img), np.array([0,0,0,1])])
    return(images)


# labelling the dataset 
Class1_data = Class1_label(Class1)
Class2_data = Class2_label(Class2)
Class3_data = Class3_label(Class3)
Class4_data = Class4_label(Class4)


# Image dataset preparation as numpy array and converting it into train, valiadtion and test set
Class1_img = np.array([i[0] for i in Class1_data]).reshape(-1,64,64,3)
Class1_label = np.array([i[1] for i in Class1_data])
Class2_img = np.array([i[0] for i in Class2_data]).reshape(-1,64,64,3)
Class2_label = np.array([i[1] for i in Class2_data])
Class3_img = np.array([i[0] for i in Class3_data]).reshape(-1,64,64,3)
Class3_label = np.array([i[1] for i in Class3_data])
Class4_img = np.array([i[0] for i in Class4_data]).reshape(-1,64,64,3)
Class4_label = np.array([i[1] for i in Class4_data])

data = np.vstack((Class1_img, Class2_img, Class3_img, Class4_img))
label = np.vstack((Class1_label, Class2_label, Class3_label, Class4_label))

train_x, valid_x, train_y, valid_y = train_test_split(data, label, test_size = 0.4, random_state =142)
val_x, test_x, val_y, test_y = train_test_split(valid_x, valid_y, test_size = 0.5, random_state = 42)
print(train_x.shape, train_y.shape, val_x.shape, val_y.shape, test_x.shape, test_y.shape)
test_x_scaled = test_x.astype('float32')
test_x_scaled /= 255
test_label = pd.DataFrame(test_y, columns = ['Class1', 'Class2', 'Class3', 'Class4'])


# Data Augmentation
train_datagen = ImageDataGenerator(rescale=1./255, zoom_range=0.3, rotation_range=50,
                                   width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2,
                                   horizontal_flip=True, fill_mode='nearest')

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow(train_x, train_y, batch_size=30)
val_generator = val_datagen.flow(val_x, val_y, batch_size=20)


# VGG CNN MODEL as feature extractor with all layers freezed
from keras.applications import vgg16
from keras.models import Model
import keras

vgg = vgg16.VGG16(include_top=False, weights='imagenet',
                  input_shape=(64,64,3))

output = vgg.layers[-1].output
output = keras.layers.Flatten()(output)
vgg_model = Model(vgg.input, output)

vgg_model.trainable = False
for layer in vgg_model.layers:
    layer.trainable = False

pd.set_option('max_colwidth', -1)
layers = [(layer, layer.name, layer.trainable) for layer in vgg_model.layers]
pd.DataFrame(layers, columns=['Layer Type', 'Layer Name', 'Layer Trainable'])


# CNN Model for training and clssification on vgg features
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, InputLayer
from keras.models import Sequential
from keras import optimizers

input_shape = vgg_model.output_shape[1]

model = Sequential()
model.add(vgg_model)
model.add(Dense(513, activation='relu', input_dim= input_shape))
model.add(Dropout(0.3))
model.add(Dense(513, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(train_y.shape[1], activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.RMSprop(lr=2e-5),
              metrics=['accuracy'])


# Model fit
history = model.fit_generator(train_generator, steps_per_epoch=100, epochs=100,
                              validation_data=val_generator, validation_steps=50,
                              verbose=1)


# Model accuracy
test_predictions = model.predict(test_x_scaled)
predictions = pd.DataFrame(test_predictions, columns = ['Class1', 'Class2', 'Class3', 'Class4'])
predictions = list(predictions.idxmax(axis = 1))
test_label = list(test_label.idxmax(axis = 1))
print(classification_report(test_label, predictions))
print(accuracy_score(test_label, predictions))

print('predictions are: ', predictions)
print('test_label are: ', test_label)


# Model Learning Curve
plt.figure(figsize=(20,10))
plt.subplot(1, 2, 1)
plt.suptitle('Optimizer : Rmsprop', fontsize=10)
plt.ylabel('Loss', fontsize=16)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend(loc='upper right')

plt.subplot(1, 2, 2)
plt.ylabel('Accuracy', fontsize=16)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend(loc='lower right')
plt.show()
