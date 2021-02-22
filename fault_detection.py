# -*- coding: utf-8 -*-
"""
Created on Fri Feb 12 00:58:10 2021

@author: HP
"""

import matplotlib.pyplot as plt
import seaborn as sns

import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D , MaxPool2D , Flatten , Dropout 
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from sklearn.metrics import classification_report,confusion_matrix
import tensorflow as tf
import cv2
import os
import glob
import numpy as np
from sklearn.model_selection import train_test_split

pathdir = input('Enter the path of the folder containing the folders OK and NOK: ')
img_size = 224
def save_model():
    model.save(pathdir)
def load_model():
    keras.models.load_model(pathdir)
def get_model():
    if glob.glob(pathdir+"/*.pb"):
      model = load_model()
      is_trained = True
    else:
      model = Sequential()
      model.add(Conv2D(32,3,padding="same", activation="relu", input_shape=(img_size,img_size,3)))
      model.add(MaxPool2D())
      
      model.add(Conv2D(32, 3, padding="same", activation="relu"))
      model.add(MaxPool2D())
      
      model.add(Conv2D(64, 3, padding="same", activation="relu"))
      model.add(MaxPool2D())
      model.add(Dropout(0.4))
      
      model.add(Flatten())
      model.add(Dense(128,activation="relu"))
      model.add(Dense(2, activation="softmax"))
      opt = Adam(lr=0.000001)
      model.compile(optimizer = opt , loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True) , metrics = ['accuracy'])
      is_trained = False
    return model,is_trained

def get_dataset(path, y, x, h, label):
    training_data = []
    for img in os.listdir(path):
        pic = cv2.imread(os.path.join(path,img))    #read images
        pic = cv2.cvtColor(pic,cv2.COLOR_BGR2RGB)   #convert BGR to RGB
        pic = pic[y:y+h, x:x+h] #630:1200 for ok    # Crop
        pic = cv2.resize(pic,(img_size,img_size))   #Resize images
        training_data.append([pic,label])
    
    return np.array(training_data)

def create_train_test(dataset):
    X = dataset[:,0]
    Y = dataset[:,1]
    #Split Data in training and testing
    ip_train, ip_test, op_train, op_test = train_test_split(X,Y, test_size = 0.2, random_state = 42, stratify = Y) 
    
    x_train = []
    y_train = []
    x_val = []
    y_val = []
    
    for feature, label in zip(ip_train,op_train):
      x_train.append(feature)
      y_train.append(label)
    
    for feature, label in zip(ip_test,op_test):
      x_val.append(feature)
      y_val.append(label)
      
    # Normalize the data
    x_train = np.array(x_train) / 255
    x_val = np.array(x_val) / 255
    
    x_train.reshape(-1, img_size, img_size, 1)
    y_train = np.array(y_train)
    
    x_val.reshape(-1, img_size, img_size, 1)
    y_val = np.array(y_val)
    
    datagen = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            rotation_range = 30,  # randomly rotate images in the range (degrees, 0 to 180)
            zoom_range = 0.2, # Randomly zoom image 
            width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
            horizontal_flip = True,  # randomly flip images
            vertical_flip=False)  # randomly flip images
    
    
    datagen.fit(x_train)
    return x_train,y_train,x_val, y_val


model, is_trained = get_model() #Load of create a model


ok_dataset = get_dataset(path= pathdir+'/OK', y=300, x=630, h=570, label=0)
nok_dataset = get_dataset(path= pathdir+'/NOK', y=300, x=570, h=570, label=1)
total_dataset = np.concatenate((ok_dataset, nok_dataset), axis=0)
x_train,y_train,x_val, y_val = create_train_test(total_dataset)
if is_trained==True:
    predictions = np.argmax(model.predict(x_val), axis=-1)
else:
    history = model.fit(x_train,y_train,epochs = 500 , validation_data = (x_val, y_val))
    save_model()
    predictions = np.argmax(model.predict(x_val), axis=-1)
# predictions = predictions.reshape(1,-1)[0]
print(classification_report(y_val, predictions, target_names = ['OK (Class 0)','NOK (Class 1)']))






