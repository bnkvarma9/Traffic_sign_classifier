import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import csv
import random
import tensorflow as tf
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dropout,Dense, Conv2D,MaxPooling2D, Flatten,BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras import metrics
#from keras.callbacks import EarlyStopping
#from keras.optimizers import Adam,SGD
#from keras.metrics import AUC
datadir="BelgiumTSC_Training/Training"
categories=["0000"+str(i) for i in range(0,10)]
categories2=["000"+str(i) for i in range(10,62)]
categories=categories+categories2
training_data=[]
test_data=[]

for category in categories:
    path=os.path.join(datadir,category)
    file=open("BelgiumTSC_Training/Training/"+category+"/GT-"+category+".csv")
    csvreader=csv.reader(file,delimiter=';')
    header=next(csvreader)
    col=[header[0]]
    class_num=categories.index(category)
    for row in csvreader:
        img=cv2.imread(os.path.join(path,row[0]))

        siz=64
        new_array=cv2.resize(img,(64,64))
        training_data.append([new_array,class_num])


random.shuffle(training_data)

x=[]
y=[]
x_test=[]
y_test=[]

for feat,lab in training_data:
    x.append(feat)
    y.append(lab)
x_train=np.array(x).reshape(-1,64,64,3)

y_train=np.array(y)

#This is data augmentation for increasing the dataset
datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        #horizontal_flip=True,
        fill_mode='nearest')
datagen.fit(x_train)
#CNN layer starts from now

print("------------------------ADAM OPTMIZER------------------------------")
model=Sequential()

model.add(Conv2D(32,(3,3),activation='relu',input_shape=(64,64,3)))
#model.add(BatchNormalization())
model.add(MaxPooling2D((2,2),strides=2))
model.add(Conv2D(64,(3,3),strides=1,activation='relu'))
#model.add(BatchNormalization())
model.add(MaxPooling2D((2,2),strides=2))
model.add(Conv2D(128,(3,3),strides=1,activation='relu'))
#model.add(BatchNormalization())
model.add(MaxPooling2D((2,2),strides=2))
model.add(Flatten())
model.add(Dense(2000,activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(500,activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(62,activation='softmax'))


model.compile(optimizer="adam",loss='categorical_crossentropy',metrics=["accuracy"])

history=model.fit_generator(datagen.flow(x_train, y_train, batch_size=46),epochs=100)
#history=model.fit(x_train,y_train,epochs=10)
#plt.plot(history.history['accuracy'],color='green')
#plt.plot(history.history['loss'],color='red')
#model.evaluate(x_test,y_test)
#plt.title('model accuracy and loss')

#plt.ylabel(' accuracy,loss')
#plt.xlabel('epoch')
#plt.legend(['accuracy','loss'])
#plt.show()




model.save("traffic_sign_model_100_adam", save_format="h5")























