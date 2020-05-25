#!/usr/bin/env python
# coding: utf-8

from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.models import Sequential

model = Sequential()

model.add(Convolution2D(filters=32, 
                        kernel_size=(3,3), 
                        activation='relu',
                   input_shape=(64, 64, 3)
                       ))
model.summary()

model.add(MaxPooling2D(pool_size=(2, 2)))
model.summary()

model.add(Flatten())
model.summary()

model.add(Dense(units=128, activation='relu'))
model.summary()

model.add(Dense(units=1, activation='sigmoid'))
model.summary()

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)
training_set = train_datagen.flow_from_directory(
        'training_set',
        target_size=[64,64],
        class_mode='binary')
test_set = test_datagen.flow_from_directory(
        'test_set',
        target_size=[64,64],
        class_mode='binary')

no_of_epocs = 10
history = model.fit(
            training_set,
            steps_per_epoch=100,
            epochs=no_of_epocs,
            validation_data=test_set,
            validation_steps=10)

accuracy = history.history['accuracy'][9] * 100

f=open("accuracy.txt",'w')
f.write("%d" % int(history.history['accuracy'][9] * 100))
f.close()
