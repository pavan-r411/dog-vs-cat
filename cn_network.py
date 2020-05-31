# -*- coding: utf-8 -*-
"""
Created on Sat May 30 09:45:11 2020

@author: PAVAN
"""

# part-1 Building CNN Model

# importing the tensorflow oackage
import tensorflow as tf

# initializing the CNN 
classifier = tf.keras.Sequential()

# Step-1 Convolution Layer
classifier.add(tf.keras.layers.Convolution2D(32, kernel_size=(3, 3), input_shape=(64, 64, 3), activation='relu'))

# step-2 pooling
classifier.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))    

# Adding another layer
classifier.add(tf.keras.layers.Convolution2D(32, kernel_size=(3, 3), activation='relu'))
classifier.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))    


# step-3 flattening
classifier.add(tf.keras.layers.Flatten())

# Step-4 Full Connection
classifier.add(tf.keras.layers.Dense(units=128, activation='relu'))
classifier.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

# Compiling CNN
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])    

# Part-2 Fitting CNN to images
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory( 'dataset/training_set',
                                                target_size=(64, 64 ),
                                                batch_size=32, 
                                                class_mode='binary')

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size=(64, 64),
                                            batch_size=32,
                                            class_mode='binary')

classifier.fit(training_set,
               steps_per_epoch=8000,
               epochs=25,
               validation_data=test_set,
               validation_steps=2000)
import numpy as np
from keras.preprocessing import image

img_width, img_height = 64, 64
img = image.load_img('dataset/test_set/cats/cat.4008.jpg', target_size = (img_width, img_height))
img = image.img_to_array(img)
img = img/255
img = np.expand_dims(img, axis = 0)

c = classifier.predict(img)






