# -*- coding: utf-8 -*-
"""face_mask_detection.ipynb
"""

# Commented out IPython magic to ensure Python compatibility.
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model, load_model, Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from shutil import copyfile
import random
from google.colab import drive
drive.mount('/content/drive')
# %matplotlib inline

img1 = cv2.imread('/content/drive/My Drive/Face Mask Detection/dataset/with mask/0-with-mask.jpg')
img1 = img1[:,:,::-1]
plt.imshow(img1)
plt.figure(figsize=(5,5))
img2 = cv2.imread('/content/drive/My Drive/Face Mask Detection/dataset/without mask/0.jpg')
img2 = img2[:,:,::-1]

plt.imshow(img2)
plt.show()

try:
    os.mkdir('/content/drive/My Drive/Colab Notebooks/Mask/')
    os.mkdir('/content/drive/My Drive/Colab Notebooks/Mask/training/')
    os.mkdir('/content/drive/My Drive/Colab Notebooks/Mask/training/With Mask/')
    os.mkdir('/content/drive/My Drive/Colab Notebooks/Mask/training/Without Mask/')
    os.mkdir('/content/drive/My Drive/Colab Notebooks/Mask/testing/')
    os.mkdir('/content/drive/My Drive/Colab Notebooks/Mask/testing/With Mask/')
    os.mkdir('/content/drive/My Drive/Colab Notebooks/Mask/testing/Without Mask/')
except OSError:
    pass

def split_data(SOURCE, TRAINING, TESTING, SPLIT_SIZE = 1.0):
  '''
  This function will split the current data into two different folders.
  One for Training and one for Testing
  '''
  sources_list = os.listdir(SOURCE)
  random.sample(sources_list, len(sources_list))
  split_len = int(round(len(sources_list)*SPLIT_SIZE))

  for i in range(split_len):
    if os.path.getsize(SOURCE + sources_list[i]) > 1:
      copyfile(os.path.join(SOURCE, sources_list[i]), os.path.join(TRAINING, sources_list[i]))

  for i in range(split_len, len(sources_list)):
    if os.path.getsize(SOURCE + sources_list[i]) > 1:
      copyfile(os.path.join(SOURCE, sources_list[i]), os.path.join(TESTING, sources_list[i]))

MASK_SOURCE_DIR = "/content/drive/My Drive/Colab Notebooks/Mask Detection/dataset/with mask/"
TRAINING_MASK_DIR = "/content/drive/My Drive/Colab Notebooks/Mask/training/With Mask/"
TESTING_MASK_DIR = "/content/drive/My Drive/Colab Notebooks/Mask/testing/With Mask/"
NONMASK_SOURCE_DIR = "/content/drive/My Drive/Colab Notebooks/Mask Detection/dataset/without mask/"
TRAINING_NONMASK_DIR = "/content/drive/My Drive/Colab Notebooks/Mask/training/Without Mask/"
TESTING_NONMASK_DIR = "/content/drive/My Drive/Colab Notebooks/Mask/testing/Without Mask/"

split_data(MASK_SOURCE_DIR, TRAINING_MASK_DIR, TESTING_MASK_DIR, SPLIT_SIZE = 0.9)
split_data(NONMASK_SOURCE_DIR, TRAINING_NONMASK_DIR, TESTING_NONMASK_DIR, SPLIT_SIZE = 0.9)

def load_data(dir_path):
  '''
  This function will load training and validation data.
  '''
  datagen = ImageDataGenerator(rescale=1.0/255,
                                rotation_range = 40,
                                width_shift_range = 0.2,
                                height_shift_range = 0.2,
                                zoom_range = 0.2,
                                shear_range = 0.2,
                                horizontal_flip = True,
                                fill_mode = 'nearest'
                                )

  generator = datagen.flow_from_directory(dir_path,
                                          batch_size = 10,
                                          class_mode = 'categorical',
                                          target_size = (150,150))
  return generator

train_generator = load_data('/content/drive/My Drive/Colab Notebooks/Mask/training')
validation_generator = load_data('/content/drive/My Drive/Colab Notebooks/Mask/testing')

from tensorflow.keras.applications.vgg16 import VGG16

pretrained_model = VGG16(
    input_shape = (150,150,3),
    weights = 'imagenet',
    include_top = False
)
pretrained_model.trainable = False

pretrained_model.summary()

last_layer_output = pretrained_model.get_layer('block5_pool').output
x = Flatten()(last_layer_output)
x = Dense(512, activation='relu')(x)
x = Dense(2, activation='softmax')(x)

model = Model(pretrained_model.input, x)

model.summary()

model.compile(optimizer = 'adam',
              loss = 'categorical_crossentropy',
              metrics = ['accuracy'])

class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('accuracy')>=0.99):
      print("\nReached 99 % accuracy")
      self.model.stop_training = True
callbacks = myCallback()

history = model.fit(train_generator, 
                    validation_data=validation_generator,
                    epochs = 60,
                    callbacks=[callbacks])

train_acc = history.history['accuracy']
validation_acc = history.history['val_accuracy']
train_loss = history.history['loss']
validation_loss = history.history['val_loss']
epochs = range(len(train_acc))

plt.plot(epochs, train_acc,'r', label='Train Accuracy' )
plt.plot(epochs, validation_acc, 'g', label = 'Validation Accuracy')
plt.title("Trainining and Validation Accuracy")

plt.figure()

plt.plot(epochs, train_loss, 'r', label = 'Training Loss')
plt.plot(epochs, validation_loss, 'g', label = 'Validation Loss')
plt.title("Training and Validation Loss")
plt.show()

model.save('/content/drive/My Drive/Colab Notebooks/Mask Detection/mask_modelbig.h5')

model = load_model('/content/drive/My Drive/Colab Notebooks/Mask Detection/mask_modelbig.h5')