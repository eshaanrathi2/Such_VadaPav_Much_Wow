import sys
import os
import numpy as np

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, load_model, Model
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
import tensorflow as tf

# uncomment if using GPU
# os.environ["CUDA_DEVICE_ORDER"]="0000:01:00.0"
# os.environ["CUDA_VISIBLE_DEVICES"]="0";
# K.tensorflow_backend._get_available_gpus() #checks if keras using GPU

DEV = False
argvs = sys.argv
argc = len(argvs)

if argc > 1 and (argvs[1] == "--development" or argvs[1] == "-d"):
  DEV = True

if DEV:
  epochs = 2
else:
  epochs = 50

img_width, img_height = 150, 150

## uncomment for using MobileNet model
# img_width, img_height = 224, 224

input_shape = (img_width, img_height, 3)

## ******************** change location ********************************************************************************
train_data = r'C:/Users/RS_Vulcan/Documents/vada_pav_data/train'
valid_data = r'C:/Users/RS_Vulcan/Documents/vada_pav_data/valid'

train_samples = 536
valid_samples = 106
# epochs = 50
batch_size = 16

# CNN model architecture 3 -------------------------------------------------------------------------

model = Sequential()

model.add(Conv2D(16, (3, 3), activation='relu', input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64, activation='relu'))

model.add(Dropout(0.5))
model.add(Dense(1, activation = 'sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# MobileNet model architecture 4 -------------------------------------------------------------------

## uncomment for using MobileNet model

# from keras.applications import MobileNet

# base_model=MobileNet(weights='imagenet', include_top=False, input_shape=input_shape) 
# x=base_model.output
# x=GlobalAveragePooling2D()(x)
# x=Dense(1024,activation='relu')(x) 
# x=Dense(1024,activation='relu')(x) 
# x=Dense(512,activation='relu')(x) 
# preds=Dense(1,activation='sigmoid')(x) 
# model=Model(inputs=base_model.input,outputs=preds)
## model.summary()

# for layer in model.layers[:20]:
#     layer.trainable=False
# for layer in model.layers[20:]:
#     layer.trainable=True

# data generator -----------------------------------------------------------------------------------

train_datagen = ImageDataGenerator(rotation_range=15,
                               rescale=1./ 255,
                               width_shift_range=0.1,
                               height_shift_range=0.1,
                               shear_range=0.01,
                               zoom_range=[0.9, 1.25],
                               horizontal_flip=True,
                               vertical_flip=False, # can be true as vadapav looks similar even upside-down
                               fill_mode='nearest',
                               data_format='channels_last',
                               brightness_range=[0.5, 1.5])

valid_datagen = ImageDataGenerator(rotation_range=15,
                               rescale=1./ 255,
                               width_shift_range=0.1,
                               height_shift_range=0.1,
                               shear_range=0.01,
                               zoom_range=[0.9, 1.25],
                               horizontal_flip=True,
                               vertical_flip=False,
                               fill_mode='nearest',
                               data_format='channels_last',
                               brightness_range=[0.5, 1.5])


train_generator = train_datagen.flow_from_directory(train_data,
                                                    target_size=(img_width, img_height),
                                                    batch_size=batch_size,
                                                    class_mode='binary')

valid_generator = valid_datagen.flow_from_directory(valid_data,
                                                    target_size=(img_width, img_height),
                                                    batch_size=batch_size,
                                                    class_mode='binary')

# train model --------------------------------------------------------------------------------------

model.fit_generator(train_generator,
                    steps_per_epoch=train_samples // batch_size, #50
                    epochs=epochs,
                    validation_data=valid_generator,
                    validation_steps= valid_samples // batch_size)  #50

# save model and weights ---------------------------------------------------------------------------
## ******************** change location ********************************************************************************
target_dir = 'C:/Users/RS_Vulcan/Documents/vada_pav_data/models/'
if not os.path.exists(target_dir):
  os.mkdir(target_dir)

## when trained again, model saved as model_4dum.h5 and weights as model_4dum_weights.h5 in models
## change model_path and model_weights_path path in app.py accordingly
## ******************** change location ********************************************************************************
model.save('C:/Users/RS_Vulcan/Documents/vada_pav_data/models/model_4dum.h5')  
model.save_weights('C:/Users/RS_Vulcan/Documents/vada_pav_data/models/model_4dum_weights.h5')