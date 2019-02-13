import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.models import Sequential, load_model
import matplotlib.pyplot as plt

img_width, img_height = 150, 150

## uncomment for using MobileNet model
# img_width, img_height = 224, 224

# simple CNN model
## ******************** change location ********************************************************************************
model_path = 'C:/Users/RS_Vulcan/Documents/vada_pav_data/models/model_3.h5'
model_weights_path = 'C:/Users/RS_Vulcan/Documents/vada_pav_data/models/model_3_weights.h5'

## uncomment for using MobileNet model
# model_path = 'C:/Users/RS_Vulcan/Documents/vada_pav_data/models/model_4.h5'
# model_weights_path = 'C:/Users/RS_Vulcan/Documents/vada_pav_data/models/model_4_weights.h5'

model = load_model(model_path)
model.load_weights(model_weights_path)


test_data = r'C:/Users/RS_Vulcan/Documents/vada_pav_data/test'
test_datagen = ImageDataGenerator(rescale=1. / 255)


# test_generator = test_datagen.flow_from_directory(test_data,
#                                                 shuffle=False,
#                                                 target_size=(img_width, img_height),
#                                                 batch_size=1,
#                                                 class_mode=None)

# test_generator.reset()
# pred = model.predict_generator(test_generator, verbose=1)

## prints probabilities as array for all the test samples
# print(pred)

i = 0
columns = 5
text_labels = []
preds = []
plt.figure(figsize=(50,40))
for batch in test_datagen.flow_from_directory(
    test_data,
    shuffle=True,
    target_size=(img_width, img_height),
    batch_size=1,
    class_mode=None):
    pred = model.predict(batch)
    preds.append(pred)
    if pred > 0.5:
        text_labels.append('a Vada Pav')
    else:
        text_labels.append('not a Vada Pav')
    plt.subplot(5 , columns, i + 1)
    # plt.title('This is ' + text_labels[i])
    plt.title('This is %s' %(text_labels[i]))
    plt.xticks([])
    plt.yticks([])
    plt.xlabel('P {}' .format(str(preds[i])))
    imgplot = plt.imshow(batch[0])
    i += 1
#     plt.show()
    if i % 25 == 0:
      break
# plt.tight_layout(pad=1.0, w_pad=2.5, h_pad=3.0)
plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9 , top=0.9, wspace=1.0, hspace=1.0)
plt.show()
