# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 16:12:56 2023

@author: hp
"""

from tensorflow.keras.models import load_model



import numpy as np

from tensorflow.keras.preprocessing.image import load_img, img_to_array



# Load the model

model = load_model("model_inception.h5")



# Define the class names

class_names =['bacterial blight','curl virus','fresh cotton leaf','sucking and chewing pest']



# Load an image from the test set

img = load_img("dataset/train/Curl Virus/curl00.jpg", target_size=(255, 255))



# Convert the image to an array

img_array = img_to_array(img)
img_array = np.reshape(img_array, (1, 224, 224, 3))

# Get the model predictions

predictions = model.predict(img_array)

print("predictions:", predictions)



# Get the class index with the highest predicted probability

class_index = np.argmax(predictions[0])



# Get the predicted class label

predicted_label = class_names[class_index]



print("The image is predicted to be '{}'.".format(predicted_label))