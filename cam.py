import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
# import matplotlib.pyplot as plt
# import cv2
# import os

def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img

img_path = './content/sample_data/dog.png'

preprocessed_img = preprocess_image(img_path)

model = VGG16(weights='imagenet')
pred = model.predict(preprocessed_img)
decode_predictions(pred, top=5)

print('Predicted:', decode_predictions(pred, top=5))