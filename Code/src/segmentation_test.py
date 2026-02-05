import numpy as np
import cv2
import tensorflow as tf

# Load the saved model
model = tf.keras.models.load_model('../models/segmentation/best_model.h5')
print('Model loaded successfully')
print(model.summary())

width = 256
height = 256

# Load test Image
testImg_path = '../Code/datasets/t2.PNG'
testImg = cv2.imread(testImg_path)
if testImg is None:
    raise RuntimeError(f"Failed to load {testImg_path}")
testImg2= cv2.resize(testImg, (width, height))
testImg2 = testImg2 / 255.0 
testImg_model = np.expand_dims(testImg2, axis=0)
print('Test image loaded successfully')

Prediction = model.predict(testImg_model)
result = Prediction[0]
print('Prediction shape:', Prediction.shape)

# Threshold the prediction to create a binary mask
result[result <= 0.5] = 0
result[result > 0.5] = 255

scalepercent = 60
w = int(result.shape[1] * scalepercent / 100)
h = int(result.shape[0] * scalepercent / 100)
dim = (w, h)
result = cv2.resize(testImg, dim, interpolation=cv2.INTER_AREA)
mask = cv2.resize(result, dim, interpolation=cv2.INTER_AREA)

# `result` is your thresholded mask of shape (256,256,1)
# mask_resized = cv2.resize(result.astype(np.uint8), dim, interpolation=cv2.INTER_AREA)
cv2.imshow('Predicted Mask', mask)
cv2.waitKey(0)
