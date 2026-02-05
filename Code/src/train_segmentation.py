import cv2
import numpy as np
import glob
from tqdm import tqdm

Height = 256
Width = 256

path = "../Code/datasets/MontgomerySet/"
imagespath = glob.glob(path + "CXR_png/*.png")
leftmaskpath = glob.glob(path + "ManualMask/leftmask/*.png")
rightmaskpath = glob.glob(path + "ManualMask/rightmask/*.png")
# print(len(imagespath))
# print(len(leftmaskpath))
# print(len(rightmaskpath))

img = cv2.imread(imagespath[0], cv2.IMREAD_COLOR)
# print(img.shape)

# Rescale the Image
img = cv2.resize(img, (Width, Height))

left_mask = cv2.imread(leftmaskpath[0], cv2.IMREAD_GRAYSCALE)
right_mask = cv2.imread(rightmaskpath[0], cv2.IMREAD_GRAYSCALE)
left_mask = cv2.resize(left_mask, (Width, Height))
right_mask = cv2.resize(right_mask, (Width, Height))   
final_mask = left_mask + right_mask
# cv2.imshow("Image", img)
# cv2.imshow("Left Mask", left_mask)
# cv2.imshow("Right Mask", right_mask)
# cv2.imshow("Final Mask", final_mask)
# cv2.waitKey(0)

# reduce the mask size to 0 and 1
mask16 = cv2.resize(left_mask, (16, 16))
mask16[mask16 > 0] = 1
# print(mask16)

# PROCESSING THE FULL IMAGE DATA
allImages = []
maskImages = []

print("Processing Images")
for imgFile, leftMaskfile, rightMaskfile in tqdm(zip(imagespath, leftmaskpath, rightmaskpath), total=len(imagespath)):
    img = cv2.imread(imgFile, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (Width, Height))
    img = img.astype(np.float32) / 255.0
    allImages.append(img)

    lMask = cv2.imread(leftMaskfile, cv2.IMREAD_GRAYSCALE)
    rMask = cv2.imread(rightMaskfile, cv2.IMREAD_GRAYSCALE)
    leftMask = cv2.resize(lMask, (Width, Height))
    rightMask = cv2.resize(rMask, (Width, Height))   
    final_mask = left_mask + rightMask
    final_mask[final_mask > 0] = 1
    maskImages.append(final_mask)

allImages = np.array(allImages)
maskImages = np.array(maskImages) 
maskImages = maskImages.astype(int)

print('Shapes of Train images and masks:')
print(allImages.shape, maskImages.shape)


#  SPLIT THE DATASET INTO TRAINING AND VALIDATE
from sklearn.model_selection import train_test_split

train_img, val_img = train_test_split(allImages, test_size=0.1, random_state=42)
train_mask, val_mask = train_test_split(maskImages, test_size=0.1, random_state=42)

print('Shapes of Train images and masks:')
print(train_img.shape, train_mask.shape)
print('Shape of Validation images and masks:')
print(val_img.shape, val_mask.shape)

# Save the numpy arrays
print('Saving the numpy arrays')
np.save('../Code/datasets/MontgomerySet/train_img.npy', train_img)
np.save('../Code/datasets/MontgomerySet/train_mask.npy', train_mask)
np.save('../Code/datasets/MontgomerySet/val_img.npy', val_img)
np.save('../Code/datasets/MontgomerySet/val_mask.npy', val_mask)
print('Numpy arrays saved successfully')


# BUILDING THE UNET MODEL
import tensorflow as tf 
from tensorflow.keras.layers import * 
from tensorflow.keras.models import Model
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
# from keras import layers, models

# Convolutional Block 
def conv_block(inputs, filters):
    x = Conv2D(filters, (3, 3), padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(filters, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x


# Build the Model
def build_unet(shape):
    filters = [64, 128, 256, 512]
    inputs = Input((shape))

    skip_x = []
    x = inputs

    # Encoder
    for f in filters:
        x = conv_block(x, f)
        skip_x.append(x)
        x = MaxPooling2D((2, 2))(x)


    # Bridge
    x = conv_block(x, 1024)
    filters.reverse()
    skip_x.reverse()

    # Decoder
    for i, f in enumerate(filters):
        x = UpSampling2D((2, 2))(x)
        xs = skip_x[i]
        x = concatenate([x, xs])
        x = conv_block(x, f)

    # Output
    outputs = Conv2D(1, (1, 1), padding ='same')(x)
    outputs = Activation('sigmoid')(outputs)
    return Model(inputs, outputs)
    


# Load the Data
print('Start Loading Data')
train_img = np.load('../Code/datasets/MontgomerySet/train_img.npy')
train_mask = np.load('../Code/datasets/MontgomerySet/train_mask.npy')
val_img = np.load('../Code/datasets/MontgomerySet/val_img.npy')
val_mask = np.load('../Code/datasets/MontgomerySet/val_mask.npy')
print('Data Loaded Successfully')

# Build Model
lr = 1e-4
batchsize = 4
epochs = 20
Height = 256
Width = 256
shape = (256, 256, 3)
print('Building Model')
shape = (Height, Width, 3)
model = build_unet(shape)
print(model.summary())

model.compile(optimizer=Adam(lr), loss='binary_crossentropy', metrics=['accuracy'])
steps_per_epoch = np.ceil(len(train_img) / batchsize)
validation_steps = np.ceil(len(val_img) / batchsize)

best_model ='../models/segmentation/best_model.h5'
callbacks = [ModelCheckpoint(best_model, verbose=1, save_best_only=True), 
             EarlyStopping(monitor='val_accuracy', verbose=1, patience=10), 
             ReduceLROnPlateau(monitor='val_loss', factor=0.1, verbose=1, patience=5, min_lr=1e-7)]
history = model.fit(train_img, train_mask, batch_size=batchsize, epochs=epochs, verbose=1, 
                    validation_data=(val_img, val_mask), steps_per_epoch=steps_per_epoch, 
                    validation_steps=validation_steps, callbacks=callbacks, shuffle=True)

# save model to file path
model.save('../models/segmentation/best_model.h5')
print('Model Saved Successfully')

# Evaluate the model on validation data
loss, accuracy = model.evaluate(val_img, val_mask, verbose=1)
print(f'Validation Loss: {loss:.4f}, Validation Accuracy: {accuracy:.4f}')

logits = model.predict(val_img, batch_size=4, verbose=1)  
probs = tf.nn.sigmoid(logits).numpy().reshape(-1, Height, Width)

# Define Dice coefficient
def dice_coef_per_sample(y_true, y_pred, smooth=1e-6):
    y_true_f = y_true.flatten().astype(np.float32)
    y_pred_f = y_pred.flatten().astype(np.float32)
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)

# Compute per-sample Dice scores
dice_scores = [
    dice_coef_per_sample(val_mask[i], probs[i])
    for i in range(val_mask.shape[0])
]
mean_dice = np.mean(dice_scores)

# Compute per-sample BCE
bce_fn = BinaryCrossentropy(from_logits=False)
bce_values = [
    bce_fn(val_mask[i], probs[i]).numpy()
    for i in range(val_mask.shape[0])
]
mean_bce = np.mean(bce_values)

# Print results
print(f"Mean Dice Coefficient over {len(dice_scores)} samples: {mean_dice:.4f}")
print(f"Mean Binary Cross-Entropy over {len(bce_values)} samples: {mean_bce:.4f}")