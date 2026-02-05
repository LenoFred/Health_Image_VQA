import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
import cv2
from PIL import Image
import random


def build_binary_model():
    model = models.resnet18(pretrained=False)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 1)
    return model

# ---------------------------
# Preprocessing Functions
# ---------------------------

def load_image(image_path):
    image = Image.open(image_path).convert("RGB")
    return np.array(image)

def normalize_image(image):
    image = (image * 255).astype(np.uint8) if image.max() <= 1.0 else image
    image = image.astype(np.float32) / 255.0
    return image

def resize_image(image, size=(224, 224)):
    image = cv2.resize(image, size)
    return image

def augment_image(image):
    if random.random() > 0.5:
        image = cv2.flip(image, 1)
    if random.random() > 0.5:
        angle = random.randint(-15, 15)
        h, w = image.shape[:2]
        M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1)
        image = cv2.warpAffine(image, M, (w, h))
    return image

def generate_segmentation_mask(image):
    if image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)

    if len(image.shape) == 3 and image.shape[2] == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    blurred = cv2.normalize(blurred, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    mask = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                 cv2.THRESH_BINARY_INV, blockSize=11, C=2)

    # Combine all significant contours, not just the largest
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    new_mask = np.zeros_like(mask)
    for contour in contours:
        area = cv2.contourArea(contour)
        if 1000 < area < 30000:  # filter out small and very large areas like shoulders
            cv2.drawContours(new_mask, [contour], -1, 255, thickness=cv2.FILLED)
    mask = new_mask

    return mask

def preprocess_image(image_path, augment=False):
    image = load_image(image_path)
    image = normalize_image(image)
    image = resize_image(image)
    if augment:
        image = augment_image(image)
    image = np.transpose(image, (2, 0, 1))
    image_tensor = torch.tensor(image, dtype=torch.float32)
    return image_tensor

# # ---------------------------
# # Visualization for Testing
# # ---------------------------
# if __name__ == "__main__":
#     import matplotlib.pyplot as plt

#     test_image_path = "datasets/NIH_Chest_Xray/00000035_001.PNG"
#     image = load_image(test_image_path)
#     normalized = normalize_image(image)
#     resized = resize_image(normalized)
#     augmented = augment_image(resized.copy())
#     mask = generate_segmentation_mask(resized)
#     masked_output = cv2.bitwise_and((resized * 255).astype(np.uint8), (resized * 255).astype(np.uint8), mask=mask)

#     fig, axs = plt.subplots(1, 5, figsize=(20, 4))
#     axs[0].imshow(image)
#     axs[0].set_title("Original")
#     axs[1].imshow(resized)
#     axs[1].set_title("Resized")
#     axs[2].imshow(augmented)
#     axs[2].set_title("Augmented")
#     axs[3].imshow(mask, cmap="gray")
#     axs[3].set_title("Segmentation Mask")
#     axs[4].imshow(masked_output)
#     axs[4].set_title("Image + Mask Overlay")

#     for ax in axs:
#         ax.axis("off")
#     plt.tight_layout()
#     plt.show()
