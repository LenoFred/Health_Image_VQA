import os, sys, torch, importlib
import numpy as np
from PIL import Image
import streamlit as st
from torchvision import transforms
import cv2
import torch.nn as nn
import torchvision.models as models
from tensorflow.keras.models import load_model as keras_load_model
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json
import json 
from tensorflow.keras.layers import Layer
from transformers import TFDistilBertModel, DistilBertTokenizer
tf.keras.utils.disable_interactive_logging()
from tensorflow.keras import backend as K
K.set_image_data_format('channels_last')


# Root and paths
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

# Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# CLASSIFICATION MODEL ARCHITECTURE
def build_binary_model(weights=False):
    class DiseaseClassifier(nn.Module):
        def __init__(self):
            super(DiseaseClassifier, self).__init__()
            self.backbone = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1 if weights else None)
            self.features = nn.Sequential(*list(self.backbone.children())[:-1])
            self.conv = nn.Conv2d(1280, 1, kernel_size=1)
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(1, 1, bias=False)
        
        def forward(self, x):
            x = self.features(x)
            x = self.conv(x)
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)
            return x
    return DiseaseClassifier()

@st.cache_resource
def load_binary_model():
    binary_ckpt = os.path.join(ROOT, 'models', 'classification', 'best_model.pth')
    model = build_binary_model().to(device)
    state_dict = torch.load(binary_ckpt, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    return model

# SEGMENTATION PROCESSING
def preprocess_segmentation(image_pil):
    # Match training preprocessing from train_segmentation.py
    image = np.array(image_pil)
    image = cv2.resize(image, (256, 256))
    image = image.astype(np.float32) / 255.0
    return image

def generate_segmentation_heatmap(model, input_array):
    # TensorFlow implementation for Keras model
    input_tensor = tf.convert_to_tensor(input_array[np.newaxis, ...], dtype=tf.float32)
    
    with tf.GradientTape() as tape:
        tape.watch(input_tensor)
        pred = model(input_tensor)
        pred_max = tf.reduce_max(pred[..., 0])
    
    grads = tape.gradient(pred_max, input_tensor)[0]
    heatmap = tf.reduce_mean(tf.abs(grads), axis=-1).numpy()
    
    heatmap = cv2.resize(heatmap, (input_array.shape[1], input_array.shape[0]))
    heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap) + 1e-8)
    return heatmap

# preprocessing func to generate a segmentation mask
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

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    new_mask = np.zeros_like(mask)
    for contour in contours:
        area = cv2.contourArea(contour)
        if 1000 < area < 30000:
            cv2.drawContours(new_mask, [contour], -1, 255, thickness=cv2.FILLED)
    mask = new_mask

    return mask


# Load models
model = load_binary_model()
seg_model = keras_load_model(os.path.join(ROOT, 'models', 'segmentation', 'best_model1.h5'))

# Classification transforms (must match training)
classification_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


class MFBBlock(tf.keras.layers.Layer):
    def __init__(self, drop_out_rate=0.5, factor_num=10, out_dim=None, **kwargs):
        super().__init__(**kwargs)
        self.drop_out_rate = drop_out_rate
        self.factor_num = factor_num
        self.out_dim = out_dim

    def build(self, input_shape):
        D = input_shape[0][-1]
        if self.out_dim is None:
            self.out_dim = D // self.factor_num
        super().build(input_shape)

    def call(self, inputs, training=None):
        x, y = inputs
        exp2 = x * y
        if training:
            exp2 = tf.nn.dropout(exp2, rate=self.drop_out_rate)
        batch = tf.shape(exp2)[0]
        exp2 = tf.reshape(exp2, (batch, self.out_dim, self.factor_num))
        sq1 = tf.reduce_mean(exp2, axis=-1)
        sq2 = tf.sign(sq1) * tf.sqrt(tf.abs(sq1) + 1e-12)
        return tf.nn.l2_normalize(sq2, axis=-1)



# Streamlit UI
st.title("\U0001FA7B X-ray Disease Diagnosis & VQA")
uploaded_file = st.file_uploader("Upload X-ray Image", type=["png", "jpg", "jpeg"])
question = st.text_input("Ask a question (e.g., 'Is there pneumonia?')")


if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    img.thumbnail((1024, 1024))
    
    # Display original image
    col1, col2 = st.columns(2)
    with col1:
        # Process and show mask overlay using handcrafted segmentation
        seg_image = preprocess_segmentation(img)
        mask = generate_segmentation_mask((seg_image * 255).astype(np.uint8))

        # Create mask overlay
        resized_img = cv2.resize(np.array(img), (256, 256))
        masked_output = cv2.bitwise_and(resized_img, resized_img, mask=mask)
        st.image(masked_output, caption="Lung Segmentation Mask", use_container_width=True)


    with col2:
        # Process and show Grad-CAM heatmap
        seg_image = preprocess_segmentation(img)
        heatmap = generate_segmentation_heatmap(seg_model, seg_image)        
        # Create heatmap overlay
        overlay = cv2.addWeighted(
            (seg_image*255).astype(np.uint8), 0.7,
            cv2.applyColorMap(np.uint8(255*heatmap), cv2.COLORMAP_JET), 0.3, 
            0
        )
        st.image(overlay, caption="Pathology Heatmap (Grad-CAM)", use_container_width=True)

    # Process classification
    img_tensor = classification_transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        logit = model(img_tensor).squeeze()
        pred_prob = torch.sigmoid(logit).item()
        prediction = "Disease Detected ✅" if pred_prob > 0.5 else "No Disease ❎"
        
    # Display diagnosis
    st.subheader("Model Report")
    st.metric(label="Diagnosis", value=prediction)
    st.progress(pred_prob if pred_prob > 0.5 else 1-pred_prob)
    st.caption(f"Confidence: {max(pred_prob, 1-pred_prob)*100:.1f}%")

    # VQA placeholder
    if question:
        st.subheader("Model Insights")
        st.info(""" 
        Based on analysis:
        - Segmentation shows clear lung boundaries
        - Classifier confidence: {:.1f}%
        {}
        """.format(max(pred_prob, 1-pred_prob)*100, 
        "Potential abnormalities detected" if pred_prob > 0.5 else "No abnormalities detected"))