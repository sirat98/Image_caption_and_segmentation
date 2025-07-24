import streamlit as st
import numpy as np
import cv2
import os
import pickle
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO

# --- CONFIG ---
IMAGE_SIZE = 128
MAX_LENGTH = 51

# --- IOU Custom Metric ---
from tensorflow.keras.layers import Flatten
def iou_coeff(y_true, y_pred):
    y_true_f = Flatten()(y_true)
    y_pred_f = Flatten()(y_pred)
    intersection = 2. * (y_true_f * y_pred_f) + tf.keras.backend.epsilon()
    union = y_true_f + y_pred_f + tf.keras.backend.epsilon()
    return intersection / union

# --- Load Models & Tokenizers ---
@st.cache_resource
def load_resources():
    caption_model = load_model("best_efficientnet_model (1).keras")
    unet_model = load_model("unet_adamw.h5", custom_objects={"iou_coeff": iou_coeff})
    with open("tokenizer (2).pkl", "rb") as f:
        tokenizer = pickle.load(f)
    with open("features_efficientnet (1).pkl", "rb") as f:
        features = pickle.load(f)
    return caption_model, unet_model, tokenizer, features

# --- Preprocess for U-Net ---
def preprocess_image(image, target_size=(128, 128)):
    image = image.convert("RGB")
    image = image.resize(target_size)
    img_array = img_to_array(image) / 255.0
    return np.expand_dims(img_array, axis=0), img_array

# --- Generate Caption and Segmentation ---
def generate_output_with_caption_and_segmentation(
    image: Image.Image, caption_model, unet_model, tokenizer, features
):
    image_filename = "uploaded_image.jpg"
    img_input = img_to_array(image.resize((300, 300)))
    img_input = tf.keras.applications.efficientnet.preprocess_input(img_input)
    img_input = np.expand_dims(img_input, axis=0)

    # Feature extraction
    if image_filename in features:
        image_feature = features[image_filename]
    else:
        base_model = tf.keras.applications.EfficientNetB3(weights='imagenet', include_top=False, pooling='avg')
        feature_extractor = tf.keras.Model(inputs=base_model.input, outputs=base_model.output)
        image_feature = feature_extractor.predict(img_input)[0]

    # Caption generation
    in_text = "startseq"
    for _ in range(MAX_LENGTH):
        seq = tokenizer.texts_to_sequences([in_text])[0]
        seq = pad_sequences([seq], maxlen=MAX_LENGTH)
        yhat = caption_model.predict([np.expand_dims(image_feature, axis=0), seq], verbose=0)
        yhat_index = np.argmax(yhat)
        word = tokenizer.index_word.get(yhat_index, None)
        if word is None or word == "endseq":
            break
        in_text += " " + word
    caption = in_text.replace("startseq", "").strip().capitalize()

    # Segmentation
    pre_img, img_array = preprocess_image(image)
    y_probas = np.stack([unet_model(pre_img, training=True) for _ in range(100)])
    result = y_probas.mean(axis=0)
    binary_mask = (result[0] > 0.5).astype(np.uint8)

    # Resize mask back to original image size
    original_size = image.size
    resized_mask = cv2.resize(binary_mask, original_size)

    # Overlay green on mask area
    overlay = np.array(image)
    green_layer = np.zeros_like(overlay)
    green_layer[:] = [0, 255, 0]
    mask_area = resized_mask == 1
    alpha = 0.5
    overlay[mask_area] = (alpha * green_layer[mask_area] + (1 - alpha) * overlay[mask_area]).astype(np.uint8)

    return caption, overlay, resized_mask, np.array(image)

# --- Streamlit UI ---
st.title(" Image Captioning + Segmentation")

uploaded_file = st.file_uploader("📤 Upload an Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    st.info("🔄 Processing Image...")
    image = Image.open(BytesIO(uploaded_file.read()))

    caption_model, unet_model, tokenizer, features = load_resources()
    caption, overlay, mask, original = generate_output_with_caption_and_segmentation(
        image, caption_model, unet_model, tokenizer, features
    )

    # Convert mask to visible grayscale (0–255)
    visible_mask = (mask * 255).astype(np.uint8)
    visible_mask_rgb = cv2.cvtColor(visible_mask, cv2.COLOR_GRAY2RGB)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.image(original, caption="Original Image", use_container_width=True)

    with col2:
        st.image(visible_mask_rgb, caption=" Segmentation Mask", use_container_width=True)

    with col3:
        st.image(overlay, caption=f"🟩 Overlay + Caption:\n{caption}", use_container_width=True)
