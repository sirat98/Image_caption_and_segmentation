import streamlit as st
import numpy as np
import cv2
import pickle
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
from io import BytesIO

# --- CONFIG ---
IMAGE_SIZE = 128
MAX_LENGTH = 51
NUM_CLASSES = 80  # ← Your model has 80 classes

# --- Generate Unique Colors for 80 Classes ---
def generate_class_colors(num_classes):
    np.random.seed(42)
    return {
        i: tuple(np.random.randint(0, 256, 3).tolist())
        for i in range(num_classes)
    }

CLASS_COLORS = generate_class_colors(NUM_CLASSES)

def colorize_mask(mask):
    color_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for class_idx, color in CLASS_COLORS.items():
        color_mask[mask == class_idx] = color
    return color_mask

# --- Load Models and Tokenizer ---
@st.cache_resource
def load_resources():
    caption_model = load_model("best_efficientnet_model (1).keras")
    unet_model = load_model("unet_adamw_multi_class_epoch_25.h5", compile=False)
    with open("tokenizer (2).pkl", "rb") as f:
        tokenizer = pickle.load(f)
    with open("features_efficientnet (1).pkl", "rb") as f:
        features = pickle.load(f)
    return caption_model, unet_model, tokenizer, features

# --- Preprocessing ---
def preprocess_image(image, target_size=(128, 128)):
    image = image.convert("RGB")
    image = image.resize(target_size)
    img_array = img_to_array(image) / 255.0
    return np.expand_dims(img_array, axis=0)

# --- Captioning + Segmentation ---
def generate_outputs(image, caption_model, unet_model, tokenizer, features):
    # Image for captioning
    image_filename = "uploaded.jpg"
    img_input = img_to_array(image.resize((300, 300)))
    img_input = tf.keras.applications.efficientnet.preprocess_input(img_input)
    img_input = np.expand_dims(img_input, axis=0)

    # Feature extraction
    if image_filename in features:
        image_feature = features[image_filename]
    else:
        base_model = tf.keras.applications.EfficientNetB3(include_top=False, pooling='avg', weights='imagenet')
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

    # Segmentation prediction
    pre_img = preprocess_image(image)  # shape: (1, 128, 128, 3)
    prediction = unet_model.predict(pre_img)[0]  # shape: (128, 128, 80)
    class_mask = np.argmax(prediction, axis=-1).astype(np.uint8)  # shape: (128, 128)

    # Resize mask to original image size
    original_size = image.size
    resized_mask = cv2.resize(class_mask, original_size, interpolation=cv2.INTER_NEAREST)
    overlay = np.array(image).copy()
    color_mask = colorize_mask(resized_mask)

    # Alpha blending
    alpha = 0.5
    overlay = cv2.addWeighted(overlay, 1 - alpha, color_mask, alpha, 0)

    return caption, overlay, color_mask, np.array(image)

# --- Streamlit UI ---
st.title("Image Captioning + Multi-Class Segmentation")

uploaded_file = st.file_uploader("📤 Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(BytesIO(uploaded_file.read()))
    st.info("🔄 Processing...")

    caption_model, unet_model, tokenizer, features = load_resources()
    caption, overlay, color_mask, original = generate_outputs(image, caption_model, unet_model, tokenizer, features)

    # Layout display
    col1, col2, col3 = st.columns(3)

    with col1:
        st.image(original, caption="Original Image", use_container_width=True)

    with col2:
        st.image(color_mask, caption="Multi-Class Mask", use_container_width=True)

    with col3:
        st.image(overlay, caption=f"🟩 Overlay + Caption:\n{caption}", use_container_width=True)
