import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import pickle
import os
from googletrans import Translator
from gtts import gTTS


@st.cache_resource
def load_resources():
    model = load_model("best_efficientnet_model (1).keras")
    with open("tokenizer (2).pkl", "rb") as f:
        tokenizer = pickle.load(f)
    with open("features_efficientnet (1).pkl", "rb") as f:
        features_dict = pickle.load(f)
    return model, tokenizer, features_dict

caption_model, tokenizer, features_dict = load_resources()
max_length = 51


def generate_caption(model, tokenizer, photo_feature, max_length=50):
    in_text = 'startseq'
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = model.predict([photo_feature, sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = None
        for w, index in tokenizer.word_index.items():
            if index == yhat:
                word = w
                break
        if word is None:
            break
        in_text += ' ' + word
        if word == 'endseq':
            break
    final = in_text.replace('startseq', '').replace('endseq', '').strip()
    return final

st.title(" Multilingual Image Caption Generator with Speech")

uploaded_file = st.file_uploader("Upload an image", type=['jpg', 'jpeg', 'png'])

language_options = {
    "English": "en",
    "Hindi": "hi",
    "French": "fr",
    "German": "de",
    "Spanish": "es",
    "Bengali": "bn"
}
selected_language = st.selectbox("Choose a language for the caption:", options=list(language_options.keys()))

if uploaded_file is not None:
    st.image(uploaded_file, caption='Uploaded Image', use_container_width=True)

    # Save file temporarily
    with open("temp.jpg", "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Preprocess image
    img = load_img("temp.jpg", target_size=(300, 300))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # Feature extraction
    image_name = os.path.basename(uploaded_file.name)
    if image_name in features_dict:
        feature = features_dict[image_name].reshape((1, -1))
    else:
        from tensorflow.keras.applications import EfficientNetB3
        from tensorflow.keras.models import Model
        base_model = EfficientNetB3(weights='imagenet', include_top=False, pooling='avg')
        feature_extractor = Model(inputs=base_model.input, outputs=base_model.output)
        feature = feature_extractor.predict(img_array)

    # Caption generation
    caption = generate_caption(caption_model, tokenizer, feature, max_length=max_length)
    speech_text = caption

    st.markdown("###  English Caption:")
    st.success(caption)

    # Translation (if needed)
    if selected_language != "English":
        translator = Translator()
        try:
            translation = translator.translate(caption, dest=language_options[selected_language])
            speech_text = translation.text
            st.markdown(f"### Caption in {selected_language}:")
            st.info(speech_text)
        except Exception as e:
            st.error(f"Translation failed: {e}")

    # Text-to-Speech
    try:
        tts = gTTS(text=speech_text, lang=language_options[selected_language])
        audio_path = "caption_audio.mp3"
        tts.save(audio_path)
        st.markdown("###  Hear the Caption:")
        st.audio(audio_path)
    except Exception as e:
        st.error(f"Text-to-speech failed: {e}")
