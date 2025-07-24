import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="Image Captioning App", layout="centered")

# Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox(
    "Select Page",
    [
        "About the Project",
        "Image Caption",
        "Image Caption and Segmentation",
        "Image caption and Multi instance segmentation"
    ]
)

# About the Project
if app_mode == "About the Project":
    st.header("Internship project - ZIDIO DEVELOPMENT")
    st.markdown("""
    ###  Image Captioning and Segmentation 

    This project integrates modern computer vision techniques to automatically:
    - **Generate captions** for images in multiple languages
    - **Perform both** simultaneously for a richer understanding

    ####  Tools & Techniques
    - Image Captioning: EfficientNet + CNN-LSTM
    - Image Segmentation: U-Net 
    - Multilingual Translation: Google Translate API
    - Text to speech generation for the caption

    ### Team Members(Group 13)
    - Gursirat Kaur
    - Chaitanya kumar Gunuru

    """)
    st.markdown("---")
    with st.expander("Need help? Chat with our bot!", expanded=False):
        st.markdown("Ask me something about the system:")
        user_input = st.text_input("You:", key="chatbot_input")
        if user_input:
            prompt = user_input.lower()

            if "hello" in prompt or "hi" in prompt:
                st.success(
                    "👋 Hello! I'm here to help you understand image captioning, segmentation, and machine learning concepts.")

            elif "caption" in prompt or "image captioning" in prompt:
                st.success(
                    " Image captioning is the process of generating descriptive sentences for images. We use EfficientNet for feature extraction and a CNN-LSTM model to generate natural language captions.")

            elif "segmentation" in prompt or "image segmentation" in prompt:
                st.success(
                    "Image segmentation is the task of classifying each pixel in an image to identify different objects or regions. We use a U-Net architecture for accurate pixel-wise segmentation.")

            elif "object detection" in prompt:
                st.success(
                    "Object detection involves identifying and locating objects in an image. We use YOLOv5 trained on the COCO dataset to detect real-world objects like people, animals, and vehicles.")

            elif "machine learning" in prompt:
                st.success(
                    "Machine learning is a subset of AI where systems learn from data to make predictions or decisions without being explicitly programmed. It includes supervised, unsupervised, and reinforcement learning.")

            elif "supervised" in prompt:
                st.success(
                    "Supervised learning involves training models on labeled data, such as using images of cats and dogs with labels to classify new images.")

            elif "unsupervised" in prompt:
                st.success(
                    " Unsupervised learning finds patterns in data without labels, such as grouping similar images using clustering techniques like K-Means.")

            elif "cnn" in prompt:
                st.success(
                    "Convolutional Neural Networks (CNNs) are deep learning models especially effective for image tasks like classification, segmentation, and detection.")

            elif "lstm" in prompt:
                st.success(
                    " LSTMs (Long Short-Term Memory) are a type of RNN used for sequence prediction tasks. We use them to generate text descriptions word by word in image captioning.")

            else:
                st.info(
                    " I'm still learning. Try asking about captioning, segmentation, object detection, or ML topics like supervised learning.")


# Image Caption Page
elif app_mode == "Image Caption":
    exec(open("image_caption.py", encoding="utf-8").read())



# Image Caption + Segmentation Page
elif app_mode == "Image Caption and Segmentation":
    exec(open("image_caption_segmentation.py", encoding="utf-8").read())

elif app_mode == "Image caption and Multi instance segmentation":
    exec(open("multinstance.py", encoding="utf-8").read())




