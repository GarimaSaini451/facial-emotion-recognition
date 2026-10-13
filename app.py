import os
import cv2
import numpy as np
import streamlit as st
from PIL import Image
from transformers import pipeline

st.set_page_config(page_title="Facial Emotion Recognition", page_icon="🙂", layout="centered")

MODEL_ID = os.getenv("EMOTION_MODEL_ID", "trpakov/vit-face-expression")
EMOTIONS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

@st.cache_resource(show_spinner="Loading the deep-learning model (first run may take a few minutes)...")
def load_classifier():
    return pipeline("image-classification", model=MODEL_ID, top_k=None)

def detect_faces(pil_image):
    """Return face crops using OpenCV's bundled Haar cascade; fall back to full image."""
    rgb = np.array(pil_image.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(cascade_path)
    faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
    if len(faces) == 0:
        return [pil_image.convert("RGB")], []
    crops, boxes = [], []
    for (x, y, w, h) in sorted(faces, key=lambda f: f[2] * f[3], reverse=True):
        pad_x, pad_y = int(w * 0.12), int(h * 0.12)
        x1, y1 = max(0, x-pad_x), max(0, y-pad_y)
        x2, y2 = min(rgb.shape[1], x+w+pad_x), min(rgb.shape[0], y+h+pad_y)
        crops.append(Image.fromarray(rgb[y1:y2, x1:x2]))
        boxes.append((int(x), int(y), int(w), int(h)))
    return crops, boxes

st.title("🙂 Facial Emotion Recognition")
st.write("Upload a face image to estimate facial-expression categories with a pretrained Vision Transformer (ViT).")
st.info("Educational demo only. Expression predictions are uncertain and must not be treated as a person's true feelings, personality, intent, or mental state.")

uploaded = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png", "webp"])
if uploaded:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Uploaded image", use_container_width=True)
    try:
        crops, boxes = detect_faces(image)
        if boxes:
            st.caption(f"Detected {len(boxes)} face(s). Showing predictions for each face, largest first.")
        else:
            st.warning("No face was detected. The model will try the full image; results may be unreliable.")
        classifier = load_classifier()
        for index, crop in enumerate(crops, start=1):
            if len(crops) > 1:
                st.subheader(f"Face {index}")
            predictions = classifier(crop)
            # Transformers pipelines can return nested lists depending on version.
            if predictions and isinstance(predictions[0], list):
                predictions = predictions[0]
            predictions = sorted(predictions, key=lambda item: item["score"], reverse=True)
            best = predictions[0]
            st.metric("Top predicted expression", best["label"].replace("_", " ").title(),
                      f"{best['score']:.1%} model score")
            st.write("Top predictions")
            chart_data = {item["label"].replace("_", " ").title(): float(item["score"])
                          for item in predictions[:7]}
            st.bar_chart(chart_data)
    except Exception as exc:
        st.error("The model could not be loaded or inference failed.")
        st.exception(exc)
        st.markdown("Check your internet connection and installed packages. The pretrained model downloads on first use.")

with st.expander("How it works"):
    st.markdown("""
    1. OpenCV attempts to detect face regions.
    2. A pretrained Vision Transformer (ViT) classifies each face crop.
    3. The app displays the model's top label and scores.

    The model is downloaded from Hugging Face the first time it is used. This is inference with a pretrained deep-learning model; it does not train a new model on your uploaded image.
    """)
