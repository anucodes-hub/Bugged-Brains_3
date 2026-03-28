import cv2

from PIL import Image

from transformers import pipeline

import streamlit as st



@st.cache_resource

def load_ai_model():

    """

    Loads the CLIP model for zero-shot classification.

    Cached to prevent reloading on every streamlit rerun.

    """

    return pipeline("zero-shot-image-classification", model="openai/clip-vit-base-patch32")



def ai_audit(img, contour):

    """

    Crops a particle from the image and classifies it using AI.

    Returns: label (str), confidence_score (float)

    """

    classifier = load_ai_model()

   

    # Get bounding box for cropping

    x, y, w, h = cv2.boundingRect(contour)

   

    # Add padding for AI context, ensuring we stay within image boundaries

    pad = 10

    img_h, img_w = img.shape[:2]

    y1, y2 = max(0, y-pad), min(img_h, y+h+pad)

    x1, x2 = max(0, x-pad), min(img_w, x+w+pad)

   

    crop = img[y1:y2, x1:x2]

   

    if crop.size == 0:
        return "Unknown", 0.0



    # Convert BGR to RGB for PIL

    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    crop_pil = Image.fromarray(crop_rgb)

   

    labels = ["synthetic fiber", "plastic fragment", "plastic film", "spherical pellet"]

   

    results = classifier(crop_pil, candidate_labels=labels)

   

    return results[0]['label'], results[0]['score']