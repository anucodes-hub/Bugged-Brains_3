# test_pipeline_refined.py
import cv2
import numpy as np
import os
from glob import glob
from utils.morphology import classify_morphology, calculate_risk
from utils.ai_audit import ai_audit

def refined_classification(geo_cat, ai_label, size_um, confidence):
    """
    Hybrid classification combining geometry, AI prediction, and size thresholds.
    """
    # Map AI labels to standard categories
    ai_cat_map = {
        "synthetic fiber": "Fiber",
        "plastic fragment": "Fragment",
        "plastic film": "Film",
        "spherical pellet": "Pellet"
    }
    ai_cat = ai_cat_map.get(ai_label.lower(), "Fragment")

    # CASE 1: Small Particles (<500µm)
    if size_um < 500:
        if ai_cat == "Fiber" and confidence > 60:
            return "Fiber"
        return "Fragment"

    # CASE 2: Large Particles (>2000µm)
    if size_um > 2000:
        if geo_cat == "Film" or ai_cat == "Film":
            return "Film"
        if geo_cat == "Pellet" and ai_cat == "Pellet":
            return "Pellet"
        return "Fragment"

    # CASE 3: Mid-range (500µm - 2000µm)
    if geo_cat == ai_cat:
        return geo_cat
    if confidence > 75:
        return ai_cat
    return geo_cat

def test_single_image(image_path, scale=10.0):
    print(f"\n--- Testing Image: {os.path.basename(image_path)} ---")
    
    # 1. Load Image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image at {image_path}")
        return

    # 2. Pre-processing
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # 3. Contour Detection
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(f"Found {len(contours)} potential particles.")

    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        if area < 80: 
            continue  # Skip noise

        # 4. Morphology Analysis
        rect = cv2.minAreaRect(cnt)
        geo_category, base_risk, solidity = classify_morphology(cnt, rect)
        size_um = max(rect[1]) * scale

        # 5. AI Audit
        print(f"  > Particle {i+1}: Running AI Audit...")
        ai_label, ai_score = ai_audit(img, cnt)
        ai_confidence = ai_score * 100  # convert to %

        # 6. Refined Hybrid Classification
        final_label = refined_classification(geo_category, ai_label, size_um, ai_confidence)

        # 7. Risk Calculation
        final_risk = calculate_risk(final_label, size_um)

        # 8. Print Results
        print(f"    [Result] Geometry: {geo_category} | AI: {ai_label} | Final: {final_label}")
        print(f"    [Metrics] Size: {size_um:.1f}µm | Confidence: {ai_confidence:.1f}% | Risk: {final_risk:.1f}/100")

def main():
    sample_dir = "samples"
    image_extensions = ['*.jpg', '*.jpeg', '*.png']
    
    # Find all images in samples/
    files = []
    for ext in image_extensions:
        files.extend(glob(os.path.join(sample_dir, ext)))

    if not files:
        print(f"No images found in '{sample_dir}/'. Please add sample images.")
        return

    print(f"Starting Pipeline Test on {len(files)} files...")
    for f in files:
        test_single_image(f)

if __name__ == "__main__":
    main()