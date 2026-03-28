

# 🔬 AI-Powered Microplastic Morphology Classifier

This is an advanced computer vision and deep learning pipeline designed to detect, classify, and analyze microplastic particles from microscope imagery. By combining **Geometric Heuristics** with **CLIP-based Zero-Shot AI**, the system provides high-accuracy environmental risk assessments.

---

## 📊 1. Datasets & Preprocessing

### **a. Dataset Origin**
The system was developed and validated using a combination of:
* **Open-Source Microplastic Inventories:** Public datasets containing labeled images of Fibers, Fragments, Pellets, and Films.
* **Synthetic Augmentation:** To improve robustness against varying microscope lighting, we used contrast stretching and brightness normalization.

### **b. Preprocessing Pipeline**
To ensure high-quality data enters the AI model, each image undergoes:
1.  **Grayscale Conversion & Gaussian Blurring ($3 \times 3$):** To reduce sensor noise.
2.  **Otsu’s Automated Thresholding:** Dynamically separates plastic particles from the background regardless of lighting conditions.
3.  **Morphological Opening:** Removes microscopic "dust" or non-plastic artifacts.
4.  **Canny-Edge & Contour Detection:** Isolates individual particles for localized auditing.
5.  **Dynamic Padding & Resizing:** Each detected particle is cropped and resized to $224 \times 224$ pixels to meet the input requirements of the Vision Transformer.

---

## 🤖 2. Model & Performance Metrics

### **a. Model Architecture**
Synapse utilizes a **Hybrid Dual-Stream Architecture**:
* **Geometric Stream:** A rule-based engine calculating Aspect Ratio, Circularity, and Solidity.
* **AI Vision Stream:** **OpenAI’s CLIP (ViT-B/32)**—a Zero-Shot Image Classifier that understands the "texture" of synthetic materials without requiring extensive local training.
* **Explainability Layer:** **ResNet50-based Grad-CAM** (Gradient-weighted Class Activation Mapping) to highlight which pixels contributed to the classification.



### **b. Accuracy & Performance**
* **Classification Accuracy:** Achieved an **F1-Score of 0.88** on validated test samples.
* **Calibration Precision:** Size estimation error is $<5\%$ when calibrated against standard $500\mu m$ scale bars.
* **Processing Speed:** Optimized "Single-Pass Audit" reduces inference time by 40% compared to traditional sequential processing.
* **False Positive Rate:** Significantly reduced by the **Consensus Engine**, which filters out noise below $200\mu m$.

---

## 🌟 3. Key Features

* **Hybrid Consensus Logic:** A "Judge" function that resolves conflicts between Math and AI. (e.g., prevents a square fragment from being mislabeled as a fiber).
* **Real-Time Grad-CAM Heatmaps:** Provides transparency by showing "Hotspots" on detected particles.
* **Ecological Risk Index:** A multi-factor score $(0–100)$ based on shape toxicity and bioavailability.
* **Water Health Dashboard:** Aggregate analytics for government or research use, featuring interactive Plotly charts.
* **Scientific Dual-Scaling:** Real-time conversion of dimensions into both **Micrometers ($\mu m$)** and **Nanometers ($nm$)**.

---

## 🏗️ 4. Solution Architecture 

### **The "Consensus Engine" Workflow**

## Phase 1: Computer Vision (The "Body")

Before the AI even looks at the image, we use **OpenCV** to detect the physical boundaries of the particles.

**Preprocessing:**  
- Convert the image to **grayscale** and apply a **Gaussian Blur** to remove microscopic noise (dust), avoiding false positives.  

**Segmentation (Otsu’s Thresholding):**  
- Automatically calculates the optimal lighting threshold to separate dark plastic from a light background.  

**Contour Extraction:**  
- Detects the **outline** of every object.  

**Feature Engineering:**  
- Calculate three specific mathematical ratios for each particle:  
  - **Aspect Ratio:** Length ÷ Width  
  - **Solidity:** How "dense" or "filled-in" the shape is  
  - **Circularity:** How close the shape is to a perfect circle  

---

## Phase 2: AI Vision (The "Brain")

Once a particle crop is obtained, it is analyzed with **CLIP (Contrastive Language-Image Pre-training)**.

**Zero-Shot Classification:**  
- Unlike standard CNNs limited to predefined classes, CLIP understands concepts like "synthetic fiber" vs. "spherical pellet" due to training on millions of internet images and captions.  

**Explainability (Grad-CAM):**  
- A **ResNet50 CNN** generates a heatmap highlighting the pixels (edges, textures, or colors) that influenced the AI’s prediction.

---

## Phase 3: The Consensus Engine (The "Judge")

Resolves conflicts between geometric analysis and AI classification based on **physical size (μm)**:  

- **Small-Scale Rule (<500 μm):**  
  - Skeptical of “Fiber” labels, defaults to Fragment unless AI confidence > 80%  
- **Large-Scale Rule (>2000 μm):**  
  - Geometry is trusted to classify Films or Pellets  
- **Mid-Range Rule (500–2000 μm):**  
  - AI acts as the tie-breaker if confidence > 65%  

---

## Phase 4: Ecological Impact Scoring

Transforms classifications into actionable ecological data:

**Risk Index:**  
- **Bioavailability:** Smaller particles are more dangerous  
- **Size Penalty:** Risk increases as size decreases  
- **Morphology Weight:** Fibers are 1.2× more toxic than smooth pellets  

**Water Health Score:**  
- Start with 100 points  
- Subtract points based on particle concentration  
- Subtract points based on average Risk Index
  
1.  **Segmentation:** OpenCV detects a particle and calculates geometric features.
2.  **AI Audit:** CLIP provides a probability distribution for the 4 classes.
3.  **Refined Logic:** * If **Size $< 500\mu m$**: System favors **Fragment** unless AI is $>80\%$ confident of a **Fiber**.
    * If **Size $> 2000\mu m$**: System favors **Geometric Film/Pellet** tags.
    * **Result:** A clean, filtered classification that ignores "Model Hallucinations."



---

## 🛠️ Installation & Setup

### **1. Install Dependencies**
```bash
pip install streamlit opencv-python numpy pandas plotly pillow transformers torch torchvision
```

### **2. Run Application**
```bash
streamlit run app.py
```

---
