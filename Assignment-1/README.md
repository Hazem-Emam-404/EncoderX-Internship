# Intel Image Classification CNN

**EncoderX Remote Internship - Batch 02 | Track: AI/ML | Week 01 Task**

## 📌 Project Overview
This repository contains a Convolutional Neural Network (CNN) built with PyTorch to classify images into six distinct categories. This project was developed as part of the EncoderX AI/ML Internship to gain practical experience in computer vision, neural networks, and model evaluation.

## 📂 Dataset
**Intel Image Classification Dataset**
*   **Total Images:** 16,900
*   **Classes (6):** Buildings, Forest, Glacier, Mountain, Sea, Street
*   **Split:** 70% Training (11,830 samples), 15% Validation (2,535 samples), 15% Testing (2,535 samples)

## ⚙️ Data Preprocessing & Augmentation
To ensure optimal model performance and prevent overfitting, the following preprocessing pipeline was applied:
*   **Standardization:** Resized all images to `224x224` pixels.
*   **Normalization:** Applied standard ImageNet channel means `[0.485, 0.456, 0.406]` and standard deviations `[0.229, 0.224, 0.225]`.
*   **Augmentation (Training Set Only):**
    *   `RandomHorizontalFlip()`
    *   `RandomRotation(15 degrees)`

## 🧠 Model Architecture
A custom, modular Convolutional Neural Network was built from scratch using `torch.nn.Module`:
1.  **Conv Block 1:** Conv2d (32 filters) + ReLU + MaxPool2d
2.  **Conv Block 2:** Conv2d (64 filters) + ReLU + MaxPool2d
3.  **Conv Block 3:** Conv2d (128 filters) + ReLU + MaxPool2d
4.  **Classifier Head:** Flatten → Linear (512) → ReLU → Dropout (p=0.5) → Linear (6 classes)

## 🚀 Training Process
*   **Framework:** PyTorch (GPU accelerated via Kaggle)
*   **Loss Function:** Cross-Entropy Loss
*   **Optimizer:** Adam (`lr=0.001`)
*   **Epochs:** 10
*   **Tracking:** Real-time epoch monitoring using `tqdm`. The loop dynamically saves the best model state based on the lowest validation loss.

## 📊 Evaluation & Results
The model was rigorously evaluated on the strictly isolated 15% test set (2,535 images).
*   **Overall Accuracy:** **86%**
*   **Macro Average F1-Score:** **0.86**

### Performance Highlights:
*   **Top Performer (`forest`):** Achieved the highest metrics (Precision: 0.97, Recall: 0.94, F1-score: 0.96). The distinct color palette and textures made this class highly distinguishable.
*   **Lowest Performer (`buildings`):** Yielded the lowest precision (0.79) and F1-score (0.83). 
*   **Key Misclassifications:** The confusion matrix revealed that the model occasionally struggled with overlapping visual features, heavily confusing natural topographies (Glacier vs. Mountain) and structural urban scenes (Street vs. Buildings).

