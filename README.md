# 🌿 Plant Disease Detection Using Deep Learning

A deep learning-based plant disease detection system that classifies plant leaf images into 38 disease and healthy classes using a MobileNetV2-based image classification model trained on the PlantVillage dataset.

The project includes model training, evaluation, confusion-matrix analysis, and a Streamlit web application for real-time image-based prediction.

---

## 📌 Project Overview

Plant diseases can significantly affect crop productivity and quality. Early identification can help farmers and agricultural professionals take timely action.

This project uses transfer learning with **MobileNetV2** to classify plant leaf images into **38 PlantVillage classes**.

The system provides:

- Plant disease classification from leaf images
- Confidence score for predictions
- Top 3 predicted classes
- Recommended action for the detected condition
- Interactive Streamlit web interface
- Model evaluation using a separate test set
- Classification report and confusion matrix

---

## 🧠 Model

**Architecture:** MobileNetV2  
**Learning Approach:** Transfer Learning  
**Image Size:** 224 × 224  
**Number of Classes:** 38  
**Framework:** TensorFlow / Keras

The MobileNetV2 feature extractor is initialized with ImageNet weights and used with a custom classification head for PlantVillage disease classification.

---

## 📊 Dataset

The project uses the **PlantVillage** dataset through Hugging Face.

Dataset:

`geraldmc/plantvillage-full`

Dataset characteristics:

- 54,304 images
- 38 classes
- RGB leaf images
- 43,356 training images
- 10,948 test images

The test set was kept separate from the training process and used for final evaluation.

---

## 📈 Results

### Final Test Performance

| Metric | Result |
|---|---:|
| Test Accuracy | **92.52%** |
| Test Images | **10,948** |
| Correct Predictions | **10,129** |
| Incorrect Predictions | **819** |
| Macro F1-Score | **0.9075** |
| Weighted F1-Score | **0.9241** |

### Validation Performance

The trained model achieved a validation accuracy of:

**93.25%**

---

## 🔍 Evaluation

The project includes:

- Classification report
- Precision
- Recall
- F1-score
- Confusion matrix

Generated evaluation files:

```text
models/classification_report.txt
models/confusion_matrix.png