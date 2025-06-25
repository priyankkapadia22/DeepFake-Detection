Now a days clssifying Deepfake video and original video with naked eye is likely impossible thing and in this AI world,detecting Deepfake video is very important from the media forensic side 
so to solve that I have implemented This prototype project called **"DeepShield"**. This project implements a Convolutional Neural Network (CNN)-based pretrained Models and advanced techniques to make deepfake detection system. 
This project Detects the Video Fake/Real using the trained model.
# Details:-
---
## Steps:
1. Data Preparation
2. Data Augmentation
3. Model Architecture
4. Model Training
5. Model Testing
6. Real-time Prediction

---

## 📌 Features

- ✅ Detects Is video is deepfake or real?

---

## 🗃️ Dataset

- **Dataset**: FaceForensics++ from Kaggle
- **Classes**: `Real`, `Fake`
- **Preprocessing**: Face extraction, resizing, normalization

---

## 🧠 Model Architecture

- Xception Model (CNN based) 
- Layers include:
  - Conv2D + ReLU
  - MaxPooling
  - Dropout
  - Fully Connected (Dense)
  - Bidirectional LSTM
- Optimizer: Adam
- Loss: Categorical Crossentropy with Label smoothing
- Metrics: Accuracy, Precision, Recall, F1-score

---

## 📊 Results

| Metric     | Real Class | Fake Class |
|------------|------------|------------|
| Precision  | 0.89       | 0.92       |
| Recall     | 0.94       | 0.85       |
| F1-Score   | 0.91       | 0.88       |

**Overall Accuracy**: 90%  
**Confusion Matrix**:

|                  | **Predicted: Real** | **Predicted: Fake**  |
|------------------|---------------------|----------------------|
| **Actual: Real** |          31         |           2          |
| **Actual: Fake** |          4          |           23         |
