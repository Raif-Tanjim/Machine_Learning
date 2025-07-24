# 🔌 EVSE Hybrid Transformer Classifier

This project implements a deep learning-based approach to detect Denial of Service (DoS) attacks in Electric Vehicle Supply Equipment (EVSE) systems using event log data. We build and evaluate a hybrid Transformer + MLP classifier that processes time-series kernel event features to predict attack presence.

---

## 📍 What This Project Does

- Loads and preprocesses event log features from the EVSE-B-HPC-Kernel-Events dataset
- Removes leakage-prone columns and scales features using MinMaxScaler
- Calculates class distribution and computes class weights dynamically
- Builds a hybrid model: Transformer encoder + MLP classifier using PyTorch
- Trains the model with 5-fold stratified cross-validation and early stopping
- Logs training/validation metrics per fold
- Evaluates the model with confusion matrix, ROC, PR curves, and F1/Accuracy plots
- Visualizes per-fold and average metric scores (Accuracy, Precision, Recall, F1)

---

## 🧠 Model Overview

We use a Transformer encoder to process temporal kernel event features, followed by a fully connected MLP for binary classification.

- Input shape: [batch, sequence_length, num_features]
- Transformer with multi-head attention
- Global average pooling across sequence
- Final MLP with dropout and ReLU
- Optimizer: Adam with learning rate = 1e-4
- Weighted binary cross-entropy loss for class imbalance
- Early stopping based on validation loss

---

## 📊 Performance Summary

After 5-fold stratified cross-validation:

| Metric     | Average ± Std Dev |
|------------|-------------------|
| Accuracy   | 99.03% ± 0.16%     |
| Precision  | 93.71% ± ~         |
| Recall     | 99.39% ± ~         |
| F1-score   | 96.4% ± ~          |

Best validation fold:  
✅ **Validation Loss = 0.0415**, Accuracy = 0.9903, Precision = 0.9371, Recall = 0.9939

---

## 📈 Visualizations Included

The notebook includes the following visualizations:

### 🔹 Training Process

- Train vs Validation Loss plot  
- Train vs Validation Accuracy plot  
- Train vs Validation F1-score plot  

### 🔹 Final Model Evaluation

- Confusion Matrix of best fold
- ROC Curve (AUC)
- Precision-Recall (PR) Curve

### 🔹 Cross-Validation Summary

- Bar plots: Accuracy, Precision, Recall, F1-score across 5 folds
- Bar plot: Mean ± Std Dev for all 4 metrics

Each plot includes proper labels, value annotations, legends, and consistent color schemes for better interpretability.

---

## 📁 Folder Structure

EVSE-Hybrid-Transformer-Classifier/
│
├── EVSE_Transformer_Final.ipynb ← Main training notebook
├── best_evse_model.pth ← Saved model after best validation fold
├── README.md ← Project documentation
└── dataset/
└── EVSE-B-HPC-Kernel-Events-processed.csv


##  Install dependencies via pip:
pip install -r requirements.txt

## Notes & Considerations
The dataset is imbalanced (~86% normal vs ~14% DoS); class weights are used to address this.

Early stopping ensures we don’t overfit small folds.

We removed features that directly leaked labels to ensure proper generalization.

The transformer handles sequence learning while the MLP handles classification.

Feel free to experiment with hidden sizes, heads, or number of transformer layers.
