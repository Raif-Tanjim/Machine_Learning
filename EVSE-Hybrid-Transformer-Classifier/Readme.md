# 🔌 EVSE Hybrid Transformer Classifier

This project implements a deep learning-based approach to detect Denial of Service (DoS) attacks in Electric Vehicle Supply Equipment (EVSE) systems using event log data. We build and evaluate a hybrid Transformer + MLP classifier that processes time-series kernel event features to predict attack presence.

---
🚗 What's This All About?
In smart mobility, protecting Electric Vehicle Supply Equipment (EVSE) from cyberattacks like DoS is critical. This project presents a Hybrid Transformer + MLP classifier that leverages time-series kernel event logs to detect such attacks with 99.98% accuracy via 5-fold cross-validation.
## 📍 What This Project Does
---

- Loads and preprocesses event log features from the EVSE-B-HPC-Kernel-Events dataset
- Removes leakage-prone columns and scales features using MinMaxScaler
- Calculates class distribution and computes class weights dynamically
- Builds a hybrid model: Transformer encoder + MLP classifier using PyTorch
- Trains the model with 5-fold stratified cross-validation and early stopping
- Logs training/validation metrics per fold
- Evaluates the model with confusion matrix, ROC, PR curves, and F1/Accuracy plots
- Visualizes per-fold and average metric scores (Accuracy, Precision, Recall, F1)

---

🛠️ Key Features
✅ Transformer Encoder to capture temporal dependencies
✅ MLP Classifier to enhance non-linear decision boundaries
✅ Imbalanced data handling via dynamic class weights
✅ Robust evaluation with stratified CV and early stopping
✅ Beautiful visualizations: Confusion Matrix, ROC, PR, Accuracy/F1 trends
✅ Plug-and-play code using PyTorch and Sklearn

🧠 Model Architecture

[Event Log Sequence] → [Transformer Encoder] → [Global Pooling] → [MLP Classifier] → [Binary Output]
Sequence input: [batch, time_steps, features]

Transformer: Multi-head attention, positional encoding

MLP: Dense layers + ReLU + Dropout

Loss: Weighted BCE

Optimizer: Adam (lr=1e-4)

CV: 5-fold Stratified + Early Stopping



📈 Final Performance (Cross-Validation)
Metric	Mean	Std Dev
Accuracy	99.98%	± 0.03%
Precision	100.0%	± 0.00%
Recall	99.88%	± 0.23%
F1-score	99.94%	± 0.12%
Best validation fold:  
✅ **Validation Loss = 0.0415**, Accuracy = 0.9903, Precision = 0.9371, Recall = 0.9939
💯 Final test set also shows perfect detection:
Accuracy=1.0 | Precision=1.0 | Recall=1.0 | F1=1.0
---

## 📈 Visualizations Included

The notebook includes the following visualizations:

🔄 Training Dynamics: Loss, Accuracy, F1-score curves

🔍 Model Evaluation: Confusion Matrix, ROC-AUC, PR Curve

🧪 Cross-Fold Summary: Metric bars with Std Dev error bars

All plots come with clean annotations, legends, and consistent color schemes.



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

⚙️ Quick Start
git clone https://github.com/Raif-Tanjim/Machine_Learning.git
cd Machine_Learning/EVSE-Hybrid-Transformer-Classifier
pip install -r requirements.txt
