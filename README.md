🧠 MNIST Handwritten Digit Classifier

This project implements a convolutional neural network (CNN) using PyTorch to classify handwritten digits from the MNIST dataset. It includes data preprocessing, model training, performance evaluation using R, and visual reporting with Tableau.

📊 Project Overview

Dataset: MNIST

Model: PyTorch CNN

Accuracy: 98.6% on 10,000 test images

Evaluation: Confusion matrix, per-class precision, recall, and F1-score

Tools Used:

PyTorch for model architecture and training

R (caret + tidyverse) for model evaluation

SQLite for storing predictions

Tableau for visualizing results

🛠️ Features

✅ Data augmentation to boost generalization

✅ Hyperparameter tuning

✅ Serialized predictions to SQLite

✅ Exported predictions to CSV

✅ Confusion matrix and per-class metrics in R

✅ Per-class accuracy visualization in Tableau

📈 Performance Metrics (Class '1' example)

Metric

Value

Precision

99.6%

Recall

99.0%

F1 Score

99.3%
