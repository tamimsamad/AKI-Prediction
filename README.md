Machine Learning for Early Prediction of Acute Kidney Injury

Machine learning project for early prediction of acute kidney injury (AKI) using clinical data.

Project Overview

This project investigates whether machine-learning models can identify patients at increased risk of acute kidney injury using clinical measurements available early during hospitalization.

The project uses a public anonymized clinical dataset and compares two classification models:

Logistic Regression
Random Forest

The models are evaluated using metrics useful for imbalanced medical classification, including ROC-AUC, Average Precision, precision, recall, F1-score, and confusion matrices.

Dataset

The dataset contains hospital encounter records with demographic information, vital signs, laboratory measurements, and an AKI outcome label.

For this project, 21 clinical and demographic variables were selected.

The target variable is:

0 = No AKI
1 = AKI

The dataset is highly imbalanced. AKI cases represent approximately 2.1% of the selected records.

Results

Using an 80/20 stratified train-test split:

Metric	Logistic Regression	Random Forest
ROC-AUC	0.671	0.621
Average Precision	0.040	0.029
AKI Recall	0.634	0.012
AKI Precision	0.035	0.014
AKI F1-score	0.065	0.013

The results show that accuracy alone is not sufficient for evaluating this dataset because the AKI outcome is highly imbalanced.

At the default classification threshold, Logistic Regression identified substantially more AKI cases than Random Forest, while Random Forest achieved high overall accuracy but detected very few AKI cases.

Feature Importance

For the Random Forest model, the features with the highest model-based importance included:

Age
Pulse
Systolic blood pressure
BMI
Diastolic blood pressure

Feature importance describes how the model used the variables for prediction. It should not be interpreted as evidence that a feature causes AKI.

Limitations
This is an educational research prototype and has not been clinically validated.
The dataset is highly imbalanced.
Model performance depends on the selected prediction window, features, preprocessing, and classification threshold.
The results should not be used for clinical diagnosis or treatment decisions.
Feature importance does not establish causal relationships.
Project Structure
AKI-Prediction/
├── data/
├── models/
│   └── precision_recall_curve.png
├── src/
│   └── train_model.py
├── .gitignore
├── README.md
└── requirements.txt

The dataset in data/ is excluded from Git tracking.

Disclaimer

This project is intended for educational and research purposes only. It is not a clinically validated prediction system and should not be used to make medical decisions.