# ML-Assignment-2
Machine Learning Assignment 2

Developed By: Mangesh Tembhurne
BITS ID: 2026AK05674


a. Problem Statement

The objective of this project is to develop and evaluate multiple machine learning models for predicting the target variable using the given dataset. Different classification/regression algorithms are trained, tested, and compared based on various evaluation metrics. The goal is to identify the best-performing model and analyse its strengths and weaknesses.

b. Dataset Description
Dataset Overview
Dataset Name: Healthcare Stroke Prediction
Source: Kaggle
Number of Records: 5,110
Number of Features: 12
Target Variable: stroke (0 = No Stroke, 1 = Stroke)
Kaggle Link: https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset

Data Preprocessing
Handled missing values.
Removed duplicate records.
Encoded categorical variables.
Feature scaling/normalisation applied where required.
Split data into training and testing sets.

d. Models Used

The following machine learning models were implemented and evaluated:

Logistic Regression
Decision Tree
Random Forest
Support Vector Machine (SVM)
K-Nearest Neighbours (KNN)

Model Evaluation Metrics

| ML Model Name | Accuracy      | AUC           | Precision     | Recall        | F1            | MCC           |
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
|Logistic Regression       | 0.847261  | 0.905644          |  0.739421  | 0.592857  | 0.658077  | 0.566948  |
| Decision Tree  | 0.811843            | 0.748620  | 0.619893  | 0.623214  | 0.621549 |  0.496351  |
| KNN   |0.825567          | 0.853417   | 0.672557  | 0.577679  | 0.621518  |  0.511517  |
|Naive Bayes  |  0.645711       |  0.851614  |0.404946  |  0.913839   | 0.561206  |  0.409813  |
| Support Vector
Machine  | 0.851688         |  0.898492  |  0.766588| 0.577679 |  0.658859 | 0.575739  |




e. How to Run Locally
Clone the repository:
git clone https://github.com/mangesh1234/ML-Assignment-2.git
Verify Python Version (3.12): -python --version (Expected output: Python 3.12.x)

Create and Activate Virtual Environment:

Linux / macOS / BITS Virtual Lab: -python3.12 -m venv venv -source venv/bin/activate

Windows (Command Prompt): -py -3.12 -m venv venv -venv\Scripts\activate

Windows (PowerShell): - py -3.12 -m venv venv -.\venv\Scripts\Activate.ps1

Install Dependencies: -python -m pip install --upgrade pip (Option If in case you encounter any issues/errors related to package installation, pip etc..) -pip install -r requirements.txt

Train Models and Generate Artifacts: -python train_and_export.py

Run the Streamlit Application Locally: -streamlit run app.py

Streamlit Cloud Deployment Guide: -For Streamlit Cloud Deployment refer this link and navigate to Docs for Deployment https://streamlit.io/cloud

f. Conclusion
This project demonstrates the application of multiple machine learning classification algorithms on the Bank Marketing dataset. By comparing Logistic Regression, Decision Tree, KNN, Naive Bayes, and SVM models, the project identifies the most suitable classifier for predicting term deposit subscriptions. The developed Streamlit application provides a practical interface for real-time customer prediction and model demonstration.