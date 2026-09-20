# Customer Churn Prediction

A machine learning application that estimates customer churn probability from demographic, account, and banking-related features. The repository includes model experimentation, persisted preprocessing artifacts, a trained TensorFlow/Keras model, and a Streamlit interface for interactive inference.

## Overview

The model uses customer attributes including credit score, geography, gender, age, tenure, account balance, number of products, credit-card ownership, active-member status, and estimated salary.

The application returns a churn probability and interprets probabilities above 0.5 as likely churn.

## Prediction Pipeline

User input → categorical encoding → feature scaling → trained Keras model → churn probability → binary prediction.

The preprocessing objects are persisted separately so the inference application can reproduce the transformations used during training.

## Repository Structure

- **app.py** — Streamlit inference application
- **experiments.ipynb** — model experimentation
- **prediction.ipynb** — prediction workflow
- **Churn_Modelling.csv** — dataset
- **model.h5** — trained TensorFlow/Keras model
- **label_encoder_gender.pkl** — gender encoder
- **onehot_encoder_geography.pkl** — geography encoder
- **scaler.pkl** — feature scaler
- **pyproject.toml** — project dependencies

## Running the Application

Install the project dependencies declared in pyproject.toml, then start the Streamlit application:

    streamlit run app.py

Enter customer information in the UI to obtain the model's churn probability.

## Tech Stack

**Python · TensorFlow/Keras · Scikit-learn · Pandas · NumPy · Streamlit**

## Notes

This project demonstrates the complete path from tabular-data preprocessing and neural-network inference to an interactive ML application. The repository does not currently document a reproducible final evaluation score, so no performance number is claimed here.

## Author

Debanjan Sarkar