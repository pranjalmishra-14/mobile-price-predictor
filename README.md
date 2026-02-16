📱 Mobile Price Range Prediction
Machine Learning Classification Web Application

🔗 Live Application:

https://mobile-price-predictor-1.streamlit.app/

 Overview

This project implements an end-to-end Machine Learning system to classify mobile phones into price categories based on their technical specifications.

The system analyzes features such as RAM, battery power, camera resolution, processor cores, screen resolution, and connectivity support to predict the mobile’s price range.

The trained model is deployed as an interactive web application using Streamlit.

 Problem Statement

To build a supervised machine learning classification model capable of predicting the price category of a mobile phone using its hardware and connectivity features.

Price Categories:

Class 0 → Low Cost

Class 1 → Medium Cost

Class 2 → High Cost

Class 3 → Premium

 Machine Learning Pipeline

The project follows a complete ML workflow:

Data Collection

Data Preprocessing

Feature Scaling (StandardScaler)

Train-Test Split (80/20)

Model Training

Model Evaluation

Model Serialization (Pickle)

Web Deployment

 Model Selection

The following algorithms were evaluated:

Logistic Regression

K-Nearest Neighbors

Decision Tree

Random Forest (Final Model)

 Final Model: Random Forest Classifier

Test Accuracy: 89.25%

Random Forest was selected due to:

High accuracy

Reduced overfitting

Strong feature interpretability

Stability across categories

 Model Evaluation

The system was evaluated using:

Accuracy Score

Confusion Matrix

Feature Importance Analysis

The confusion matrix shows strong diagonal dominance, indicating high class-wise prediction performance with minimal misclassification between adjacent categories.

 Web Application Features

The deployed application provides:

Interactive user input panel

Real-time prediction

Model confidence score

Feature importance visualization

Clean dashboard-style interface

 Technology Stack

Programming Language

Python

Libraries

Pandas

NumPy

Scikit-learn

Matplotlib

Streamlit

Tools

Git

GitHub

Streamlit Cloud

📁 Project Structure
mobile-price-predictor/
│
├── app.py              # Streamlit application
├── train_model.py      # Model training script
├── model.pkl           # Trained Random Forest model
├── scaler.pkl          # StandardScaler object
├── requirements.txt    # Project dependencies
├── train.csv           # Training dataset
└── test.csv            # Testing dataset

Deployment

The application is deployed on Streamlit Community Cloud and can be accessed through the live link above.

To run locally:

pip install -r requirements.txt
streamlit run app.py

 Future Enhancements

Predict exact mobile price using regression

Integrate real-world e-commerce data

Add advanced UI/UX improvements

Implement hyperparameter tuning

Deploy with custom domain

👨‍💻 Author

Pranjal Kumar Mishra
B.Tech – Computer Science & Engineering
