Titanic Survival Prediction

This project is a machine learning-based solution for predicting the survival of passengers on the Titanic. The dataset used contains information about passengers, and the goal is to predict whether a given passenger survived the disaster based on features such as age, gender, ticket class, etc.

Table of Contents

Overview
Project Structure
Dataset
Installation
Usage
Model
Results
Contributing
License
Overview

The Titanic Survival Prediction project aims to predict which passengers survived the sinking of the Titanic using a machine learning model trained on the famous Titanic dataset. The project involves data cleaning, feature engineering, and training different models to identify the best approach for making accurate predictions.

Key steps:

Data preprocessing (handling missing values, encoding categorical variables)
Feature selection and engineering
Training machine learning models (e.g., Logistic Regression, Random Forest)
Model evaluation using accuracy, precision, recall, and F1-score
Project Structure


Dataset

The dataset used in this project is the Titanic dataset from Kaggle. It contains the following key columns:

Survived: 0 = No, 1 = Yes (target variable)

Pclass: Passenger’s class (1 = 1st, 2 = 2nd, 3 = 3rd)

Name, Sex, Age, SibSp, Parch, Fare: Various details about the passenger

Embarked: Port of Embarkation (C = Cherbourg, Q = Queenstown, S = Southampton)

Usage

1. Data Preprocessing
To preprocess the data and prepare it for model training:
python src/data_preprocessing.py --input data/train.csv --output data/processed_data.csv

2. Train the Model
Train the Titanic survival prediction model:
python src/train_model.py --input data/processed_data.csv --model_output models/titanic_model.pkl

3. Predict
Make predictions on new data using the trained model:
python src/predict.py --model models/titanic_model.pkl --input data/test.csv

Model

Several machine learning algorithms are explored in this project, including:

Logistic Regression
Random Forest
Support Vector Machines (SVM)
K-Nearest Neighbors (KNN)
The model is evaluated based on performance metrics such as accuracy, precision, recall, and F1-score. Cross-validation is also performed to ensure the robustness of the model.

Results

The performance of the model was evaluated on the test set using the following metrics:

Algorithm	Accuracy	Precision	Recall	F1-score
Logistic Regression	80.3%	79.5%	71.2%	75.1%
Random Forest	82.1%	81.0%	73.5%	77.1%
The Random Forest algorithm provided the best results in this project.

Contributing

Contributions are welcome! If you have any improvements or bug fixes, feel free to open an issue or submit a pull request.

