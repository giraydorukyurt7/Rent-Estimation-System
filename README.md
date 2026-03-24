# Rent Estimation System

This project focuses on predicting house rent prices in Türkiye using machine learning and ensemble regression methods.

The system combines extensive feature engineering, preprocessing, model comparison, and ensemble techniques to improve prediction quality on a public Turkish housing dataset.

## Overview

The project was built on the **Turkish House Rent Prediction** dataset and aims to estimate rental prices from structured housing attributes such as city, district, neighborhood, apartment type, house type, age, size, room information, floor level, furnishing status, bathroom count, heating type, heating fuel, and price. :contentReference[oaicite:7]{index=7}

A major part of the work focused on transforming noisy real-world tabular data into a model-ready representation through data cleaning, filtering, encoding, and feature redesign. After preprocessing, the final feature matrix reached **6,624 samples and 178 features**. :contentReference[oaicite:8]{index=8}

## Data Preparation

The original train and test datasets were first merged for analysis, producing an **8,808 × 16** matrix before preprocessing. After cleaning and feature engineering, the data was split again using an **80/20 train-test split**, resulting in **5,299 training** and **1,325 test** samples. :contentReference[oaicite:9]{index=9}

Key preprocessing steps included:

- splitting room information into separate room and living room counts
- converting house age and house size to numeric values
- filtering rare and extreme values
- engineering location features for city / district / neighborhood
- handling categorical variables with one-hot encoding
- reducing noisy or low-frequency heating and housing-type categories
- converting selected binary-like fields into model-friendly form

## Models

The project evaluates 7 regression algorithms:

- KNN
- SVR
- Decision Tree
- Random Forest
- GBM
- XGBoost
- LightGBM

All models were tuned with **GridSearchCV** and evaluated using **RMSE** and **MAPE**. :contentReference[oaicite:10]{index=10}

## Ensemble Methods

To go beyond single-model performance, two ensemble strategies were applied:

- **Averaging**
- **Stacking**

For stacking, base model predictions were used as inputs to a **Linear Regression** meta-model. The project evaluated **all combinations of 3 or more models**, resulting in **198 ensemble configurations** across the two ensemble types. :contentReference[oaicite:11]{index=11}

## Results

### Best single model
- **XGBoost**
- **RMSE:** 4744.08
- **MAPE:** 0.2039 :contentReference[oaicite:12]{index=12}

### Best stacking ensemble
- **DecisionTree + RandomForest + XGBoost**
- **RMSE:** 4657.24
- **MAPE:** 0.1994 :contentReference[oaicite:13]{index=13}

### Best averaging ensemble shown in the report
- **RandomForest + GBM + XGBoost**
- **RMSE:** 4660.31
- **MAPE:** 0.2031 :contentReference[oaicite:14]{index=14}

The results showed that both ensemble approaches reduced error compared with weaker single models, and stacking performed better than averaging in the best reported configurations. :contentReference[oaicite:15]{index=15}

## Technologies Used

- Python
- pandas
- NumPy
- scikit-learn
- XGBoost
- LightGBM
- Matplotlib / visualization tools

## Key Takeaways

This project is best viewed as a **feature engineering + ensemble regression study** on real-world tabular housing data.

Its main strengths are:

- strong preprocessing over noisy real-world data
- structured categorical feature engineering
- broad model comparison
- systematic ensemble search across many combinations
- measurable gains from stacking and averaging
