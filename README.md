# 🏠 House Price Prediction for Urban Cities (NYC)

## 📌 Project Overview
This project builds a machine learning system to predict house prices in New York City using regression techniques.  
The model estimates realistic property prices based on physical and location-related features, similar to systems used by real-estate platforms and banks for valuation and loan approvals.

---

## 🎯 Problem Statement
Predict the **sale price of a house** based on measurable property attributes such as:
- building area
- number of units
- number of floors
- building age
- borough (location)

This is a **regression problem** with a continuous target variable.

---

## 📊 Dataset
**NYC Housing Sales Dataset**

Each row represents a property sale in New York City.

### Target Variable
- `sale_price` — Final sale price of the property

### Features Used
- `lotarea` – Land area
- `bldgarea` – Total building area
- `resarea` – Residential area
- `comarea` – Commercial area
- `unitsres` – Residential units
- `unitstotal` – Total units
- `numfloors` – Number of floors
- `building_age` – Age of the building
- `borough_x` – Encoded borough location

Irrelevant identifier columns were removed to avoid noise.

---

## 🧠 Feature Engineering
- Created non-linear features such as squared building age to capture complex pricing behavior
- Removed leakage-prone and ID-based features
- Applied feature scaling to ensure fair regularization

---

## ⚙️ Models Implemented
- Multiple Linear Regression  
- Polynomial Regression  
- Ridge Regression  
- Lasso Regression  

---

## 📈 Model Evaluation
The models were evaluated using:
- **RMSE (Root Mean Squared Error)**
- **MAE (Mean Absolute Error)**
- **R² Score**
- **5-Fold Cross-Validation**

A baseline model (mean price predictor) was used for comparison.

---

## 🏆 Results
- **Ridge Regression** achieved the lowest and most stable RMSE
- Lasso regression performed feature selection by shrinking weak coefficients to zero
- Polynomial regression captured non-linearity but showed risk of overfitting
- Cross-validation confirmed strong generalization

---

## 📊 Visualizations
- Actual vs Predicted price scatter plots
- RMSE comparison across models
- Feature coefficient (slope) analysis
- Regression trend visualizations

---

## 🛠 Tech Stack
- **Language:** Python  
- **Libraries:** Pandas, NumPy  
- **Visualization:** Matplotlib, Seaborn  
- **Machine Learning:** Scikit-learn  

---

## 🧪 Validation Strategy
- Train-test split (80–20)
- K-Fold Cross-Validation
- Baseline comparison
- Coefficient interpretability check

---

## ✅ Conclusion
This project demonstrates how classical regression models, combined with feature engineering, scaling, and validation, can provide accurate and interpretable house price predictions for urban real-estate data.  
Ridge regression was selected as the final model due to its balance of accuracy, stability, and interpretability.

---

## 📌 Author
Nihal Pandey
