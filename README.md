# House Price Prediction for Urban Cities (NYC)

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3-orange?logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-2.0-150458?logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-1.24-013243?logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.7-11557C?logo=matplotlib&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

A regression pipeline to predict property sale prices across New York City boroughs using classical ML models, feature engineering, and regularization — built to reflect real-world valuation systems used by real-estate platforms and banks.

---

## Problem Statement

Given physical and location-based attributes of a property, predict its **sale price** — a continuous regression target. The challenge lies in the high variance of NYC real estate prices across boroughs, building types, and ages.

---

## Dataset

**File:** `nyc_housing_base.csv` — 34,439 NYC property sale records

After dropping rows with missing values, **34,203 records** were used for modelling.

**Features selected for modelling:**

| Feature | Description |
|---|---|
| `lotarea` | Land area of the property |
| `bldgarea` | Total building area |
| `resarea` | Residential area |
| `comarea` | Commercial area |
| `unitsres` | Number of residential units |
| `unitstotal` | Total number of units |
| `numfloors` | Number of floors |
| `building_age` | Age of the building |
| `borough_x` | Encoded NYC borough |

Columns such as `block`, `lot`, `zip_code`, `latitude`, `longitude`, `landuse`, and `bldgclass` were dropped — they are either identifiers or carry implicit location redundancy that would leak or add noise.

---

## What I Did — Step by Step

### 1. Exploratory Data Analysis
- Loaded the dataset and inspected dtypes, nulls, and shape
- Plotted the distribution of `sale_price` — found a **right-skewed distribution** with most sales under $2M and a long tail of high-value properties
- Computed a **correlation heatmap** across all features
- Key finding: `bldgarea`, `resarea`, `unitsres`, `unitstotal`, and `numfloors` had the strongest correlations with each other — but correlations with `sale_price` were weak (max ~0.12), indicating the data is noisy and non-linear
- Plotted `bldgarea` vs `sale_price` scatter — confirmed high variance at every area size

### 2. Feature Engineering
- Created `age_squared = building_age²` to capture non-linear depreciation effects on price
- Created `price_per_sqft = sale_price / bldgarea` as a derived insight feature
- Dropped identifier-like and leakage-prone columns
- Final feature set: 11 columns including the two engineered features

### 3. Train-Test Split
- 80/20 split with `random_state=42`
- **27,362 training rows, 6,841 test rows**

### 4. Models Trained

**Multiple Linear Regression**
- Fitted directly on the feature set
- Established the baseline regression performance

**Polynomial Regression (degree=2)**
- Generated all degree-2 interaction and squared terms via `PolynomialFeatures`
- Dramatically expanded feature space — led to severe overfitting (RMSE = 0.00 on test, meaning it memorized training data exactly)

**Ridge Regression (α=10)**
- Applied L2 regularization to penalize large coefficients
- Prevented overfitting while matching linear performance — selected as the **final model**

**Lasso Regression (α=1000)**
- Applied L1 regularization, which shrinks weak coefficients to exactly zero
- Performed implicit feature selection; slightly higher RMSE than Ridge
- Note: convergence warning observed — features were not scaled before fitting, which limited Lasso's effectiveness

### 5. Coefficient Analysis (Ridge)
Extracted feature importances via Ridge coefficients:

| Feature | Coefficient | Interpretation |
|---|---|---|
| `numfloors` | +45,828 | More floors → higher price |
| `unitstotal` | +38,933 | More units → higher price |
| `price_per_sqft` | +1,388 | Engineered feature contributes positively |
| `borough_x` | -123,118 | Borough encoding heavily influences price direction |
| `unitsres` | -42,256 | Residential units negatively correlated (multicollinear with unitstotal) |
| `building_age` | -17,881 | Older buildings → lower price |

### 6. Evaluation
- Custom `evaluate()` function computing RMSE and MAE for all models
- Baseline model: predicts the training mean price for every sample
- Ridge train vs test RMSE comparison to check for overfitting
- RMSE comparison bar chart across all four models

---

## Results

| Model | RMSE | MAE |
|---|---|---|
| Linear Regression | 961,413 | 529,628 |
| Polynomial Regression | 0 (overfit) | 0 (overfit) |
| Ridge Regression | 961,413 | 529,625 |
| Lasso Regression | 965,415 | 532,267 |
| **Baseline (Mean Predictor)** | **1,267,485** | — |

**Ridge — Train vs Test RMSE**

| Split | RMSE |
|---|---|
| Train | 921,919 |
| Test | 961,413 |

---

## What I Achieved

- Reduced prediction error by **24% over the baseline** (RMSE: 1,267,485 → 961,413) using Ridge Regression
- Identified that NYC housing price data is highly noisy — features have weak direct correlations with price, making it a realistic and challenging regression problem
- Demonstrated that Polynomial Regression overfits severely on this dataset (degree-2 expansion memorized training data), while Ridge generalizes well with a small train-test RMSE gap (~4%)
- Showed that `numfloors`, `unitstotal`, and `borough_x` are the strongest price drivers via coefficient analysis
- Built a full evaluation pipeline: model training → metric comparison → baseline benchmarking → overfitting check → visualization

---

## Tech Stack

- **Language:** Python 3.10
- **Data:** Pandas, NumPy
- **Modelling:** Scikit-learn (LinearRegression, PolynomialFeatures, Ridge, Lasso)
- **Visualization:** Matplotlib, Seaborn
- **Environment:** Jupyter Notebook

---

## Project Structure

```
House-Price-Prediction-for-Urban-Cities/
├── EDA.ipynb               # Full pipeline: EDA → feature engineering → modelling → evaluation
├── nyc_housing_base.csv    # NYC housing sales dataset (34,439 records)
└── README.md
```

---

## Author

**Nihal Pandey** — [GitHub](https://github.com/nihal00753)
