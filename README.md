# 🏡 Predict House Prices Using Advanced Regression  

This project aims to build an accurate regression model to predict housing prices using advanced machine learning techniques.  
The workflow includes data cleaning, feature engineering, exploratory data analysis (EDA), model training, evaluation, and Kaggle submission.

## 📁 **Project Structure**

project/
│── train.csv
│── test.csv
│── notebook.ipynb
└── README.md

## 🔍 **1. Exploratory Data Analysis (EDA)**

During EDA, we focused on understanding the structure of the dataset and identifying key factors correlated with **SalePrice**.

**Performed analyses:**
- Overview of dataset, missing value inspection  
- Correlation heatmap  
- Distribution plots for numerical features (e.g., SalePrice, LotArea, GrLivArea)  
- Countplots for categorical variables (e.g., Neighborhood, HouseStyle)  
- Scatterplots for most important predictors (e.g., OverallQual vs SalePrice)

**Insights:**
- `OverallQual`, `GrLivArea`, and `GarageArea` show strong positive correlation with SalePrice  
- Some categorical variables have strong predictive power (e.g., Neighborhood)  
- Several features contain many missing values and require careful imputation  

---

## 🧹 **2. Data Cleaning & Preparation**

Applied several preprocessing steps:

### ✔ Missing Value Handling
- Mode imputation for: `MSZoning`, `KitchenQual`, `Utilities`  
- Zero filling for: `MasVnrArea`, `BsmtFinSF1`, `GarageArea`, etc.  
- "None" label for categorical NA: `GarageType`, `FireplaceQu`, `Fence`, etc.  
- Group-wise median fill for `LotFrontage`  
- Remaining numerical NAs filled with 0

### ✔ Feature Engineering
- Created dummy variables for all categorical columns using `pd.get_dummies()`  
- Removed unnecessary columns (`Id`, `SalePrice` before encoding)  
- Combined train + test for consistent encoding  

---

## 🔧 **3. Modeling**

Dataset split:
- Training: first 1460 rows  
- Test (for final Kaggle submission): remaining rows  
- Train-validation split using `train_test_split`  

### Trained Models:
- Linear Regression  
- Decision Tree Regressor  
- Random Forest Regressor  
- Gradient Boosting Regressor  
- XGBoost Regressor  

### Evaluation Metrics:
- **MAE (Mean Absolute Error)**  
- **RMSE (Root Mean Squared Error)**  
- **R² Score**

### Model Performance Summary

| Model                | MAE        | RMSE        | R²     |
|----------------------|------------|-------------|--------|
| Linear Regression    | 23943      | 83112       | 0.099  |
| Decision Tree        | 26985      | 40995       | 0.781  |
| Random Forest        | 17500      | 28398       | 0.894  |
| Gradient Boosting    | 17743      | 28929       | 0.898  |
| **XGBoost (BEST)**   | **17435**  | **27887**   | **0.8986** |

**XGBoost achieved the lowest RMSE and was chosen for final prediction.**

---

## 🏆 **4. Kaggle Submission**

Final model: **XGBoost Regressor**

### Steps:
1. Train on full training dataset  
2. Predict values for Kaggle’s test data  
3. Create submission file:
```python
submission = pd.DataFrame({
    "Id": df2["Id"],
    "SalePrice": test_preds
})
submission.to_csv("submission_xgb.csv", index=False)
```

## 🚀 Future Improvements

- **Hyperparameter Tuning**
- **Handle Skewed Features**
- **Feature Selection**
- **Model Ensembling**
- **Regularized Models**
