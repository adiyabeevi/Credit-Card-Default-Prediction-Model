# Credit Card Default Prediction

## 📌 Objective
To develop a machine learning model to predict whether a customer will default on their credit card payment next month, using the UCI Credit Card Default dataset.

## 📊 Dataset
- **Source**: UCI Machine Learning Repository  
- **Size**: 30,000 records × 24 features  
- **Target Variable**: `default` (1 = default, 0 = no default)

## 🧪 Steps Performed
1. **Data Loading and Cleaning**
   - Converted XLS to CSV
   - Handled missing values
   - Treated outliers (IQR & Z-score)

2. **EDA & Visualizations**
   - Histograms, Boxplots, Heatmaps, KDE, and Countplots

3. **Feature Engineering**
   - OneHotEncoding for categorical features

4. **Balancing & Scaling**
   - SMOTE for class imbalance
   - StandardScaler for feature scaling

5. **Model Building**
   - Tested 8 classification models
   - Best: Random Forest with hyperparameter tuning using GridSearchCV

6. **Evaluation**
   - Accuracy: ~80.8%
   - Precision (class 1): 60%
   - F1 Score: 48%

7. **Model Saving**
   - Final model saved as `random_forest_credit_default.pkl`

## 🔍 Tools & Tech
- Python (pandas, matplotlib, seaborn, scikit-learn)
- Google Colab
- Joblib (for saving models)

## 📁 Files Included
- `credit_card_default.csv` – Cleaned dataset
- `credit_default_model.ipynb` – Full working notebook
- `random_forest_credit_default.pkl` – Saved ML model
- `README.md` – Project description

## ✅ Prediction Example
You can test new customer data using the saved model:

```python
import joblib
model = joblib.load("random_forest_credit_default.pkl")
sample = [[50000, 35, 1, 1, 0, 0, 0, 0, 0, 0, ...]]  # input features
print("Prediction:", "Default" if model.predict(sample)[0] == 1 else "No Default")


