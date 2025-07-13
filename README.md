## 🧾 Project Details

- **Project Name**: Credit Card Default Prediction  
- **Author**: Adiya Beevi S  
- **Submission Date**: 13-July-2025  
- **Institute**: Entri Elevate – Data Science & Machine Learning  
- **LinkedIn**: https://www.linkedin.com/in/adiya-salim 
- **GitHub**: https://github.com/adiyabeevi



## 📌 Objective
To develop a machine learning model to predict whether a customer will default on their credit card payment next month, using the UCI Credit Card Default dataset.

---

## 📊 Dataset
- **Source**: [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients)
- **Total Records**: 30,000  
- **Features**: 24  
- **Target**: `default`  
   - `1` = Defaulted  
   - `0` = No Default

---

📦 [Download Trained Model (.pkl)](https://drive.google.com/your_link_here)


## 🧪 Steps Performed

### 1. Data Loading and Cleaning
- Converted `.xls` to `.csv`
- Handled missing values using median imputation
- Treated outliers using **IQR** and **Z-score** (selective)

### 2. EDA & Visualizations
- Plots used: `Histogram`, `Boxplot`, `Heatmap`, `KDE`, `Countplot`, `Pairplot`, `Pie chart`

### 3. Feature Engineering
- OneHotEncoding for categorical features (`SEX`, `EDUCATION`, `MARRIAGE`)

### 4. Balancing & Scaling
- Applied **SMOTE** to handle class imbalance
- Applied **StandardScaler** for feature normalization

### 5. Model Building
Tested multiple classification models:
- Logistic Regression  
- Decision Tree  
- Random Forest  
- SVM  
- KNN  
- Naive Bayes  
- MLP Classifier  
- AdaBoost  
- Gradient Boost

> ✅ **Best Model**: Random Forest  
> 🎯 **Hyperparameter Tuning**: GridSearchCV  

### 6. Evaluation Metrics
- **Accuracy**: ~80.8%  
- **Precision (for default class)**: 60%  
- **F1 Score**: 48%  
- **Conclusion**: Balanced model with good performance on minority class

### 7. Model Saving
Model saved as `random_forest_credit_default.pkl` using `joblib`

---

## 🧰 Tools & Technologies
- Python
- pandas, numpy
- seaborn, matplotlib
- scikit-learn
- imbalanced-learn
- joblib
- Google Colab

---

## 📁 Files Included

| File                          | Description                            |
|------------------------------|----------------------------------------|
| `credit_card_default.csv`    | Cleaned dataset                        |
| `credit_default_model.ipynb` | Complete working notebook              |
| `random_forest_credit_default.pkl` | Saved ML model                  |
| `README.md`                  | Project documentation                  |
| `requirements.txt`           | Project dependencies                   |

---

## ✅ Prediction Example

```python
import joblib

# Load model
model = joblib.load("random_forest_credit_default.pkl")

# Sample input (dummy data)
sample = [[50000, 35, 1, 1, 0, 0, 0, 0, 0, 0, ...]]  # Replace ... with all features

# Predict
result = model.predict(sample)[0]
print("Prediction:", "Default" if result == 1 else "No Default")


