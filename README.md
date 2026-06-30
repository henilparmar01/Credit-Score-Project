# 💳 Credit Risk Prediction using Machine Learning

A Machine Learning project that predicts whether a customer is likely to default on a loan using historical financial data. This project demonstrates the complete ML workflow, from data preprocessing and feature engineering to model evaluation and handling class imbalance.

---

## 📌 Project Overview

Financial institutions need reliable credit risk prediction systems to identify customers who are likely to default on loans. This project uses the **Give Me Some Credit** dataset to build and compare multiple machine learning models for predicting customer creditworthiness.

The project is divided into **two versions**:

- **Model 1:** Standard Machine Learning Pipeline
- **Model 2:** Improved Model with Class Imbalance Handling

The second model was developed after analyzing the limitations of the first model, where severe class imbalance affected the prediction of default customers.

 📂 Dataset

**Dataset:** Give Me Some Credit

**Target Variable**

`SeriousDlqin2yrs`

- **0 → No Default**
- **1 → Default**


 📊 Dataset Challenges

The dataset contains several real-world challenges:

- Missing values
- Outliers
- Skewed numerical features
- Highly imbalanced target variable
- Feature engineering requirements

Approximately **93%** of customers belong to the **Non-Default** class, while only **7%** belong to the **Default** class.

This makes prediction difficult because machine learning models naturally become biased toward the majority class.


🚀 Model 1 – Standard Credit Risk Prediction

The first implementation focuses on building a complete credit risk prediction pipeline.

## Features

- Data Cleaning
- Missing Value Imputation
- Outlier Treatment
- Feature Engineering
- Feature Scaling
- Model Training
- Model Evaluation
- Feature Importance Analysis


## 🛠 Feature Engineering

The following features were created to improve model performance:

- TotalPastDue
- DebtToIncome
- IncomePerDependent
- CreditLinesPerAge
- HasRealEstateLoan
- UtilizationBucket
- IsRetired


 🤖 Machine Learning Models

- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier


 📈 Evaluation Metrics

The models were evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC Score
- Confusion Matrix


 ⚠ Problem Identified

Although the first model achieved **high accuracy**, further analysis revealed an important issue.

The dataset is highly imbalanced, causing the model to correctly predict the majority class while missing many default customers.

As a result:

- Accuracy was high.
- Recall for the minority class was low.
- F1 Score was low.
- The model struggled to identify actual loan defaulters.

Since detecting default customers is the primary objective of a credit risk model, this issue needed to be addressed.


🚀 Model 2 – Handling Class Imbalance

The second implementation improves the model by specifically addressing class imbalance.

Three complementary techniques were applied.

 ✅ 1. Class Weight Balancing

Used

```python
class_weight="balanced"
```

to assign greater importance to minority-class samples during training.


 ✅ 2. SMOTE (Synthetic Minority Oversampling Technique)

Applied **SMOTE** only to the training dataset to generate synthetic samples for the minority class.

This prevents overfitting while improving the model's ability to recognize default customers.


 ✅ 3. Decision Threshold Tuning

Instead of using the default prediction threshold of **0.50**, the optimal threshold was selected based on the highest **F1 Score**.

Threshold tuning improves the balance between Precision and Recall.


 📊 Comparison of Both Models

| Model | Description |
|--------|-------------|
| Model 1 | Standard ML Pipeline with Feature Engineering |
| Model 2 | Class Weight + SMOTE + Threshold Tuning |

The second model provides significantly better identification of default customers while maintaining strong overall performance.


 📉 Performance Metrics

The following metrics were used for evaluation:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC
- PR-AUC
- Confusion Matrix



# 📷 Output Visualizations

The project generates the following visualizations:

- ROC Curve Comparison
- Random Forest Feature Importance
- Precision-Recall Curve
- F1 Score vs Decision Threshold
- Class Imbalance Comparison


# 📁 Project Structure

```
Credit-Risk-Prediction/
│
├── Model_1.py
├── Model_2.py
├── cs-training.csv
├── cs-test.csv
├── README.md
├── roc_curve_comparison.png
├── feature_importance.png
└── imbalance_handling_comparison.png
```

---

# 💻 Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- Imbalanced-learn (SMOTE)

# 📦 Installation

Clone the repository

```bash
git clone https://github.com/your-henilparmar01/credit-risk-prediction.git
```

Move into the project folder

```bash
cd credit-risk-prediction
```

Install dependencies

```bash
pip install numpy pandas matplotlib scikit-learn imbalanced-learn
```

Run the project

```bash
python Model_1.py
```

or

```bash
python Model_2.py
```

---

# 🎯 Key Learnings

Through this project, I gained practical experience in:

- Data Cleaning
- Feature Engineering
- Handling Missing Values
- Outlier Treatment
- Class Imbalance Handling
- SMOTE
- Threshold Optimization
- Model Evaluation
- Credit Risk Prediction
- Comparing Multiple Machine Learning Models


# 🔮 Future Improvements

- Hyperparameter Optimization
- XGBoost
- LightGBM
- CatBoost
- Cross Validation
- SHAP Explainability
- Model Deployment using Flask/FastAPI
- Docker Deployment


# 👨‍💻 Author

**Henil Parmar**
@henilparmar01

Machine Learning | Data Science | Python

If you found this project helpful, feel free to ⭐ this repository.
