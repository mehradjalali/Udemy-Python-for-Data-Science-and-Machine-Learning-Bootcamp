# Lending Club Loan Default Prediction – Udemy Bootcamp Project

This repository contains my solution for one of the neural network projects in the  
**[Python for Data Science and Machine Learning Bootcamp](https://www.udemy.com/course/python-for-data-science-and-machine-learning-bootcamp/)** on Udemy by Jose Portilla.

The project focuses on building a deep learning model using **Keras**/**TensorFlow** to predict whether a Lending Club loan will be fully paid back or charged off, given customer and loan application data.

---

## 📌 Project Overview
- **Goal:** Predict the binary target `loan_status` (Paid Off = 0, Defaulted/Charged Off = 1) based on a mix of numeric and categorical lending features.
- **Dataset:** Lending Club loan dataset (`lending_club_loan_two.csv`) with 396,030 entries and 27 columns.
- **Features:**  
  - Numeric: e.g., `loan_amnt`, `int_rate`, `installment`, `annual_inc`, `dti`, `mort_acc`  
  - Categorical: e.g., `term`, `grade`, `home_ownership`, `verification_status`, `purpose`
- **Source:** Provided as part of the Udemy course materials.

---

## ⚙ Data Preprocessing Highlights
The notebook performs:
1. **Exploratory Data Analysis (EDA)** with `pandas`, `matplotlib`, and `seaborn`.
2. **Missing value handling** for features like `mort_acc` and `pub_rec_bankruptcies`.
3. **Feature transformations**:
   - Converting categorical variables to dummy/indicator variables with `pd.get_dummies`.
   - Mapping ordinal features like `sub_grade` to numeric ranks.
4. **Train/test split** using `train_test_split`.
5. **Feature scaling** with `MinMaxScaler`.

---

## 🧠 Neural Network Architecture (Keras Sequential)
| Layer Type | Units | Activation | Notes |
|------------|-------|------------|-------|
| Dense | 78 | ReLU | Input layer |
| Dropout | 0.2 | — | Regularization |
| Dense | 32 | ReLU | Hidden layer |
| Dropout | 0.2 | — | Regularization |
| Dense | 32 | ReLU | Hidden layer |
| Dropout | 0.2 | — | Regularization |
| Dense | 1  | Sigmoid | Output layer for binary classification |

- **Loss function:** `binary_crossentropy`
- **Optimizer:** `adam`
- **Metrics:** Accuracy

---

## 🚀 Training & Evaluation
- **Epochs:** Configurable; tested over multiple epochs for convergence.
- **Batch size:** Default Keras setting (can be tuned).
- **Output:** Probability between 0 and 1, mapped to classes:
  ```python
  y_pred = (model.predict(X_test) > 0.5).astype(int)
