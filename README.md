# 📊 Student Performance Prediction using Linear Regression

This project implements a simple **linear regression** model using **gradient descent** to predict a **Performance Index** based on various student features such as study hours, previous scores, sleep, and more. The training process includes feature normalization and evaluates the model using the **coefficient of determination (R²)**.

---

## 📁 Dataset Structure

- `Train_Data.csv`: Training dataset with labeled student features and performance index.
- `Test_Data.csv`: (Currently unused) can be adapted for evaluation.

---

## 📈 Features Used

- Hours Studied
- Previous Scores
- Extracurricular Activities (converted to binary: Yes → 1, No → 0)
- Sleep Hours
- Sample Question Papers Practiced

---

## ⚙️ How It Works

1. **Normalization** of input features to bring values between 0 and 1.
2. **Gradient Descent** is used to minimize the mean squared error (MSE) cost function.
3. The model is trained over 1000 iterations with a learning rate of 0.1.
4. After training, it computes:
   - Final cost values
   - **R² Score** to evaluate model performance
5. **Visualization** of cost vs iterations using matplotlib.

---

## 📌 Output

- A **cost plot** showing how the loss decreases over iterations.
- **R² Score** printed in console to show the model's predictive strength.

Example:
