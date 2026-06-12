# 🎓 Student Academic Performance Predictor

A Machine Learning web application that predicts a student's final academic grade based on demographic, academic, and lifestyle factors.

## 🚀 Live Demo

[Streamlit App](https://predictive-analysis-dashboard-7fwt38llz6biezhum5uvgm.streamlit.app/)

## 📌 Project Overview

This project uses Machine Learning techniques to predict the final grade (G3) of students using features such as:

- Study time
- Past failures
- Parents' education
- Travel time
- Family relationships
- Alcohol consumption
- Internet access
- Higher education aspirations
- Romantic relationships
- School information
- And more...

The application provides an interactive web interface built using Streamlit.

---

## 📂 Project Structure

```text
predictive-analysis-dashboard/
│
├── app/
│   └── streamlit_app.py
│
├── src/
│   ├── preprocess.py
│   ├── feature_engineering.py
│   └── train.py
│
├── models/
│   ├── student_performance_model.pkl
│   ├── scaler.pkl
│   └── feature_names.pkl
│
├── notebooks/
│   └── eda.ipynb
│
├── results/
│
├── requirements.txt
└── README.md
```

---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-Learn
- XGBoost
- Streamlit
- Joblib
- Matplotlib

---

## 🔍 Machine Learning Pipeline

### 1. Data Cleaning
- Removed target leakage features (`G1`, `G2`)
- Checked missing values
- Processed categorical variables

### 2. Feature Engineering
Created new features:

- `parent_edu`
- `total_alcohol`
- `study_failure_ratio`

### 3. Encoding
- One-Hot Encoding for categorical features

### 4. Feature Scaling
- StandardScaler for Linear Regression

### 5. Model Training
Compared:

- Linear Regression
- Random Forest Regressor
- XGBoost Regressor

---

## 📊 Model Performance

| Model | MAE | RMSE | R² Score |
|---------|---------|---------|---------|
| Linear Regression | 2.16 | 3.03 | 0.339 |
| Random Forest | 2.15 | 3.06 | 0.326 |
| XGBoost | 2.29 | 3.22 | 0.253 |

### Best Model
**Linear Regression**

```text
R² Score = 0.339
RMSE = 3.03
MAE = 2.16
```

---

## 🎯 Features of the Web App

- Interactive Streamlit UI
- Real-time grade prediction
- User-friendly sliders and dropdowns
- Automatic feature engineering
- Model inference using saved artifacts

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/joysujeeth1761/predictive-analysis-dashboard.git
cd predictive-analysis-dashboard
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app/streamlit_app.py
```

---

## 📈 Future Improvements

- Hyperparameter tuning
- Feature importance dashboard
- Advanced ensemble models
- Better UI/UX
- Explainable AI (SHAP values)
- More accurate prediction models

---

## 👨‍💻 Author

**Joy Sujeeth**

Mathematics and Computing  
Indian Institute of Technology Delhi (IIT Delhi)

GitHub: https://github.com/joysujeeth1761
