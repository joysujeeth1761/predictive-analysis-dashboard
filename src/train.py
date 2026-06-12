import pandas as pd

from preprocess import remove_leakage

# Load dataset
df = pd.read_csv('/Users/joysujeeth/Downloads/student_performance.csv')


# Apply preprocessing
df_clean = remove_leakage(df)
from feature_engineering import create_features

df_clean = create_features(df_clean)
X = df_clean.drop("G3", axis=1)
y = df_clean["G3"]

num_cols = X.select_dtypes(include="number").columns

cat_cols = X.select_dtypes(exclude="number").columns

x_encoded = pd.get_dummies(X,columns=cat_cols,drop_first=True)

from sklearn.model_selection import train_test_split
X_train, X_test , y_train , y_test = train_test_split(x_encoded,y,test_size = 0.2,random_state = 40)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
# build model_1
from sklearn.linear_model import LinearRegression

lr_model = LinearRegression()

lr_model.fit(X_train_scaled, y_train)
# make predictions
y_pred = lr_model.predict(X_test_scaled)

#evaluate the model_1
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

mae = mean_absolute_error(
    y_test,
    y_pred
)

rmse = mean_squared_error(
    y_test,
    y_pred
) ** 0.5

r2 = r2_score(
    y_test,
    y_pred
)

"""print("Linear Regression Results")
print(f"MAE:  {mae}")
print(f"RMSE: {rmse}")
print(f"R2:   {r2}")"""

from sklearn.ensemble import RandomForestRegressor

rf_model = RandomForestRegressor(
    n_estimators=500,
    max_depth=8,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42
)

rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)

rf_mae = mean_absolute_error(y_test, rf_pred)
rf_rmse = mean_squared_error(y_test, rf_pred) ** 0.5
rf_r2 = r2_score(y_test, rf_pred)

"""print("\nRandom Forest Results")
print(f"MAE:  {rf_mae}")
print(f"RMSE: {rf_rmse}")
print(f"R2:   {rf_r2}")"""

from xgboost import XGBRegressor

xgb_model = XGBRegressor(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

xgb_model.fit(X_train, y_train)

xgb_pred = xgb_model.predict(X_test)

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

xgb_mae = mean_absolute_error(
    y_test,
    xgb_pred
)

xgb_rmse = (
    mean_squared_error(
        y_test,
        xgb_pred
    ) ** 0.5
)

xgb_r2 = r2_score(
    y_test,
    xgb_pred
)

"""print("\nXGBOOST RESULTS ")

print("MAE :", xgb_mae)
print("RMSE:", xgb_rmse)
print("R2  :", xgb_r2) """

import pandas as pd

results_df = pd.DataFrame({
    "Model": [
        "Linear Regression",
        "Random Forest",
        "XGBoost"
    ],
    "MAE": [
        mae,
        rf_mae,
        xgb_mae
    ],
    "RMSE": [
        rmse,
        rf_rmse,
        xgb_rmse
    ],
    "R2": [
        r2,
        rf_r2,
        xgb_r2
    ]
})

"""print(results_df) """

""" import matplotlib.pyplot as plt
plt.figure(figsize=(8,5))

plt.bar(
    results_df["Model"],
    results_df["R2"]
)

plt.ylabel("R² Score")
plt.title("Model Comparison")

plt.show() """

# saving the model
from pathlib import Path
import joblib

BASE_DIR = Path(__file__).resolve().parent.parent

MODELS_DIR = BASE_DIR / "models"

MODELS_DIR.mkdir(exist_ok=True)

# Save Model

joblib.dump(
    lr_model,
    MODELS_DIR / "student_performance_model.pkl"
)

joblib.dump(
    scaler,
    MODELS_DIR / "scaler.pkl"
)

joblib.dump(
    x_encoded.columns.tolist(),
    MODELS_DIR / "feature_names.pkl"
)

print("Models saved successfully!")