import os
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
import joblib

# Paths
HERE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(HERE, "data", "insurance_small.csv")
MODEL_PATH = os.path.join(HERE, "model.pkl")


df = pd.read_csv(DATA_PATH)

FEATURES_CAT = ["sex", "smoker", "region"]
FEATURES_NUM = ["age", "bmi", "children"]
TARGET = "charges"

X = df[FEATURES_NUM + FEATURES_CAT]
y = df[TARGET]

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), FEATURES_CAT),
        ("num", StandardScaler(), FEATURES_NUM),
    ]
)


model = LinearRegression()

pipe = Pipeline(steps=[("preprocess", preprocessor),
                      ("model", model)])


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

pipe.fit(X_train, y_train)


y_pred = pipe.predict(X_test)
r2 = r2_score(y_test, y_pred) if len(y_test) > 0 else float("nan")
mae = mean_absolute_error(y_test, y_pred) if len(y_test) > 0 else float("nan")

print(f"Trained LinearRegression on {len(X_train)} rows.")
print(f"Test R^2: {r2:.4f}")
print(f"Test MAE: {mae:.4f}")


joblib.dump(pipe, MODEL_PATH)
print(f"Saved model to: {MODEL_PATH}")
