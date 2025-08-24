import os
import joblib
import pandas as pd
from flask import Flask, render_template, request

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "model.pkl")

app = Flask(__name__)
model = joblib.load(MODEL_PATH)  # Pipeline with preprocessing + model

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    # Read form fields
    age = request.form.get("age", type=float)
    sex = request.form.get("sex", type=str)
    bmi = request.form.get("bmi", type=float)
    children = request.form.get("children", type=int)
    smoker = request.form.get("smoker", type=str)
    region = request.form.get("region", type=str)

    # Build a DataFrame for model
    input_df = pd.DataFrame([{
        "age": age,
        "sex": sex,
        "bmi": bmi,
        "children": children,
        "smoker": smoker,
        "region": region
    }])

    # Predict
    pred = model.predict(input_df)[0]

    return render_template("predict.html", 
                           prediction=round(float(pred), 2),
                           form_data=input_df.iloc[0].to_dict())

if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)
