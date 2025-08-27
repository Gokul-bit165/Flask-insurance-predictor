import gradio as gr
import joblib
import pandas as pd
import os


HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "model.pkl")
model = joblib.load(MODEL_PATH)  


def predict(age, sex, bmi, children, smoker, region):
    input_df = pd.DataFrame([{
        "age": age,
        "sex": sex,
        "bmi": bmi,
        "children": children,
        "smoker": smoker,
        "region": region
    }])
    pred = model.predict(input_df)[0]
    return f"Predicted Charges: ${round(float(pred), 2)}"


demo = gr.Interface(
    fn=predict,
    inputs=[
        gr.Number(label="Age"),
        gr.Radio(["male", "female"], label="Sex"),
        gr.Number(label="BMI"),
        gr.Number(label="Children"),
        gr.Radio(["yes", "no"], label="Smoker"),
        gr.Dropdown(["northeast", "northwest", "southeast", "southwest"], label="Region")
    ],
    outputs="text",
    title="Insurance Cost Predictor",
    description="Enter details to predict medical insurance charges"
)

if __name__ == "__main__":
    demo.launch()
