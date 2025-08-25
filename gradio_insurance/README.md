# Medical Insurance Flask Regression (Starter)

A tiny, step-by-step starter to train a regression model (scikit-learn) and serve predictions via Flask on localhost.

## Project Structure
```text
insurance_flask/
├── app.py
├── train.py
├── model.pkl               # created after running train.py
├── requirements.txt
├── data/
│   └── insurance_small.csv # sample data (5 rows)
└── templates/
    └── index.html
```

## 1) Setup Python env
```bash
cd insurance_flask
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Mac/Linux:
# source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

## 2) Train the model
```bash
python train.py
```
This creates `model.pkl`.

## 3) Run the Flask app (localhost:5000)
```bash
python app.py
```
Open http://127.0.0.1:5000

## Notes
- The pipeline includes OneHotEncoder for categorical features and StandardScaler for numeric ones.
- For better accuracy, replace `data/insurance_small.csv` with a larger dataset (same columns), then re-run `train.py`.
