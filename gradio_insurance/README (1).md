# 🏥 Medical Insurance Cost Predictor (Gradio + Hugging Face)

An interactive web app to **predict medical insurance charges** using a regression model (scikit-learn).  
Deployed with **Gradio** and ready to run on **Hugging Face Spaces** 🚀  

---
## 🌟 Live 

Try the app instantly: [Medical Insurance Cost Predictor](https://huggingface.co/spaces/GokulV/insurance-predictor)
## 📂 Project Structure
```text
insurance_predictor/
├── app.py              # Gradio app entrypoint
├── train.py            # trains regression model
├── model.pkl           # created after running train.py
├── requirements.txt    # dependencies for Hugging Face
├── data/
│   └── insurance_small.csv   # sample dataset
```

---

## ⚙️ 1) Setup Locally
```bash
git clone https://github.com/Gokul-bit165/Gradio-insurance-predictor
cd Gradio-insurance-predictor

# (optional) create venv
python -m venv .venv
.venv\Scripts\activate   # Windows
# source .venv/bin/activate  # Mac/Linux

pip install -r requirements.txt
```

---

## 🏋️ 2) Train the Model
```bash
python train.py
```
- This generates `model.pkl`.  
- The pipeline includes:
  - **OneHotEncoder** for categorical features (`sex`, `smoker`, `region`)  
  - **StandardScaler** for numeric features (`age`, `bmi`, `children`)  

---

## 🌐 3) Run the Gradio App Locally
```bash
python app.py
```
- Runs on `http://127.0.0.1:7860`  
- Gradio also provides a **shareable public link**  

---

## 🚀 4) Deploy on Hugging Face Spaces
1. Go to [Hugging Face Spaces](https://huggingface.co/spaces) → **Create Space**  
2. Choose:
   - **SDK:** `Gradio`  
   - **Space name:** `insurance-predictor`  
   - **Public** (free)  
3. Upload:
   - `app.py`  
   - `model.pkl`  
   - `requirements.txt`  
4. Hugging Face auto-builds → Your app will be live at:
   ```
   https://huggingface.co/spaces/GokulV/insurance-predictor
   ```

---

## 📦 requirements.txt
```text
gradio
pandas
joblib
scikit-learn==1.2.2
```

---

## ✨ Features
- Clean regression pipeline with preprocessing  
- Interactive Gradio UI  
- Free, shareable deployment on Hugging Face Spaces  

---

## 📸 Example (UI Screenshot)
   ![alt text](<gradio_insurance/ui.png>)
   ![Medical Insurance Cost Predictor UI](gradio_insurance/ui.png)

