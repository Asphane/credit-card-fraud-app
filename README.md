# 💳 Credit Card Fraud Detection App

An interactive machine learning web app to detect fraudulent credit card transactions. This project uses **Logistic Regression**, **SGDClassifier (Linear Model)**, and **Random Forest** to identify suspicious activity. Built with `scikit-learn`, deployed using `Streamlit`.

---

## 🚀 Features

✅ **Multiple Model Support**:  
- Logistic Regression  
- SGDClassifier (Linear SVM-like model)  
- Random Forest Classifier  

📈 **Visualizations**:  
- Class Distribution  
- Confusion Matrix  
- ROC-AUC Curve

🧪 **Prediction Interface**:  
- Upload a sample `.csv` file with transactions  
- Choose ML model for prediction  
- Get real-time fraud **probability (%)**  
- See results instantly with interpretation

---

## 📁 Project Structure

credit-card-fraud-app/
├── streamlit_app.py       # Streamlit interface code
├── rf_model.pkl           # Trained Random Forest model
├── sdgc_model.pkl         # Trained SGDClassifier model
├── lr_model.pkl           # Trained Logistic Regression model
├── requirements.txt       # Project dependencies
├── .gitignore             # Exclude dataset file
├── README.md              # Project documentation

---

## 🧠 Model Training Summary

- Data Preprocessing: Handled nulls, scaling
- Handled Imbalance: SMOTE (oversampling minority class)
- Models Trained: Logistic Regression, SGDClassifier, Random Forest
- Evaluation Metrics: Accuracy, Precision, Recall, F1-score, ROC-AUC
- Best AUC: Random Forest

---

## 📊 ROC-AUC Scores

| Model                | ROC-AUC Score |
|---------------------|---------------|
| Logistic Regression | 0.95          |
| SGDClassifier       | 0.94          |
| Random Forest       | 0.98 ✅        |

---

## 📂 Dataset

The dataset used is the publicly available [Kaggle Credit Card Fraud Detection dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud) — contains anonymized features (`V1` to `V28`) and a binary target `Class`.

📥 **Dataset Download Link** (Google Drive):  
[👉 Download creditcard.csv](https://drive.google.com/file/d/11w-g1eStJxtxGCC5iy7op0K9vE_A8hUV/view?usp=sharing)

> Note: File is not included in this repo due to GitHub's 100MB file size limit.

---

## ▶️ How to Run the Streamlit App Locally

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/Asphane/credit-card-fraud-app.git
cd credit-card-fraud-app
```

### 2️⃣ Install Dependencies

Create virtual environment (optional but recommended):

```bash
python -m venv venv
venv\Scripts\activate  # On Windows
```

Install all required packages:

```bash
pip install -r requirements.txt
```

### 3️⃣ Run the App

```bash
streamlit run streamlit_app.py
```

App will open in your browser at `http://localhost:8501`

---

## 🧪 Sample CSV Format for Prediction

Make sure your input `.csv` has the same columns as the original dataset, excluding the `Class` column:

```csv
Time,V1,V2,V3,...,V28,Amount
0,-1.359807,-0.072781,2.536347,...,0.021,149.62
```

---

## 🌐 Want to Deploy on Streamlit Cloud?

1. Push this repo to GitHub  
2. Go to [https://streamlit.io/cloud](https://streamlit.io/cloud)  
3. Click **"New App"** → Select this GitHub repo  
4. Set `streamlit_app.py` as the entry point  
5. Done! Your app is live

---

## 🛠 Requirements

Generated using `pip freeze`, this project uses:

streamlit
pandas
scikit-learn
matplotlib
seaborn
joblib
numpy


---

## 📄 License

This project is licensed under the MIT License.  
You are free to use, modify, and distribute.

---

## 👩‍💻 Author

Developed with ❤️ by [@Asphane](https://github.com/Asphane)

If you like this project, feel free to ⭐ star the repo!

---
