# 💳 Credit Card Fraud Detection App

An interactive machine learning web app to detect fraudulent credit card transactions. This project uses **Logistic Regression**, **Support Vector Machine**, and **Random Forest** to identify suspicious activity. Built with `scikit-learn`, deployed using `Streamlit`.

---

## 🚀 Features

✅ **Multiple Model Support**:  
- Logistic Regression  
- Support Vector Machine (SVM)  
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

```
credit-card-fraud-app/
├── streamlit_app.py       # Streamlit interface code
├── rf_model.pkl           # Trained Random Forest model
├── svm_model.pkl          # Trained SVM model
├── lr_model.pkl           # Trained Logistic Regression model
├── requirements.txt       # Project dependencies
├── .gitignore             # Exclude dataset file
├── README.md              # Project documentation
```


---

## 🧠 Model Training Summary

- Data Preprocessing: Handled nulls, scaling
- Handled Imbalance: SMOTE (oversampling minority class)
- Models Trained: Logistic Regression, SGDClassifier (SVM), Random Forest
- Evaluation Metrics: Accuracy, Precision, Recall, F1-score, ROC-AUC
- Best AUC: Random Forest

---

## 📊 ROC-AUC Scores

| Model                | ROC-AUC Score |
|---------------------|---------------|
| Logistic Regression | 0.95          |
| SVM (SGDClassifier) | 0.94          |
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

git clone https://github.com/Asphane/credit-card-fraud-app.git
cd credit-card-fraud-app

2️⃣ Install Dependencies

Create virtual env (optional but recommended):
python -m venv venv
venv\Scripts\activate  # Windows

Install required packages:
pip install -r requirements.txt

3️⃣ Run the App

streamlit run streamlit_app.py
Open your browser at http://localhost:8501 and use the UI to upload a transaction CSV.

🧪 Sample CSV Format for Prediction

Upload a .csv file with the same structure as the dataset (except Class column):
Time,V1,V2,V3,...,V28,Amount
0,-1.359807,-0.072781,2.536347,...,0.021,149.62

🌐 Want to Deploy on Streamlit Cloud?
-> Push this repo to GitHub

-> Go to: https://streamlit.io/cloud

-> Click "New App" → Choose your repo

-> Select streamlit_app.py as entry point

Done! You now have a shareable public app

🛠 Requirements
Generated using pip freeze > requirements.txt, this project uses:

streamlit
pandas
scikit-learn
matplotlib
seaborn
joblib
numpy

📄 License
This project is licensed under the MIT License.
You are free to use, modify, and distribute.

👩‍💻 Author
Developed with ❤️ by @Asphane

If you like this project, feel free to ⭐ star the repo!
