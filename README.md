# CreditWise AI 🏦🤖

An end-to-end Machine Learning web application that predicts whether a loan application is likely to be approved or rejected based on applicant information.

🌐 **Live Demo:** https://creditwise-ai-1.onrender.com/

---

## Features

* ✅ Real-time loan approval prediction
* ✅ Logistic Regression model
* ✅ Data preprocessing pipeline
* ✅ Missing value imputation
* ✅ One-Hot Encoding for categorical variables
* ✅ Feature scaling
* ✅ Probability/confidence score
* ✅ Responsive web interface
* ✅ Flask REST API backend
* ✅ Deployed on Render

---

## Tech Stack

### Machine Learning

* Python
* Pandas
* NumPy
* Scikit-Learn
* Joblib

### Backend (vibecoded)

* Flask

### Frontend (vibecoded)

* HTML
* CSS
* JavaScript

### Deployment 

* Render

---

## Model Performance

| Metric    | Score  |
| --------- | ------ |
| Accuracy  | 87.5%  |
| Precision | 79.03% |
| Recall    | 80.33% |
| F1 Score  | 79.67% |

Trained on 800 samples with 27 engineered features.

---

## Project Structure

```text
CreditWise-AI/
│
├── app.py
├── train_model.py
├── requirements.txt
├── Procfile
│
├── models/
│   ├── model.pkl
│   ├── scaler.pkl
│   ├── ohe.pkl
│   ├── education_encoder.pkl
│   ├── target_encoder.pkl
│   ├── num_imputer.pkl
│   ├── cat_imputer.pkl
│   ├── feature_columns.pkl
│   └── metrics.json
│
├── templates/
│   └── index.html
│
├── static/
│   ├── css/
│   │   └── style.css
│   └── js/
│       └── app.js
│
├── credit_wise.ipynb
└── loan_approval_data.csv
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/grishampambra/CreditWise-AI.git
cd CreditWise-AI
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
python app.py
```

Open:

```
http://127.0.0.1:5000
```

---

## Future Improvements

* SHAP Explainability
* XGBoost Model
* Prediction History Database
* User Authentication
* PDF Report Generation
* Dashboard Analytics
* Docker Support

---

## Author

**Grisham Pambra**

GitHub: https://github.com/grishampambra

---

⭐ If you found this project useful, consider giving it a star!
