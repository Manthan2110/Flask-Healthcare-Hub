# 🏥 Flask-Based Healthcare Hub 🧬

Welcome to your intelligent health companion!  
This **Flask-powered web app** brings together predictive machine learning, conversational AI, and real-time health tools — all in one seamless medical assistant platform.

---

## 🌐 Web App Overview

An all-in-one healthcare solution where AI meets empathy. Diagnose. Advise. Monitor. Guide.
<img width="1860" height="897" alt="image" src="https://github.com/user-attachments/assets/f22eef1e-531c-4b2c-8403-020521c13dcd" />

---

## 🧩 Problem Statement

In a world where access to instant and reliable health advice is a luxury, many individuals delay care due to uncertainty or lack of tools.  
This project answers a critical question:

> ❓ **"Can AI provide fast, accurate, and human-friendly healthcare assistance from a single unified platform?"**

---

## 🚀 Key Features

| Feature                     | Description                                                                                                                |
|----------------------------|----------------------------------------------------------------------------------------------------------------------------|
| 🧑‍⚕️ **AI Health Chatbot** | Talks like a real doctor. Gives advice, prescriptions, and comfort in natural language.                                    |
| 🧪 **Disease Predictor**    | Supports early prediction of **Diabetes**, **Heart Disease**, **Cancer**, and **Viral Illnesses** using trained ML models. |
| 📊 **24x7 Vital Monitor**   | Calculates BMI and gives instant body fitness feedback from basic metrics (height, weight, age).                           |
| 🥗 **Smart Diet Advisor**   | Recommends personalized food plans based on health goals and conditions.                                                   |
| 📞 **Contact Us Section**   | Let users reach out for support, queries, or follow-up.                                                                    |
| 🧭 **Navigation Bar**       | Fully integrated navbar with **Home**, **Features**, **Chatbot**, **Contact**, and **About Us** pages for smooth browsing. |
| 🔐 **User Authentication**  | Simple and secure login system for personalized access (if enabled).                                                       |

---

## 🦠 Diseases Detected

| Disease            | Technique Used                      |
|--------------------|-------------------------------------|
| 🩸 **Diabetes**     | Random Forest / Logistic Regression |
| 💓 **Heart Disease**| Logistic Regression / Decision Tree |
| 🧬 **Cancer**       | SVM with clinical parameters        |
| 🦠 **Viral Illness**| Symptom-based classification model  |

---

## 🛠️ Tech Stack

| Layer           | Technology                                               |
|-----------------|----------------------------------------------------------|
| **Frontend**    | HTML, CSS, Bootstrap, JavaScript                         |
| **Backend**     | Flask (Python)                                           |
| **ML Libraries**| scikit-learn, pandas, NumPy, pickle                      |
| **IDE**         | Jupyter Notebook & VS Code                               |
| **Deployment**  | Localhost (Flask) or [can be deployed to Render/Heroku]  |

---
## 📂 Project Structure
flask-healthcare-hub/
│
├── templates/              # HTML Templates for UI Pages
│   ├── index.html          # Homepage
│   ├── chatbot.html        # AI Health Chatbot Page
│   ├── cancer.html      # Disease Prediction Interface
│   ├── monitor.html        # Vital Monitor (BMI, Age, etc.)
│   ├── About.html        # Smart Diet Advisor Page
│   ├── App.html        # Smart Diet Advisor Page
│   ├── Contact.html        # Smart Diet Advisor Page
│   ├── diabetes.html        # Smart Diet Advisor Page
│   ├── Feature.html        # Smart Diet Advisor Page
│   ├── Food.html        # Smart Diet Advisor Page
│   ├── heart.html        # Smart Diet Advisor Page
│   ├── Home.html        # Smart Diet Advisor Page
│   ├── Login.html        # Smart Diet Advisor Page
│   ├── Monitor.html        # Smart Diet Advisor Page
│   ├── parkinson.html        # Smart Diet Advisor Page
│   └── contact.html        # Contact Form Page
│
├── static/                 # CSS and JS Files
│   └── styles.css          # Custom Styles
│   └── styles.js          # JS Styles
│
├── models/                 # Pre-trained ML Model Files (.pkl)
│   ├── diabetes_model.pkl
│   ├── heart_disease_model.pkl
│   ├── breast_cancer_model
│   ├── parkinsons_cancer_model
│   ├── Random_forest_model
│   └── cancer_model.pkl
│
├── app.py                  # Main Flask Application Script
├── utils.py                # Utility Functions (e.g., preprocessing, model loading)
└── requirements.txt        # Python Dependencies


---

## ⚙️ How to Run the App

1. **Clone the Repo**

```bash
git clone https://github.com/your-username/flask-healthcare-hub
cd flask-healthcare-hub
```

2. **Install the Required Packages**

```bash
pip install -r requirements.txt
```

3. **Run the Flask Server**

```bash
python app.py
```

4. **Access the Web App**

Open your browser and navigate to: http://localhost:5000

----

## 💡 Sample Chatbot Conversation

👤: "I have a cold and mild headache. What should I do?"  
🤖: "It might be a viral infection. Take rest, drink warm fluids, and consult a doctor if fever persists."

---

## 🧠 Machine Learning Models

- Trained using **Random Forest**, **SVM**, and **Logistic Regression**
- Feature selection and normalization applied
- Outputs binary classification (0 = No Disease, 1 = Disease)
- 🧪 Built using:
    - Evaluation Metrics: Accuracy, Precision, Recall
    - train_test_split
    - Standard Scaling (StandardScaler)
    - 
---

## 🛠️ Tech Stack

- **Frontend:** Streamlit, Streamlit Option Menu
- **Backend:** Python, scikit-learn, joblib, pickle
- **IDE:** Jupyter Notebook 
- **UI Styling:** Custom CSS with markdown embedding

---

---

## 📧 Contact

- 📌 Developed by: Manthan Jadav
- 📫 [LinkedIn](https://www.linkedin.com/in/manthanjadav/)
- ✉️ [Email](mailto:manthanjadav746@gmail.com)

---

## 📢 License

Free to use, improve, or deploy. 
This tool is for educational and assistive use only. Not a substitute for professional medical advice.

---

