# 🏥 Flask-Based Healthcare Hub 🧬

Welcome to your intelligent health companion!  
This **Flask-powered web app** unifies predictive machine learning, medical image analysis, and conversational AI into one seamless **AI Healthcare Assistant** platform.

---

## 🌐 Web App Overview

An all-in-one healthcare solution where AI meets empathy.
Diagnose. Advise. Monitor. Guide.
<img width="1860" height="897" alt="image" src="https://github.com/user-attachments/assets/f22eef1e-531c-4b2c-8403-020521c13dcd" />

---

## 🧩 Problem Statement

In a world where access to instant and reliable health advice is a luxury, many individuals delay care due to uncertainty or lack of tools.  
This project answers a critical question:

> ❓ **"Can AI provide fast, accurate, and human-friendly healthcare assistance from a single unified platform?"**

---

## 🚀 Key Features

| Feature                             | Description                                                                                                                |
|-------------------------------------|----------------------------------------------------------------------------------------------------------------------------|
| 🧑‍⚕️ **AI Health Chatbot**           | Talks like a real doctor. Gives advice, prescriptions, and comfort in natural language.                                    |
| 🧪 **Disease Predictor**           | Supports early prediction of **Diabetes**, **Heart Disease**, **Cancer**, and **Viral Illnesses** using trained ML models. |
| 📊 **24x7 Vital Monitor**          | Calculates BMI and gives instant body fitness feedback from basic metrics (height, weight, age).                           |
| 🥗 **Smart Diet Advisor**          | Recommends personalized food plans based on health goals and conditions.                                                   |
| 🧬  **Symptoms-Based Predictor**   | Recommends personalized food plans based on health goals and conditions.                                                   |
| 📞 **Contact Us Section**          | Let users reach out for support, queries, or follow-up.                                                                    |
| 🧭 **Navigation Bar**              | Fully integrated navbar with **Home**, **Features**, **Chatbot**, **Contact**, and **About Us** pages for smooth browsing. |
| 🔐 **User Authentication**         | Simple and secure login system for personalized access (if enabled).                                                       |


---

## 🦠 Diseases Detected

| Disease                 | Technique Used                                                             |
|-------------------------|----------------------------------------------------------------------------|
| 🩸 **Diabetes**        | Random Forest / Logistic Regression                                        |
| 💓 **Heart Disease**   | Logistic Regression / Decision Tree                                        |
| 🧬 **Cancer**          | SVM with clinical parameters                                               |
| 🦠 **Viral Illness**   | Symptom-based classification model                                         |
| 🧠 **Parkinson’s**     | Random Forest & feature-based classification                               |
| 🩻  **Pneumonia**      | CNN model trained on chest X-ray dataset                                   |
| 🩹  **Skin Cancer**    | CNN model (dermatology images)                                             |
| 🤒 **Symptoms-Based**  | NLP + ML hybrid model for disease prediction with description + treatment  |

---

## 🛠️ Tech Stack

| Layer           | Technology                                               |
|-----------------|----------------------------------------------------------|
| **Frontend**    | HTML, CSS, Bootstrap, JavaScript                         |
| **Backend**     | Flask (Python)                                           |
| **ML Libraries**| scikit-learn, pandas, NumPy, pickle                      |
| **IDE**         | Jupyter Notebook & VS Code                               |
| **Deployment**  | Localhost (Flask) → Render/Heroku/AWS (future-ready)  |

---
## 📂 Project Structure
flask-healthcare-hub/     <br>
│<br>
├── templates/              # HTML Templates for UI Pages <br>
│   ├── index.html          <br>
│   ├── chatbot.html         <br> 
│   ├── cancer.html       <br>
│   ├── monitor.html        <br>
│   ├── About.html         <br>
│   ├── App.html        <br>
│   ├── Contact.html         <br>
│   ├── diabetes.html         <br>
│   ├── Feature.html         <br>
│   ├── Food.html         <br>
│   ├── heart.html         <br>
│   ├── Home.html        <br>
│   ├── Login.html         <br>
│   ├── Monitor.html       <br>
│   ├── parkinson.html       <br>
│   ├── skin_cancer.html     <br>
│   ├── Lungs.html         <br>
│   └── contact.html        <br>
│<br> 
├── static/                 # CSS and JS Files <br> 
│   └── styles.css           <br> 
│   └── styles.js          <br>
│ <br>
├── models/                 # Pre-trained ML Model Files (.pkl) <br>
│   ├── diabetes_model.pkl <br>
│   ├── heart_disease_model.pkl <br>
│   ├── breast_cancer_model <br>
│   ├── parkinsons_cancer_model <br>
│   ├── Random_forest_model <br>
│   └── cancer_model.pkl <br>
│   └── skin_cancer_cnn_model.h5 <br>
│   └── vgg_unfrized.h5 <br>
│ <br>
├── app.py                  # Main Flask Application Script <br>
├── utils.py                <br>
└── requirements.txt        # Python Dependencies <br>


---

## 🔮 Future Enhancements  

To expand the scope and impact of the **Flask-Based Healthcare Hub**, the following improvements are planned:  

- 🧠 **Deep Learning Models**  
  - Brain Tumor Classification using MRI scans with pre-trained architectures like ResNet / VGG16.  

- 📷 **Medical Image Upload & Analysis**  
  Allow users to upload X-rays, MRIs, or skin images for real-time AI-powered diagnosis.  

- 🧬 **Advanced Disease Coverage**  
  Extend ML models to cover more conditions like Liver Disease, Kidney Disease, Pneumonia, Alzheimer’s, etc.  

- 📡 **Cloud Deployment**  
  Deploy on scalable platforms (AWS, Render, Heroku) with integrated APIs for remote access.  

- 📱 **Mobile App Integration**  
  Extend healthcare hub as a cross-platform mobile app (Flutter/React Native) for 24x7 health support.  

- 🗣️ **Multilingual Chatbot**  
  Support for multiple languages like Hindi, Gujarati, etc. to improve accessibility across diverse user groups.  

- 📊 **Health Dashboard**  
  Personalized health history tracking with analytics & visualization of past predictions.  

- 🤖 **AI-Powered Symptom Checker 2.0**  
  Upgrade chatbot with Large Language Models (LLMs) (like GPT) for more accurate medical explanations and empathetic conversations.  


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

