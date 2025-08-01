from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import numpy as np
import pandas as pd
import joblib
import pickle
from datetime import datetime

app = Flask(__name__)
CORS(app)

# Load models
diabetes_model = pickle.load(open('models/diabetes_model.pkl', 'rb'))
heart_disease_model = pickle.load(open('models/heart_disease_model.pkl', 'rb'))
parkinsons_model = pickle.load(open('models/parkinsons_model.pkl', 'rb'))
breast_cancer_model = pickle.load(open('models/breast_cancer_model.pkl', 'rb'))
model = joblib.load("models/Random_forest_model.pkl")

# Disease mapping and symptom list for infectious disease
symptom_list = ['itching', 'skin_rash', 'nodal_skin_eruptions', 'continuous_sneezing',
                'shivering', 'chills', 'stomach_pain', 'acidity', 'ulcers_on_tongue',
                'muscle_wasting', 'vomiting', 'burning_micturition', 'spotting_urination',
                'fatigue', 'weight_gain', 'anxiety', 'cold_hands_and_feets']

disease_mapping = {0: 'Vertigo', 1: 'AIDS', 2: 'Acne', 3: 'Alcoholic hepatitis',
                   4: 'Allergy', 5: 'Arthritis', 6: 'Bronchial Asthma', 7: 'Cervical spondylosis',
                   8: 'Chicken pox', 9: 'Chronic cholestasis', 10: 'Common Cold', 11: 'Dengue',
                   12: 'Diabetes', 13: 'Piles', 14: 'Drug Reaction', 15: 'Fungal infection',
                   16: 'GERD', 17: 'Gastroenteritis', 18: 'Heart attack', 19: 'Hepatitis B',
                   20: 'Hepatitis C', 21: 'Hepatitis D', 22: 'Hepatitis E', 23: 'Hypertension',
                   24: 'Hyperthyroidism', 25: 'Hypoglycemia', 26: 'Hypothyroidism', 27: 'Impetigo',
                   28: 'Jaundice', 29: 'Malaria', 30: 'Migraine', 31: 'Osteoarthritis',
                   32: 'Paralysis', 33: 'Peptic ulcer', 34: 'Pneumonia', 35: 'Psoriasis',
                   36: 'Tuberculosis', 37: 'Typhoid', 38: 'UTI', 39: 'Varicose veins',
                   40: 'Hepatitis A'}

@app.route('/')
def index():
    return render_template('App.html')


@app.route('/infectious', methods=['GET', 'POST'])
def infectious_disease():
    symptoms = sorted([
        'itching', 'skin_rash', 'nodal_skin_eruptions', 'continuous_sneezing',
        'shivering', 'chills', 'joint_pain', 'stomach_pain', 'acidity', 'ulcers_on_tongue',
        'muscle_wasting', 'vomiting', 'burning_micturition', 'spotting_urination',
        'fatigue', 'weight_gain', 'anxiety', 'cold_hands_and_feets', 'mood_swings',
        'weight_loss', 'restlessness', 'lethargy', 'patches_in_throat', 'irregular_sugar_level',
        'cough', 'high_fever', 'sunken_eyes', 'breathlessness', 'sweating', 'dehydration',
        'indigestion', 'headache', 'yellowish_skin', 'dark_urine', 'nausea', 'loss_of_appetite',
        'pain_behind_the_eyes', 'back_pain', 'constipation', 'abdominal_pain',
        'diarrhoea', 'mild_fever', 'yellow_urine', 'yellowing_of_eyes', 'acute_liver_failure',
        'fluid_overload', 'swelling_of_stomach', 'swelled_lymph_nodes', 'malaise',
        'blurred_and_distorted_vision', 'phlegm', 'throat_irritation', 'redness_of_eyes',
        'sinus_pressure', 'runny_nose', 'congestion', 'chest_pain', 'weakness_in_limbs',
        'fast_heart_rate', 'pain_during_bowel_movements', 'pain_in_anal_region',
        'bloody_stool', 'irritation_in_anus', 'neck_pain', 'dizziness', 'cramps',
        'bruising', 'obesity', 'swollen_legs', 'swollen_blood_vessels', 'puffy_face_and_eyes',
        'enlarged_thyroid', 'brittle_nails', 'swollen_extremeties', 'excessive_hunger',
        'extra_marital_contacts', 'drying_and_tingling_lips', 'slurred_speech',
        'knee_pain', 'hip_joint_pain', 'muscle_weakness', 'stiff_neck',
        'swelling_joints', 'movement_stiffness', 'spinning_movements', 'loss_of_balance',
        'unsteadiness', 'weakness_of_one_body_side', 'loss_of_smell', 'bladder_discomfort',
        'foul_smell_of urine', 'continuous_feel_of_urine', 'passage_of_gases',
        'internal_itching', 'toxic_look_(typhos)', 'depression', 'irritability',
        'muscle_pain', 'altered_sensorium', 'red_spots_over_body', 'belly_pain',
        'abnormal_menstruation', 'dischromic _patches', 'watering_from_eyes',
        'increased_appetite', 'polyuria', 'family_history', 'mucoid_sputum',
        'rusty_sputum', 'lack_of_concentration', 'visual_disturbances',
        'receiving_blood_transfusion', 'receiving_unsterile_injections',
        'coma', 'stomach_bleeding', 'distention_of_abdomen', 'history_of_alcohol_consumption',
        'fluid_overload', 'blood_in_sputum', 'prominent_veins_on_calf', 'palpitations',
        'painful_walking', 'pus_filled_pimples', 'blackheads', 'scurring', 'skin_peeling',
        'silver_like_dusting', 'small_dents_in_nails', 'inflammatory_nails',
        'blister', 'red_sore_around_nose', 'yellow_crust_ooze'
    ])

    prediction_text = None
    recommendation = None

    if request.method == 'POST':
        s1 = request.form['Symptom1']
        s2 = request.form['Symptom2']
        s3 = request.form['Symptom3']
        s4 = request.form['Symptom4']
        s5 = request.form['Symptom5']

        # Predict
        result = model.predict([[s1, s2, s3, s4, s5]])[0]
        prediction_text = result

        # Recommendation lookup
        recommendations = {
            '(vertigo) Paroymsal Positional Vertigo': "Perform targeted vestibular exercises and avoid rapid head movements; consult your ENT specialist if episodes persist.",
            'AIDS': "Strictly follow your antiretroviral therapy, maintain a balanced diet, and have regular check-ups to monitor your immune status.",
            'Acne': "Maintain a gentle skincare routine, avoid harsh products, and consider seeing a dermatologist for persistent breakouts.",
            'Alcoholic hepatitis': "Abstain from alcohol immediately and adhere to a specialized treatment plan and liver-friendly diet under your doctor’s guidance.",
            'Allergy': "Identify and avoid known allergens, use antihistamines as needed, and consider professional advice for long-term management.",
            'Arthritis': "Engage in low-impact exercises, incorporate anti-inflammatory foods in your diet, and use pain management strategies as advised by your doctor.",
            'Bronchial Asthma': "Follow your inhaler regimen without fail, avoid known triggers, and monitor symptoms regularly with your pulmonologist’s help.",
            'Cervical spondylosis': "Practice proper posture, perform neck-strengthening exercises, and use prescribed therapies to alleviate discomfort.",
            'Chicken pox': "Rest well, stay hydrated, use soothing lotions for itching, and isolate yourself to prevent spreading the virus.",                'Chronic cholestasis': "Follow your doctor’s treatment plan, monitor liver function tests regularly, and adhere to dietary recommendations that reduce liver strain.",
            'Common Cold': "Rest, keep yourself well-hydrated, and use over-the-counter remedies to ease symptoms while your body recovers.",
            'Dengue': "Stay hydrated, use paracetamol for fever (avoiding NSAIDs), and seek medical care promptly if symptoms worsen.",
            'Diabetes ': "Monitor your blood sugar levels consistently, follow dietary and exercise guidelines, and take medications as prescribed.",
            'Dimorphic hemmorhoids(piles)': "Increase your fiber intake, drink plenty of water, and use topical treatments or seek medical advice to ease discomfort.",
            'Drug Reaction': "Discontinue the suspected medication immediately and consult your healthcare provider for proper evaluation and treatment.",
            'Fungal infection': "Maintain good hygiene, keep affected areas dry, and apply antifungal medications as directed by your doctor.",
            'GERD': "Avoid trigger foods, eat smaller meals, and consider lifestyle modifications alongside your prescribed acid-reducing treatments.",                'Gastroenteritis': "Stay well-hydrated with oral rehydration solutions, follow a bland diet, and rest until your symptoms improve.",
            'Heart attack': "Follow your cardiologist’s rehabilitation program, adopt a heart-healthy lifestyle, and strictly adhere to your medication schedule.",
            'Hepatitis B': "Stick to your antiviral regimen, monitor your liver function regularly, and maintain lifestyle adjustments that support liver health.",
            'Hepatitis C': "Complete your full course of antiviral therapy, keep regular appointments for liver monitoring, and consider dietary modifications for overall wellness.",
            'Hepatitis D': "Follow your doctor's treatment plan diligently, avoid alcohol, and have regular liver function evaluations to manage your condition.",                'Hepatitis E': "Rest, maintain proper hydration, and follow your healthcare provider’s recommendations for supportive recovery of liver function.",
            'hepatitis A': "Rest adequately, follow a light and nutritious diet, stay hydrated, and adhere to your healthcare provider’s advice for liver recovery."
        }

        recommendation = recommendations.get(prediction_text, "Please consult a healthcare provider for further guidance.")

    return render_template("infectious.html",
                           symptoms=symptoms,
                           prediction_text=prediction_text,
                           recommendation=recommendation)

# -----------------------------(Diabetes)----------------------------------- #
# Load the trained model
with open('models/diabetes_model.pkl', 'rb') as f:
    diabetes_model = pickle.load(f)

@app.route('/diabetes', methods=['GET'])
def diabetes_form():
    return render_template('diabetes.html')  # Ensure your HTML file is in the templates/ folder

@app.route('/diabetes', methods=['POST'])
def diabetes_predict():
    data = request.get_json()
    try:
        features = [
            float(data['pregnancies']), float(data['glucose']), float(data['blood_pressure']),
            float(data['skin_thickness']), float(data['insulin']), float(data['bmi']),
            float(data['dpf']), float(data['age'])
        ]
        prediction = diabetes_model.predict([features])[0]
        recommendation = "Maintain a healthy lifestyle." if prediction == 0 else "Consult a doctor and monitor regularly."

        return jsonify({'prediction': int(prediction), 'recommendation': recommendation})
    except Exception as e:
        print("Error:", e)
        return jsonify({'error': 'Prediction failed'}), 400

if __name__ == '__main__':
    app.run(debug=True)

# -----------------------------(Heart)----------------------------------- #
# Load heart model
with open('models/heart_disease_model.pkl', 'rb') as f:
    heart_disease_model = pickle.load(f)

@app.route('/heart', methods=['GET'])
def heart():
    return render_template('heart.html')

@app.route('/heart', methods=['POST'])
def heart_predict():
    data = request.get_json()
    try:
        features = [
            float(data['age']), int(data['sex']), int(data['cp']),
            float(data['trestbps']), float(data['chol']), int(data['fbs']),
            int(data['restecg']), float(data['thalach']), int(data['exang']),
            float(data['oldpeak']), int(data['slope']), int(data['ca']),
            int(data['thal'])
        ]
        prediction = heart_disease_model.predict([features])[0]
        recommendation = "Consult cardiologist." if prediction == 1 else "Heart looks healthy."

        return jsonify({'prediction': int(prediction), 'recommendation': recommendation})
    except Exception as e:
        print("Error:", e)
        return jsonify({'error': 'Prediction failed'}), 400


if __name__ == "__main__":
    app.run(debug=True)

# -----------------------------(Parkinson's)----------------------------------- #
@app.route('/parkinson', methods=['POST'])
def parkinson_predict():
    try:
        data = request.get_json()
        print("Received data:", data)
        features = [
            data['mdvp_fo'], data['mdvp_fhi'], data['mdvp_flo'], data['mdvp_jitter_percent'], data['mdvp_jitter_abs'],
            data['mdvp_rap'], data['mdvp_ppq'], data['jitter_ddp'], data['mdvp_shimmer'], data['mdvp_shimmer_db'],
            data['shimmer_apq3'], data['shimmer_apq5'], data['mdvp_apq'], data['shimmer_dda'], data['nhr'],
            data['hnr'], data['rpde'], data['dfa'], data['spread1'], data['spread2'], data['d2'], data['ppe']
        ]
        prediction = parkinsons_model.predict([features])[0]
        recommendation = "Follow up with neurology therapy." if prediction == 1 else "No signs of Parkinson's detected."

        return jsonify({'prediction': int(prediction), 'recommendation': recommendation})
    except Exception as e:
        print("Error during prediction:", e)
        return jsonify({'error': 'Prediction failed'}), 400

    
if __name__ == '__main__':
    app.run(debug=True)

# -----------------------------(Breast Cancer)----------------------------------- #
@app.route('/cancer', methods=['POST'])
def cancer_predict():
    data = request.get_json()
    try:
        features = [
            float(data['worst_area']), float(data['mean_concavity']), float(data['mean_radius']),
            float(data['mean_area']), float(data['worst_concave_points']), float(data['worst_perimeter']),
            float(data['worst_concavity']), float(data['area_error']), float(data['mean_concave_points']),
            float(data['worst_radius']), float(data['mean_perimeter']), float(data['worst_compactness'])
        ]

        prediction = breast_cancer_model.predict([features])[0]
        recommendation = "Benign tumor. Regular monitoring suggested." if prediction == 1 else "Malignant tumor. Immediate follow-up required."

        return jsonify({'prediction': int(prediction), 'recommendation': recommendation})
    except Exception as e:
        return jsonify({'error': str(e)}), 400