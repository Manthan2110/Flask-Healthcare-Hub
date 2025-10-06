from flask import Flask, request, jsonify, render_template, send_from_directory
import numpy as np
import pandas as pd
from tensorflow.keras.applications.vgg19 import preprocess_input as vgg19_preprocess
import joblib
import pickle
from datetime import datetime
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array, save_img 
import ssl
import os
from flask_cors import CORS
from sklearn.preprocessing import StandardScaler
from werkzeug.utils import secure_filename

app = Flask(__name__)
CORS(app)

@app.route('/monitor')
def monitor():
    return render_template("Monitor.html")

@app.route('/food')
def food():
    return render_template("Food.html")

@app.route('/')
def home():
    return render_template("Home.html")

@app.route('/features')
def features():
    return render_template("Feature.html")

@app.route('/contact')
def contact():
    return render_template("Contact.html")

@app.route('/chatbot')
def chatbot():
    return render_template("Chatbot.html")

@app.route('/about')
def about():
    return render_template("About.html")

@app.route('/login')
def login():
    return render_template("Login.html")

# Load models
diabetes_model = pickle.load(open('models/diabetes_model.pkl', 'rb'))
heart_disease_model = pickle.load(open('models/heart_disease_model.pkl', 'rb'))
parkinsons_model = pickle.load(open('models/parkinsons_model.pkl', 'rb'))
breast_cancer_model = pickle.load(open('models/breast_cancer_model.pkl', 'rb'))
model = joblib.load("models/Random_forest_model.pkl")
Symptoms_model = pickle.load(open('models/svc.pkl', 'rb'))
blood_group_model = pickle.load(open('models/blood_grp_detection.pkl', 'rb'))
# brain_model = load_model("models/brain_model.keras", safe_mode=False, compile=False)
# skin_cancer_model = keras.models.load_model("models/skin_cancer_cnn_fixed.keras", compile=False)


@app.route('/app')
def index():
    return render_template('App.html')

# -----------------------------Symptoms-Based----------------------------------- #
# Dataset Load
sym_des = pd.read_csv("Dataset/symtoms_df.csv")
precautions = pd.read_csv("Dataset/precautions_df.csv")
workout = pd.read_csv("Dataset/workout_df.csv")
description = pd.read_csv("Dataset/description.csv")
medications = pd.read_csv('Dataset/medications.csv')
diets = pd.read_csv("Dataset/diets.csv")

#helper function
def helper(dis):
    desc = description[description['Disease'] == dis]['Description']
    desc = " ".join([w for w in desc])

    pre = precautions[precautions['Disease'] == dis][['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']]
    pre = [col for col in pre.values]

    med = medications[medications['Disease'] == dis]['Medication']
    med = [med for med in med.values]

    die = diets[diets['Disease'] == dis]['Diet']
    die = [die for die in die.values]

    wrkout = workout[workout['disease'] == dis] ['workout']


    return desc,pre,med,die,wrkout

symptoms_dict = {'itching': 0, 'skin_rash': 1, 'nodal_skin_eruptions': 2, 'continuous_sneezing': 3, 'shivering': 4, 'chills': 5, 'joint_pain': 6, 'stomach_pain': 7, 'acidity': 8, 'ulcers_on_tongue': 9, 'muscle_wasting': 10, 'vomiting': 11, 'burning_micturition': 12, 'spotting_ urination': 13, 'fatigue': 14, 'weight_gain': 15, 'anxiety': 16, 'cold_hands_and_feets': 17, 'mood_swings': 18, 'weight_loss': 19, 'restlessness': 20, 'lethargy': 21, 'patches_in_throat': 22, 'irregular_sugar_level': 23, 'cough': 24, 'high_fever': 25, 'sunken_eyes': 26, 'breathlessness': 27, 'sweating': 28, 'dehydration': 29, 'indigestion': 30, 'headache': 31, 'yellowish_skin': 32, 'dark_urine': 33, 'nausea': 34, 'loss_of_appetite': 35, 'pain_behind_the_eyes': 36, 'back_pain': 37, 'constipation': 38, 'abdominal_pain': 39, 'diarrhoea': 40, 'mild_fever': 41, 'yellow_urine': 42, 'yellowing_of_eyes': 43, 'acute_liver_failure': 44, 'fluid_overload': 45, 'swelling_of_stomach': 46, 'swelled_lymph_nodes': 47, 'malaise': 48, 'blurred_and_distorted_vision': 49, 'phlegm': 50, 'throat_irritation': 51, 'redness_of_eyes': 52, 'sinus_pressure': 53, 'runny_nose': 54, 'congestion': 55, 'chest_pain': 56, 'weakness_in_limbs': 57, 'fast_heart_rate': 58, 'pain_during_bowel_movements': 59, 'pain_in_anal_region': 60, 'bloody_stool': 61, 'irritation_in_anus': 62, 'neck_pain': 63, 'dizziness': 64, 'cramps': 65, 'bruising': 66, 'obesity': 67, 'swollen_legs': 68, 'swollen_blood_vessels': 69, 'puffy_face_and_eyes': 70, 'enlarged_thyroid': 71, 'brittle_nails': 72, 'swollen_extremeties': 73, 'excessive_hunger': 74, 'extra_marital_contacts': 75, 'drying_and_tingling_lips': 76, 'slurred_speech': 77, 'knee_pain': 78, 'hip_joint_pain': 79, 'muscle_weakness': 80, 'stiff_neck': 81, 'swelling_joints': 82, 'movement_stiffness': 83, 'spinning_movements': 84, 'loss_of_balance': 85, 'unsteadiness': 86, 'weakness_of_one_body_side': 87, 'loss_of_smell': 88, 'bladder_discomfort': 89, 'foul_smell_of urine': 90, 'continuous_feel_of_urine': 91, 'passage_of_gases': 92, 'internal_itching': 93, 'toxic_look_(typhos)': 94, 'depression': 95, 'irritability': 96, 'muscle_pain': 97, 'altered_sensorium': 98, 'red_spots_over_body': 99, 'belly_pain': 100, 'abnormal_menstruation': 101, 'dischromic _patches': 102, 'watering_from_eyes': 103, 'increased_appetite': 104, 'polyuria': 105, 'family_history': 106, 'mucoid_sputum': 107, 'rusty_sputum': 108, 'lack_of_concentration': 109, 'visual_disturbances': 110, 'receiving_blood_transfusion': 111, 'receiving_unsterile_injections': 112, 'coma': 113, 'stomach_bleeding': 114, 'distention_of_abdomen': 115, 'history_of_alcohol_consumption': 116, 'fluid_overload.1': 117, 'blood_in_sputum': 118, 'prominent_veins_on_calf': 119, 'palpitations': 120, 'painful_walking': 121, 'pus_filled_pimples': 122, 'blackheads': 123, 'scurring': 124, 'skin_peeling': 125, 'silver_like_dusting': 126, 'small_dents_in_nails': 127, 'inflammatory_nails': 128, 'blister': 129, 'red_sore_around_nose': 130, 'yellow_crust_ooze': 131}
diseases_list = {15: 'Fungal infection', 4: 'Allergy', 16: 'GERD', 9: 'Chronic cholestasis', 14: 'Drug Reaction', 33: 'Peptic ulcer diseae', 1: 'AIDS', 12: 'Diabetes ', 17: 'Gastroenteritis', 6: 'Bronchial Asthma', 23: 'Hypertension ', 30: 'Migraine', 7: 'Cervical spondylosis', 32: 'Paralysis (brain hemorrhage)', 28: 'Jaundice', 29: 'Malaria', 8: 'Chicken pox', 11: 'Dengue', 37: 'Typhoid', 40: 'hepatitis A', 19: 'Hepatitis B', 20: 'Hepatitis C', 21: 'Hepatitis D', 22: 'Hepatitis E', 3: 'Alcoholic hepatitis', 36: 'Tuberculosis', 10: 'Common Cold', 34: 'Pneumonia', 13: 'Dimorphic hemmorhoids(piles)', 18: 'Heart attack', 39: 'Varicose veins', 26: 'Hypothyroidism', 24: 'Hyperthyroidism', 25: 'Hypoglycemia', 31: 'Osteoarthristis', 5: 'Arthritis', 0: '(vertigo) Paroymsal  Positional Vertigo', 2: 'Acne', 38: 'Urinary tract infection', 35: 'Psoriasis', 27: 'Impetigo'}

# Model Prediction function
def get_predicted_value(patient_symptoms):
    input_vector = np.zeros(len(symptoms_dict))
    for item in patient_symptoms:
        input_vector[symptoms_dict[item]] = 1
    return diseases_list[Symptoms_model.predict([input_vector])[0]]

# creating routes========================================

@app.route('/predict', methods=['GET', 'POST'])
def infectious_predict():
    symptoms_list = list(symptoms_dict.keys())  # all 132 symptoms

    if request.method == 'POST':
        user_symptoms = request.form.getlist('symptoms')  # multiple selections

        if not user_symptoms:  # only check on form submit
            message = "⚠️ Please select at least one symptom"
            return render_template('infectious.html',
                                   message=message,
                                   symptoms_list=symptoms_list)

        # validate symptoms
        invalid = [s for s in user_symptoms if s not in symptoms_dict]
        if invalid:
            message = f"⚠️ These symptoms are not recognized: {', '.join(invalid)}"
            return render_template('infectious.html',
                                   message=message,
                                   symptoms_list=symptoms_list)

        # predict
        predicted_disease = get_predicted_value(user_symptoms)
        dis_des, precautions, medications, rec_diet, workout = helper(predicted_disease)

        my_precautions = list(precautions[0])

        return render_template(
            'infectious.html',
            symptoms_list=symptoms_list,
            predicted_disease=predicted_disease,
            dis_des=dis_des,
            my_precautions=my_precautions,
            medications=medications,
            my_diet=rec_diet,
            workout=workout
        )

    # ✅ On first page load → just render form with dropdown
    return render_template('infectious.html', symptoms_list=symptoms_list)


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
        recommendation = "😊 Good news!<br> Maintain a balanced diet. You can check out Food Composition Section. Exercise regularly and keep a healthy weight."\
            if prediction == 0 else "Consult your doctor for ongoing care and personalized treatment. <br> Monitor your blood sugar regularly and follow prescribed medication. Check Our Food Composition Section to make diet plan."

        return jsonify({'prediction': int(prediction), 'recommendation': recommendation})
    except Exception as e:
        print("Error:", e)
        return jsonify({'error': 'Prediction failed'}), 400

# -----------------------------(Heart)----------------------------------- #
# Load heart model
with open('models/heart_disease_model.pkl', 'rb') as f:
    heart_disease_model = pickle.load(f)

@app.route('/heart', methods=['GET'])
def heart_form():
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
        features = np.array(features).reshape(1, -1)
        prediction = heart_disease_model.predict(features)[0]

        # Generate recommendation
        if prediction == 1:
            recommendation = (
                "⚠️ Serious alert! <br> Lifestyle modifications are strongly advised: "
                "maintain a balanced low-fat diet, reduce salt intake, exercise regularly under medical supervision, "
                "quit smoking, limit alcohol, and follow up with a cardiologist."
            )
        else:
            recommendation = (
                "😊 Good news! <br>But Keep maintaining a healthy lifestyle exercise regularly, eat nutritious food, and get routine checkups to keep your heart in good condition. Check out our Food Composition to get nutritious Plan.")


        return jsonify({'prediction': int(prediction), 'recommendation': recommendation})
    except Exception as e:
        print("Error:", e)
        return jsonify({'error': 'Prediction failed'}), 400

# -----------------------------(Parkinson's)----------------------------------- #
@app.route('/parkinson', methods=['GET'])
def parkinson():
    return render_template('parkinson.html')

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
        recommendation = "⚠️ Serious alert! <br> consult a neurologist ASAP. Maintain an active lifestyle with regular exercise and a balanced diet. Go for routine health check-ups to ensure long-term well-being." \
            if prediction == 1 else "😊 Good news! <br> No signs of Parkinson's detected but still Follow up with a neurologist for regular monitoring and treatment. Engage in physical therapy and exercises to maintain mobility."

        return jsonify({'prediction': int(prediction), 'recommendation': recommendation})
    except Exception as e:
        print("Error during prediction:", e)
        return jsonify({'error': 'Prediction failed'}), 400

# -----------------------------(Breast Cancer)----------------------------------- #

@app.route('/cancer', methods=['GET'])
def cancer():
    return render_template('cancer.html')

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
        recommendation = "😊 Good news! <br>Continue regular health check-ups and screenings. Maintain a healthy lifestyle with balanced diet and exercise." \
            if prediction == 1 else "🚨 Serious alert! <br>Seek immediate medical consultation with an oncologist. Follow the recommended treatment plan (surgery, chemo, or radiation if advised)."

        return jsonify({'prediction': int(prediction), 'recommendation': recommendation})
    except Exception as e:
        return jsonify({'error': str(e)}), 400
    
# -----------------------------(Skin Cancer Detection)----------------------------------- #
# --- Config ---
UPLOAD_FOLDER = "uploads"
ALLOWED_EXT = {"png", "jpg", "jpeg"}
MODEL_PATH = "models/skin_cancer_cnn.h5"   # <-- change to your actual model file
IMG_SIZE = (224, 224)          # <-- must match model training size

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# --- Load model once at startup ---
try:
    skin_cancer_model = load_model(MODEL_PATH)
    print(f"[INFO] Loaded model from {MODEL_PATH}")
except Exception as e:
    skin_cancer_model = None
    print(f"[ERROR] Failed to load model: {e}")

# --- Helpers ---
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXT

def predict_skin_cancer(img_path, model):
    """
    Loads image, preprocesses, runs model.predict and returns:
    - class_label: "Malignant" or "Benign"
    - prob: probability/confidence (float in [0,1]) of detected class (interpreted)
    """
    # load + preprocess
    img = load_img(img_path, target_size=IMG_SIZE)
    arr = img_to_array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)

    pred = model.predict(arr, verbose=0)  # shape may be (1,1) or (1,2) or (1,n)
    pred = np.asarray(pred)

    # handle different output shapes:
    if pred.ndim == 2 and pred.shape[1] == 1:
        # single-probability output like sigmoid -> [[0.87]]
        prob = float(pred[0][0])
        class_label = "Malignant" if prob > 0.5 else "Benign"
    elif pred.ndim == 2 and pred.shape[1] >= 2:
        # multi-class softmax -> take argmax
        class_idx = int(np.argmax(pred[0]))
        prob = float(pred[0][class_idx])
        # assume class index 1 => Malignant, 0 => Benign (adapt if your classes differ)
        class_label = "Malignant" if class_idx == 1 else "Benign"
    else:
        # fallback
        prob = float(pred.ravel()[0])
        class_label = "Malignant" if prob > 0.5 else "Benign"

    return class_label, prob

# --- Routes ---
@app.route('/skin', methods=['GET'])
def skin_form():
    # make sure skin_cancer.html is inside templates/
    return render_template("skin_cancer.html")


@app.route('/skin', methods=['POST'])
def skin_predict():
    if skin_cancer_model is None:
        return jsonify({'error': 'Model not loaded on server'}), 500

    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': 'File type not allowed'}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    try:
        class_label, prob = predict_skin_cancer(filepath, skin_cancer_model)

        recommendation = (
            "😊 Good news! The lesion appears benign, but monitor for changes and maintain routine dermatology checks."
            if class_label == "Benign"
            else "🚨 Urgent: The lesion shows malignant signs.<brhea> Consult a dermatologist immediately for confirmatory tests/biopsy."
        )

        # optional: remove uploaded file to save space
        # try: os.remove(filepath)
        # except Exception: pass

        return jsonify({
            'prediction': class_label,
            'probability': prob,
            'recommendation': recommendation
        })
    except Exception as e:
        # return error in JSON so frontend can show it
        return jsonify({'error': str(e)}), 500

# -----------------------------(Lungs Infectious Detection)----------------------------------- #

# ---------------- Pneumonia Model Setup ----------------
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Flatten, Dense, Dropout
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.utils import load_img, img_to_array

# Build the same architecture as training
def build_pneumonia_model(weights_path="models/vgg_unfrozen.h5"):
    base_model = VGG19(include_top=False, input_shape=(128, 128, 3))
    x = base_model.output
    flat = Flatten()(x)
    class_1 = Dense(4608, activation='relu')(flat)
    drop_out = Dropout(0.2)(class_1)
    class_2 = Dense(1152, activation='relu')(drop_out)
    output = Dense(2, activation='softmax')(class_2)
    model = Model(base_model.inputs, output)
    model.load_weights(weights_path)
    return model

try:
    pneumonia_model = build_pneumonia_model("models/vgg_unfrozen.h5")
    print("[INFO] Pneumonia model loaded successfully.")
except Exception as e:
    pneumonia_model = None
    print(f"[ERROR] Could not load pneumonia model: {e}")


def predict_pneumonia(img_path, model):
    img = load_img(img_path, target_size=(128, 128))
    arr = img_to_array(img)
    arr = np.expand_dims(arr, axis=0)  # (1,128,128,3)
    preds = model.predict(arr, verbose=0)
    class_idx = np.argmax(preds, axis=1)[0]
    confidence = float(np.max(preds))

    if class_idx == 0:
        label = "Normal"
        recommendation = "✅ No pneumonia detected. Maintain lung health with exercise, avoid smoking, and get regular check-ups."
    else:
        label = "Pneumonia"
        recommendation = "⚠️ Signs of pneumonia detected. Please consult a doctor immediately for further medical evaluation and treatment."

    return label, confidence, recommendation


# ---------------- Pneumonia Routes ----------------
@app.route('/pneumonia', methods=['GET'])
def pneumonia_form():
    return render_template("Lungs.html")  # make sure you have this template


@app.route('/pneumonia', methods=['POST'])
def pneumonia_predict():
    if pneumonia_model is None:
        return jsonify({'error': 'Pneumonia model not loaded'}), 500

    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join("uploads", filename)
    file.save(filepath)

    try:
        label, confidence, recommendation = predict_pneumonia(filepath, pneumonia_model)
        return jsonify({
            'prediction': label,
            'probability': round(confidence, 4),
            'recommendation': recommendation
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# -----------------------------(Brain Tumor Detection)----------------------------------- #
#load the train model

# #class labels
# class_labels = ['pituitary', 'gloima', 'notumor', 'meningioma']
#
# #define the upload folder
# UPLOAD_FOLDER = "./brain_img_uploads"
# if not os.path.exists(UPLOAD_FOLDER):
#     os.makedir(UPLOAD_FOLDER)
#
# # Helper function to predict tumor type
# def predict_tumor(image_path, model, class_labels):
#     IMAGE_SIZE = 128
#
#     # Load and preprocess image
#     img = tf.keras.utils.load_img(image_path, target_size=(IMAGE_SIZE, IMAGE_SIZE))
#     img_array = tf.keras.utils.img_to_array(img) / 255.0  # Normalize
#     img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
#
#     # Prediction
#     predictions = model.predict(img_array)
#     predicted_class_index = np.argmax(predictions, axis=1)[0]
#     confidence_score = np.max(predictions, axis=1)[0]
#
#     if class_labels[predicted_class_index] == 'notumor':
#         return "No Tumor", confidence_score
#     else:
#         return f"Tumor: {class_labels[predicted_class_index]}", confidence_score
#
# #Routes
# @app.route("/brain", methods=['GET', 'POST'])
# def brain():
#     if request.method == 'POST':
#         #Handle file upload
#         file = request.files['file']
#
#         if file:
#             #save the file
#             file_location = os.path.join(app.config('UPLOAD_FOLDER'), file.filename)
#             file.save(file_location)
#
#             #predict function
#             result, confidence = predict_tumor(file_location)
#
#             #return result path
#             return render_template('brain.html', result=result, confidence=f'{confidence*100:.2f}%', file_path= f'/brain_img_uploads/{file.filename}')
#         return render_template('brain.html', result=None)
#
# # Route to serve uploaded files
# @app.route('/brain_img_uploads/<filename>')
# def get_uploaded_file(filename):
#     return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# -----------------------------(Liver Disease Detection)----------------------------------- #
MODELS_DIR = "models"

# ---------------- Load Model ----------------
TOP7_PATHS = [
    os.path.join(MODELS_DIR, "liver_random_forest_top7.pkl"),
    os.path.join(MODELS_DIR, "random_forest_top7.pkl"),
]
BEST_PATHS = [
    os.path.join(MODELS_DIR, "liver_random_forest_best.pkl"),
    os.path.join(MODELS_DIR, "random_forest_best.pkl"),
]

liver_model = None
liver_top7_features = None

def _try_load(paths, expect_tuple=False):
    for p in paths:
        if os.path.exists(p):
            obj = joblib.load(p)
            if expect_tuple:
                return obj
            return obj
    return None

_top7_tuple = _try_load(TOP7_PATHS, expect_tuple=True)
if _top7_tuple is not None and isinstance(_top7_tuple, tuple) and len(_top7_tuple) == 2:
    liver_model, liver_top7_features = _top7_tuple

if liver_model is None:
    liver_model = _try_load(BEST_PATHS)

if liver_model is None:
    raise RuntimeError("No liver model found in models/")

# ---------------- Features ----------------
if liver_top7_features is None:
    EXPECTED_FULL_FEATURES = [
        'Age','Gender','Total_Bilirubin','Direct_Bilirubin','Alkaline_Phosphotase',
        'Alamine_Aminotransferase','Aspartate_Aminotransferase','Total_Protiens',
        'Albumin','Albumin_and_Globulin_Ratio'
    ]
else:
    EXPECTED_FULL_FEATURES = liver_top7_features

# ---------------- Inline Scaler ----------------
# 👉 Option 1: If you have training dataset CSV, fit scaler here:
# df = pd.read_csv("data/ILPD.csv")   # adjust path
# X = df[EXPECTED_FULL_FEATURES]
# scaler = StandardScaler().fit(X)
#
# 👉 Option 2: If no dataset available, build a scaler with "safe defaults":
scaler = StandardScaler()
scaler.mean_ = np.zeros(len(EXPECTED_FULL_FEATURES))
scaler.scale_ = np.ones(len(EXPECTED_FULL_FEATURES))
scaler.n_features_in_ = len(EXPECTED_FULL_FEATURES)
# This acts like "no scaling" but keeps code compatible.

# ---------------- Feature Vector Builder ----------------
def _make_feature_vector(payload):
    def parse_float(x): return float(x) if x not in [None, ""] else np.nan
    def parse_int(x): return int(float(x)) if x not in [None, ""] else np.nan

    def normalize_gender(v):
        if isinstance(v, str):
            s = v.strip().lower()
            if s in ("male","m"): return 1
            if s in ("female","f"): return 0
        try:
            vv = int(float(v))
            return 1 if vv == 1 else 0
        except:
            return 0

    values = {}
    for f in EXPECTED_FULL_FEATURES:
        if f.lower() == "gender":
            values[f] = normalize_gender(payload.get("Gender"))
        elif f in ("Age",):
            values[f] = parse_int(payload.get(f))
        else:
            values[f] = parse_float(payload.get(f))

    row = np.array([values[f] for f in EXPECTED_FULL_FEATURES], dtype=float).reshape(1, -1)

    # Always apply scaler (safe default: identity if no dataset fitted)
    row = scaler.transform(row)
    return row

# ---------------- Routes ----------------
@app.route("/liver", methods=["GET"])
def liver_form():
    return render_template("liver.html")

@app.route("/liver", methods=["POST"])
def liver_predict():
    try:
        data = request.get_json() if request.is_json else request.form
        x = _make_feature_vector(data)

        if hasattr(liver_model, "predict_proba"):
            proba = liver_model.predict_proba(x)[0, 1]
        else:
            proba = float(liver_model.predict(x)[0])
        pred = int(liver_model.predict(x)[0])

        if pred == 1:
            msg = (
                "⚠️ Possible liver disease detected.<br>"
                "Please consult a hepatologist for evaluation. "
                "Avoid alcohol, review medications with your doctor, "
                "and consider LFT follow-ups."
            )
        else:
            msg = (
                "✅ No liver disease predicted.<br>"
                "Maintain a balanced diet, regular exercise, hydration, and routine checkups."
            )

        return jsonify({
            "prediction": pred,
            "probability": round(float(proba), 4),
            "used_features": EXPECTED_FULL_FEATURES,
            "recommendation": msg
        })
    except Exception as e:
        return jsonify({"error": f"Liver prediction failed: {str(e)}"}), 400
    

# -----------------------------(Blood Group Detection)----------------------------------- #
# Disable SSL verification (if needed)
ssl._create_default_https_context = ssl._create_unverified_context

# Define allow file types
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp'}

#function to check allowed file types
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS 

@app.route('/blood')
def blood():
    return render_template('blood.html')

@app.route('/predict_blood_group', methods=['POST'])
def predict_blood_group():
    try:
        if 'fingerprint' not in request.files:
            return jsonify({'error': 'No fingerprint image uploaded'}), 400

        file = request.files['fingerprint']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        if not allowed_file(file.filename):
            return jsonify({'error': 'File type not allowed. Please upload a PNG, JPG, or BMP image'}), 400

        # Create uploads directory if it doesn't exist
        if not os.path.exists('uploads'):
            os.makedirs('uploads')

        filename = secure_filename(file.filename)
        filepath = os.path.join('uploads', filename)
        file.save(filepath)

        try:
            # Load and preprocess the image
            img = load_img(filepath, target_size=(64,64))
            img_array = img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            
            # Make prediction
            predictions = blood_group_model.predict(img_array)
            blood_groups = ['A+', 'A-', 'B+', 'B-', 'AB+', 'AB-', 'O+', 'O-']
            predicted_class = np.argmax(predictions[0])
            confidence = float(predictions[0][predicted_class] * 100)
            
            return jsonify({
                'blood_group': blood_groups[predicted_class],
                'confidence': round(confidence, 2)
            })
            
        finally:
            # Clean up - remove the uploaded file
            if os.path.exists(filepath):
                os.remove(filepath)
                
    except Exception as e:
        print('Error:', str(e))  # Log the error for debugging
        return jsonify({'error': 'An error occurred during processing'}), 500


if __name__ == "__main__":
    app.run(debug=True)
