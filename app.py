from flask import Flask, jsonify, request
from flask_cors import CORS
import random, time
import pandas as pd
import os
from deep_translator import GoogleTranslator
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY')
# -----------------------------------------------------
# VITALS SECTION (original sensor_server.py)
# -----------------------------------------------------
heart_rate_data = []
spo2_data = []
max_points = 20

def generate_data():
    """Simulate vitals data"""
    global heart_rate_data, spo2_data
    hr = random.randint(60, 110)       # Heart Rate BPM
    spo2 = random.randint(88, 100)     # SpO2 %
    now = time.strftime("%H:%M:%S")
    # keep max 20 points
    if len(heart_rate_data) >= max_points:
        heart_rate_data.pop(0)
        spo2_data.pop(0)
    heart_rate_data.append({"time": now, "value": hr})
    spo2_data.append({"time": now, "value": spo2})
    return hr, spo2

@app.route("/vitals", methods=["GET"])
def vitals():
    hr, spo2 = generate_data()
    alerts = []
    if hr > 100:
        alerts.append(f"High heart rate: {hr} BPM")
    elif hr > 90:
        alerts.append(f"Elevated heart rate: {hr} BPM")

    if spo2 < 90:
        alerts.append(f"Low SpO₂: {spo2}%")
    elif spo2 < 95:
        alerts.append(f"Slightly low SpO₂: {spo2}%")

    if not alerts:
        alerts.append("All readings normal")

    return jsonify({
        "heart_rate": hr,
        "spo2": spo2,
        "heart_rate_chart": heart_rate_data,
        "spo2_chart": spo2_data,
        "alerts": alerts
    })


# -----------------------------------------------------
# SYMPTOM PREDICTION SECTION (original symptom.py)
# -----------------------------------------------------

# Helper: translation
def translate_to_english(text):
    try:
        return GoogleTranslator(source='auto', target='en').translate(text)
    except Exception as e:
        print("Translation error:", e)
        return text  # fallback

# Load data + train once
data = pd.read_csv("data.csv")
doctors = pd.read_csv("doctors.csv")
model = make_pipeline(TfidfVectorizer(), MultinomialNB())
model.fit(data["symptom"], data["disease"])

def predict_disease_and_doctor(user_input):
    translated = translate_to_english(user_input)
    disease = model.predict([translated])[0]
    doctor_row = doctors[doctors["disease"] == disease]
    if not doctor_row.empty:
        doctor = doctor_row["doctor"].values[0]
    else:
        doctor = "Doctor not found"
    return translated, disease, doctor

@app.route('/predict', methods=['POST'])
def predict():
    user_input = request.json.get('symptom')
    translated, disease, doctor = predict_disease_and_doctor(user_input)
    return jsonify({
        "translated": translated,
        "disease": disease,
        "doctor": doctor
    })

# -----------------------------------------------------
# MAIN ENTRYPOINT
# -----------------------------------------------------
if __name__ == "__main__":
    # for local testing
    app.run(host="0.0.0.0", port=5000, debug=True)
