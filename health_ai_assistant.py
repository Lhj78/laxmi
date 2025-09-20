#from apythonsyncio import timeout

import streamlit as st
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
import os

import overpy
import folium
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
from streamlit_folium import st_folium


obesity_model = joblib.load("models/obesity_model.pkl")
disease_model = joblib.load("models/multi_disease_model.pkl")
scaler = joblib.load("models/scaler.pkl")
label_encoders = joblib.load("models/label_encoders.pkl")

obesity_classes = {
    0: "Insufficient Weight", 1: "Normal Weight", 2: "Overweight Level I",
    3: "Overweight Level II", 4: "Obesity Type I", 5: "Obesity Type II", 6: "Obesity Type III"
}

disease_classes = ["Low", "Moderate", "High"]

diet_recommendations = {
    "Insufficient Weight": "Increase calories with protein-rich foods like eggs, milk, nuts, and meat. Avoid junk food.",
    "Normal Weight": "Maintain a balanced diet. Include fruits, vegetables, lean proteins, and whole grains.",
    "Overweight Level I": "Reduce sugar/fat. Opt for grilled/baked foods. Eat smaller portions.",
    "Overweight Level II": "Consult a dietician. Eat fiber-rich meals. Avoid snacking between meals.",
    "Obesity Type I": "Follow a medically guided low-calorie diet. Focus on high-fiber vegetables, low-GI foods.",
    "Obesity Type II": "Strict low-calorie diet. Avoid processed/sugary foods completely.",
    "Obesity Type III": "Medical supervision needed. Consider meal replacements and professional intervention."
}

exercise_recommendations = {
    "Insufficient Weight": "Light strength training, yoga. Avoid over-exertion.",
    "Normal Weight": "Regular brisk walking, cycling, or 30 mins of moderate activity daily.",
    "Overweight Level I": "Brisk walking, swimming, low-impact cardio for 30–45 mins.",
    "Overweight Level II": "Daily walking and light aerobic exercise. Start slow and increase.",
    "Obesity Type I": "Supervised workouts – walking, elliptical. Avoid high-intensity at first.",
    "Obesity Type II": "Low-impact under guidance. Start with 10-min walks and build up.",
    "Obesity Type III": "Specialist-supervised physical therapy and minimal strain activity."
}

DATA_FILE = "user_health_data.xlsx"

st.title("\U0001F3E5 Obesity & Disease Risk Tracker")

st.header("\U0001F9CD Enter Your Health Data")
username = st.text_input("Name")
gender = st.selectbox("Gender", label_encoders["Gender"].classes_)
age = st.slider("Age", 10, 80, 25)
height = st.number_input("Height (meters)", 1.0, 2.5, 1.6)
weight = st.number_input("Weight (kg)", 20.0, 100.0, 80.0)

family_history = st.selectbox("Family History of Overweight", label_encoders["family_history_with_overweight"].classes_)
favc = st.selectbox("Frequent High-Calorie Food", label_encoders["FAVC"].classes_)
fcvc = st.slider("Vegetable Consumption (1-3)", 1.0, 3.0, 2.0)
ncp = st.slider("Main Meals per Day", 1.0, 4.0, 3.0)
caec = st.selectbox("Eating Between Meals", label_encoders["CAEC"].classes_)
smoke = st.selectbox("Do You Smoke?", label_encoders["SMOKE"].classes_)
ch2o = st.slider("Water Intake (liters/day)", 1.0, 3.0, 2.0)
scc = st.selectbox("Monitor Calorie Intake?", label_encoders["SCC"].classes_)
faf = st.slider("Physical Activity (hrs/week)", 0.0, 50.0, 2.0)
tue = st.slider("Tech Use (hrs/day)", 0.0, 20.0, 1.0)
calc = st.selectbox("Alcohol Consumption", label_encoders["CALC"].classes_)
mtrans = st.selectbox("Transportation Mode", label_encoders["MTRANS"].classes_)

if st.button("\U0001F50D Predict & Track"):
    input_data = [
        label_encoders["Gender"].transform([gender])[0], age, height, weight,
        label_encoders["family_history_with_overweight"].transform([family_history])[0],
        label_encoders["FAVC"].transform([favc])[0], fcvc, ncp,
        label_encoders["CAEC"].transform([caec])[0],
        label_encoders["SMOKE"].transform([smoke])[0], ch2o,
        label_encoders["SCC"].transform([scc])[0], faf, tue,
        label_encoders["CALC"].transform([calc])[0],
        label_encoders["MTRANS"].transform([mtrans])[0],
    ]

    scaled = scaler.transform([input_data])
    obesity_pred = obesity_model.predict(scaled)[0]
    obesity_level = obesity_classes[obesity_pred]

    disease_pred = disease_model.predict(scaled)[0]
    diabetes_risk = disease_classes[disease_pred[0]]
    hypertension_risk = disease_classes[disease_pred[1]]
    heart_risk = disease_classes[disease_pred[2]]

    bmi = round(weight / (height ** 2), 2)
    today = datetime.today().strftime("%Y-%m-%d")

    record = pd.DataFrame([{
        "Date": today, "Name": username, "Weight": weight, "Height": height,
        "BMI": bmi, "Obesity_Level": obesity_level,
        "Diabetes_Risk": diabetes_risk,
        "Hypertension_Risk": hypertension_risk,
        "Heart_Disease_Risk": heart_risk
    }])

    if os.path.exists(DATA_FILE):
        df = pd.read_excel(DATA_FILE)
        df = pd.concat([df, record], ignore_index=True)
    else:
        df = record

    df.to_excel(DATA_FILE, index=False)

    st.subheader("\U0001F4CA Prediction Results")
    st.success(f"Obesity Level: **{obesity_level}**")

    st.subheader("\U0001F957 Diet Recommendation")
    st.info(diet_recommendations[obesity_level])

    st.subheader("\U0001F3C3 Exercise Suggestion")
    st.info(exercise_recommendations[obesity_level])

    st.info(f"Diabetes Risk: {diabetes_risk}\nHypertension Risk: {hypertension_risk}\nHeart Disease Risk: {heart_risk}")

    user_data = df[df["Name"] == username]
    if len(user_data) >= 2:
        st.subheader("\U0001F4C8 Weight & BMI Trend")
        st.line_chart(user_data.set_index("Date")["Weight"])

        old, new = user_data.iloc[-2], user_data.iloc[-1]
        st.subheader("\U0001F4C9 Weight Change")
        delta = new["Weight"] - old["Weight"]
        trend = "Stable" if abs(delta) < 0.5 else ("Gained" if delta > 0 else "Lost")
        st.info(f"Since last check: **{trend}**")

        st.subheader("\U0001F4C9 Disease Risk Changes")
        for disease in ["Diabetes", "Hypertension", "Heart_Disease"]:
            old_risk = old[f"{disease}_Risk"]
            new_risk = new[f"{disease}_Risk"]
            if old_risk == new_risk:
                st.info(f"{disease} Risk: No Change (**{new_risk}**)")
            elif new_risk in disease_classes and old_risk in disease_classes and \
                disease_classes.index(new_risk) < disease_classes.index(old_risk):
                st.success(f"{disease} Risk Improved: {old_risk} → {new_risk}")
            elif new_risk in disease_classes and old_risk in disease_classes:
                st.error(f"{disease} Risk Worsened: {old_risk} → {new_risk}")
            else:
                st.warning(f"⚠️ Unknown risk value: {old_risk} or {new_risk}")

st.header("\U0001F3D9 Hospital Locator")
user_location = st.text_input("Enter your location (e.g. City, Address)", "Bhopal, India")
if user_location:
    geolocator = Nominatim(user_agent="health_app",timeout = 30)
    location = geolocator.geocode(user_location)
    if location:
        lat, lon = location.latitude, location.longitude
        api = overpy.Overpass()
        query = f"""
        [out:json];
        node["amenity"="hospital"](around:15000,{lat},{lon});
        out;
        """
        result = api.query(query)

        hospital_map = folium.Map(location=[lat, lon], zoom_start=12)
        for node in result.nodes:
            name = node.tags.get("name", "Unnamed Hospital")
            hosp_location = (float(node.lat), float(node.lon))
            distance_km = geodesic((lat, lon), hosp_location).km
            direction_url = f"https://www.google.com/maps/dir/{lat},{lon}/{node.lat},{node.lon}"

            folium.Marker(
                location=hosp_location,
                popup=f"{name}<br>Distance: {distance_km:.2f} km<br><a href='{direction_url}' target='_blank'>Get Directions</a>",
                icon=folium.Icon(color="blue", icon="plus-sign")
            ).add_to(hospital_map)

        st.subheader("\U0001F5FA Nearby Hospitals")
        st_folium(hospital_map, width=700, height=500)
    else:
        st.error("Could not geocode the location. Please try another input.")