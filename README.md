# laxmi

I make a health prediction tool it is custom base .It take user data and predict the health risk of obesity and diebetes. apart from this will also predict the risk of other diseases , those are related to obesity and diebetes. And suggest the exercise and food user don't have to take. if user try to check health again after takin health it track .

key features and tools used in this tool:
Tools Used in the Project
1. Programming & Frameworks

Python → Core programming language.

Streamlit → For building the interactive web-based application.

2. Machine Learning & Data Science

Scikit-learn → For ML models (RandomForest, GradientBoosting, XGBoost, etc.).

Pandas & NumPy → For data preprocessing, handling health/lifestyle datasets.

Matplotlib / Seaborn / Plotly → For visualization (trends, graphs, risk comparison).

3. Data Storage & Model Handling

Pickle (.pkl) → To save and load ML models for real-time predictions.

Excel (openpyxl / pandas) → To store user inputs, track previous vs. current data.

4. Healthcare Integration

Geopy / Google Maps API → For hospital locator (showing nearest hospitals and paths).

5. Other Supporting Tools

asyncio / timeout → For efficient execution where needed.

PyTorch / TensorFlow (optional if DL used) → For deep learning parts like personalized diet planning or food recognition.

🌟 Features of the Project
🔹 Core Features

Obesity Classification → Predicts category (Normal, Overweight, Obesity Type I/II/III) using health & lifestyle data.

Multi-Disease Risk Prediction → Diabetes, hypertension, heart disease, etc., predicted simultaneously.

🔹 Personalization & Tracking

BMI Calculator → Calculates Body Mass Index instantly.

Weight Trend Prediction → Shows if the user is gaining, losing, or stable.

Improvement Tracking → Compares past vs. current results (whether risk improved or worsened).

Data Storage in Excel → Keeps record of first-time and follow-up inputs for future comparison.

🔹 Visualization & Insights

Graphs & Charts → Trend graphs for weight, obesity, and disease risks.

PDF Report Generator → Summarizes user health, predictions, and recommendations.

🔹 Healthcare Support

Hospital Locator → Finds nearby hospitals with name, distance, and map path.

Personalized Recommendations → Diet, exercise, and lifestyle suggestions based on results.
