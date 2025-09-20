import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from xgboost import XGBClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os


df = pd.read_csv("obesity.csv")
print(df.shape)

diseases = ["Diabetes_Risk", "Hypertension_Risk", "Heart_Disease_Risk"]
for disease in diseases:
    if disease not in df.columns:
        df[disease] = "Unknown"


label_encoders = {}
for col in df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le


X = df.drop(columns=["NObeyesdad"] + diseases)
y = df["NObeyesdad"]
y_diseases = df[diseases]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Classifier models
models = [
    RandomForestClassifier(n_estimators=100),
    GradientBoostingClassifier(n_estimators=100),
    AdaBoostClassifier(n_estimators=100),
    XGBClassifier(eval_metric='mlogloss')
]
names = ["Random Forest", "Gradient Boost", "AdaBoost", "XGBoost"]

scores = []
reports = {}
conf_matrices = {}

# Plot confusion matrices
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for i, model in enumerate(models):
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    score = accuracy_score(y_test, y_pred)
    scores.append(score * 100)
    reports[names[i]] = classification_report(y_test, y_pred, output_dict=True)

    cm = confusion_matrix(y_test, y_pred)
    conf_matrices[names[i]] = cm

    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[i])
    axes[i].set_title(f"{names[i]}: {score*100:.2f}%")
    axes[i].set_xlabel("Predicted")
    axes[i].set_ylabel("Actual")

plt.tight_layout()
plt.show()

# Accuracy comparison chart
results_df = pd.DataFrame({"Model": names, "Accuracy": scores})
results_df = results_df.sort_values("Accuracy", ascending=False)
results_df["Accuracy"] = results_df["Accuracy"].round(2)

plt.figure(figsize=(8, 6))
sns.barplot(data=results_df, x="Model", y="Accuracy", palette="viridis")
plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy (%)")
plt.xlabel("ML Model")
plt.xticks(rotation=45)
for i, acc in enumerate(results_df["Accuracy"]):
    plt.text(i, acc + 0.2, f"{acc:.2f}%", ha='center')
plt.tight_layout()
plt.show()

# PCA visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_train_scaled)
plt.figure(figsize=(6,6))
plt.scatter(X_pca[:,0], X_pca[:,1], c=y_train, cmap='viridis', s=10)
plt.title("PCA Visualization of Training Data")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.show()

# t-SNE visualization
tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, random_state=42)
X_tsne = tsne.fit_transform(X_train_scaled)
plt.figure(figsize=(6,6))
plt.scatter(X_tsne[:,0], X_tsne[:,1], c=y_train, cmap='plasma', s=10)
plt.title("t-SNE Visualization of Training Data")
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.show()


os.makedirs("models", exist_ok=True)

# Save best obesity model (top accuracy model)
best_model_index = np.argmax(scores)
joblib.dump(models[best_model_index], "models/obesity_model.pkl")

# Save scaler and encoders
joblib.dump(scaler, "models/scaler.pkl")
joblib.dump(label_encoders, "models/label_encoders.pkl")

# === 🩺 Multi-disease Prediction ===
X_full_scaled = scaler.transform(X)
X_train_d, X_test_d, y_train_d, y_test_d = train_test_split(X_full_scaled, y_diseases, test_size=0.2, random_state=42)

multi_disease_model = MultiOutputClassifier(RandomForestClassifier(n_estimators=100, random_state=42))
multi_disease_model.fit(X_train_d, y_train_d)
y_pred_diseases = multi_disease_model.predict(X_test_d)

# Accuracy for each disease
print("\n🔬 Multi-Disease Model Accuracy:")
for i, disease in enumerate(diseases):
    acc = accuracy_score(y_test_d.iloc[:, i], y_pred_diseases[:, i])
    print(f"{disease}: {acc:.2f}")

joblib.dump(multi_disease_model, "models/multi_disease_model.pkl")

print("\n✅ All models saved successfully!")
