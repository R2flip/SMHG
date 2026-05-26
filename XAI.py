# app.py

import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix



st.set_page_config(page_title="Help Seeking Behavior XAI System", layout="wide")

st.title("Help Seeking Behavior Prediction with Explainable AI")


# -----------------------------
# 1. Load dataset
# -----------------------------
@st.cache_data
def load_data():
    data = pd.read_excel(
        r"SMHG20260519_R03.xlsx",
        engine="openpyxl"
    )
    return data


data = load_data()

#st.subheader("Dataset Preview")
#st.dataframe(
#    data.head(10),
#    use_container_width=True,
#    hide_index=True
#)


# -----------------------------
# 2. Prepare data
# -----------------------------
X = data.drop("Real_CH", axis=1)
y = data["Real_CH"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# -----------------------------
# 3. Train model
# -----------------------------
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

st.sidebar.subheader("Model Performance")
st.sidebar.write(f"Accuracy: {accuracy:.2f}")


# -----------------------------
# 4. User Input
# -----------------------------
st.sidebar.subheader("Student Information")

user_input = {}

for col in X.columns:
    min_val = float(X[col].min())
    max_val = float(X[col].max())
    mean_val = float(X[col].mean())

    user_input[col] = st.sidebar.slider(
        col,
        min_value=min_val,
        max_value=max_val,
        value=mean_val
    )

input_df = pd.DataFrame([user_input])


# -----------------------------
# 5. Prediction
# -----------------------------
prediction = model.predict(input_df)[0]
probability = model.predict_proba(input_df)[0][1]

st.subheader("Prediction Result")

if prediction == 1:
    st.error(f"High Probability of Help Seeking Behavior: {probability * 100:.2f}%")
else:
    st.success(f"Low Probability of Help Seeking Behavior: {(1 - probability) * 100:.2f}%")


# -----------------------------
# 6. Feature Influence with SHAP Waterfall Plot
# -----------------------------
st.subheader("Feature Influence (Explainability)")

explainer = shap.TreeExplainer(model)
shap_values = explainer(input_df)

# 取第 1 類：High Help-Seeking Behavior
if shap_values.values.ndim == 3:
    explanation = shap.Explanation(
        values=shap_values.values[0, :, 1],
        base_values=shap_values.base_values[0, 1],
        data=input_df.iloc[0].values,
        feature_names=input_df.columns
    )
else:
    explanation = shap.Explanation(
        values=shap_values.values[0],
        base_values=shap_values.base_values[0],
        data=input_df.iloc[0].values,
        feature_names=input_df.columns
    )

plt.figure(figsize=(9, 6))

shap.plots.waterfall(
    explanation,
    max_display=15,
    show=False
)

st.pyplot(plt.gcf())
plt.clf()

# -----------------------------
# 7. Global Feature Importance
# -----------------------------
st.subheader("Global Feature Importance")

sample_X = X_test.copy()
shap_values_all = explainer(sample_X)

values_all = shap_values_all.values

if values_all.ndim == 3:
    global_shap = values_all[:, :, 1]
elif values_all.ndim == 2:
    global_shap = values_all
else:
    global_shap = values_all.reshape(sample_X.shape)

st.write("SHAP values shape:", global_shap.shape)
st.write("X_test shape:", sample_X.shape)

plt.figure(figsize=(10, 6))

shap.summary_plot(
    global_shap,
    sample_X,
    show=False
)

st.pyplot(plt.gcf())
plt.clf()

# -----------------------------
# 8. ROC Curve and AUC
# -----------------------------
st.subheader("ROC Curve and AUC")

y_prob = model.predict_proba(X_test)[:, 1]

fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

fig3, ax3 = plt.subplots(figsize=(6, 5))

ax3.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
ax3.plot([0, 1], [0, 1], linestyle="--")

ax3.set_xlabel("False Positive Rate")
ax3.set_ylabel("True Positive Rate")
ax3.set_title("ROC Curve")
ax3.legend(loc="lower right")

st.pyplot(fig3)


# -----------------------------
# 9. Confusion Matrix
# -----------------------------
st.subheader("Confusion Matrix")

cm = confusion_matrix(y_test, y_pred)

fig4, ax4 = plt.subplots(figsize=(5, 4))

im = ax4.imshow(cm)

ax4.set_xlabel("Predicted Label")
ax4.set_ylabel("True Label")
ax4.set_title("Confusion Matrix")

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax4.text(j, i, cm[i, j], ha="center", va="center")

st.pyplot(fig4)
