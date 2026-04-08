# ---------------- MODEL COMPARISON ---------------- #

# st.subheader("📊 Model Comparison")

# comparison_data = {
#     "Model": ["Logistic Regression", "Random Forest", "KNN"],
#     "Accuracy": [0.79, 0.82, 0.82],
#     "Precision": [0.77, 0.79, 0.79],
#     "Recall": [0.71, 0.77, 0.77]
# }

# comparison_df = pd.DataFrame(comparison_data)

# st.dataframe(comparison_df)

import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# ---------------- LOAD MODEL ---------------- #
model = pickle.load(open('model.pkl', 'rb'))

# Load dataset
df = pd.read_csv("titanic.csv")

# ---------------- UI ---------------- #
st.set_page_config(page_title="Titanic Survival Predictor")

st.title("🚢 Titanic Survival Prediction")
st.write("Enter passenger details to predict survival")

# ---------------- SIDEBAR ---------------- #

st.sidebar.header("📌 About This App")

st.sidebar.markdown("""
This app predicts whether a passenger would survive the Titanic disaster using Machine Learning.

### 🔍 Model Used
- Random Forest Classifier

### 📊 Input Features
- Passenger Class  
- Sex  
- Age  
- Fare  
- Family Size  
- Embarked Location  

### ⚙️ Key Logic
- `IsAlone` is automatically derived from Family Size  

👩‍💻 Developed by **Bhumika Yadav**
""")

# ---------------- INPUT ---------------- #

pclass = st.selectbox("Passenger Class", [1, 2, 3])
sex = st.selectbox("Sex", ["male", "female"])
age = st.number_input("Age (0 – 80)", value=25.0)
fare = st.number_input("Fare (0 – 512)", value=50.0)
family_size = st.number_input("Family Size (1 – 11)", value=1)
embarked = st.selectbox("Embarked", ["C", "Q", "S"])

# ---------------- ENCODING ---------------- #

sex_encoded = 1 if sex == "male" else 0
embarked_mapping = {"C": 0, "Q": 1, "S": 2}
embarked_encoded = embarked_mapping[embarked]

# ---------------- DERIVED ---------------- #

is_alone = 1 if family_size == 1 else 0
st.info(f"ℹ️ IsAlone is automatically set to: {is_alone}")

# ---------------- VALIDATION ---------------- #

valid_input = True

if age < 0 or age > 80:
    st.error("❌ Age must be between 0 and 80")
    valid_input = False

if fare < 0 or fare > 512:
    st.error("❌ Fare must be between 0 and 512")
    valid_input = False

if family_size < 1 or family_size > 11:
    st.error("❌ Family Size must be between 1 and 11")
    valid_input = False

# ---------------- PREDICTION ---------------- #

if st.button("Predict"):

    if not valid_input:
        st.warning("⚠️ Please fix input errors before prediction.")

    else:
        input_data = np.array([[pclass,
                                sex_encoded,
                                age,
                                fare,
                                family_size,
                                is_alone,
                                embarked_encoded]])

        prediction = model.predict(input_data)
        prob = model.predict_proba(input_data)[0][1]

        st.subheader("Prediction Result")

        if prediction[0] == 1:
            st.success(f"✅ Survived (Confidence: {round(prob*100,2)}%)")
        else:
            st.error(f"❌ Did Not Survive (Confidence: {round((1-prob)*100,2)}%)")

# ================= ANALYSIS SECTION ================= #

st.markdown("---")
st.header("📊 Model & Data Analysis")

# ---------------- 1. FEATURE IMPORTANCE ---------------- #

st.subheader("🔍 Feature Importance")

features = ['Pclass','Sex_encoded','Age','Fare',
            'FamilySize','IsAlone','Embarked_encoded']

importance_df = pd.DataFrame({
    'Feature': features,
    'Importance': model.feature_importances_
}).sort_values('Importance')

fig, ax = plt.subplots(figsize=(6,3))

ax.barh(importance_df['Feature'], importance_df['Importance'])
ax.set_title("Which features matter most?")
st.pyplot(fig)

st.markdown("""
**Analysis:**
- Features with higher values have more impact on prediction.
- Typically, **Sex, Fare, and Pclass** strongly influence survival.
- This helps us understand how the model makes decisions.
""")

# ---------------- 3. PIVOT HEATMAP ---------------- #

st.subheader("🔥 Survival Rate (Class × Gender)")

pivot = df.pivot_table(values='Survived',
                       index='Pclass',
                       columns='Sex',
                       aggfunc='mean')

fig, ax = plt.subplots(figsize=(5,3.5))

sns.heatmap(pivot, annot=True, fmt='.0%', vmin=0, vmax=1, ax=ax)

st.pyplot(fig)

st.markdown("""
**Analysis:**
- Females had significantly higher survival rates.
- 1st class passengers survived more than 2nd and 3rd.
- 3rd class males had the lowest survival rate.
- Clearly shows social priority during rescue.
""")


# ---------------- 2. CONFUSION MATRIX ---------------- #

from sklearn.model_selection import train_test_split

# Create same features as training
df_cm = df.copy()

df_cm['Sex_encoded'] = df_cm['Sex'].map({'male':1, 'female':0})
df_cm['Embarked_encoded'] = df_cm['Embarked'].map({'C':0,'Q':1,'S':2})

df_cm['FamilySize'] = df_cm['SibSp'] + df_cm['Parch'] + 1
df_cm['IsAlone'] = (df_cm['FamilySize'] == 1).astype(int)

# Use SAME features as model
X = df_cm[['Pclass','Sex_encoded','Age','Fare',
           'FamilySize','IsAlone','Embarked_encoded']].fillna(0)

y = df_cm['Survived']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Predict
y_pred = model.predict(X_test)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots(figsize=(4,3))

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Predicted: Died','Predicted: Survived'],
            yticklabels=['Actual: Died','Actual: Survived'], ax=ax)

st.pyplot(fig)

st.markdown("""
**Analysis:**
- Shows how many predictions were correct vs incorrect.
- Diagonal values = correct predictions ✅
- Off-diagonal = mistakes ❌
- Helps evaluate model performance beyond accuracy.
""")



# ---------------- FOOTER ---------------- #

st.markdown("---")
st.caption("Model trained on Titanic dataset with input validation.")