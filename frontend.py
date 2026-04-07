
import streamlit as st
import pickle
import numpy as np

# ---------------- LOAD MODEL ---------------- #
model = pickle.load(open('model.pkl', 'rb'))

# ---------------- UI ---------------- #
st.set_page_config(page_title="Titanic Survival Predictor")

st.title("🚢 Titanic Survival Prediction")
st.write("Enter passenger details to predict survival")


st.sidebar.header("📌 About This App")

st.sidebar.markdown("""
This app predicts whether a passenger would survive the Titanic disaster using Machine Learning.

### 🔍 Model Used
- Random Forest Classifier

### 📊 Input Features
- Passenger Class  
- Sex  
- Age  
- Fare  (amount of money a passenger paid for their ticket)
- Family Size  
- Embarked Location  (port where the passenger boarded -C → Cherbourg (France) , Q → Queenstown (Ireland) , S → Southampton (England)

### ⚙️ Key Logic
- `IsAlone` is automatically derived from Family Size  
- Inputs are validated to ensure realistic values  

### 📁 Dataset
- Titanic dataset (Kaggle)

---

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

# ---------------- BUTTON ---------------- #

predict_clicked = st.button("Predict")

if predict_clicked:

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

# ---------------- FOOTER ---------------- #

st.markdown("---")
st.caption("Model trained on Titanic dataset with input validation.")