import streamlit as st
import pickle
import pandas as pd
import os
import zipfile
import io

# Load the model: support model.zip (containing a .pkl) or model.pkl
if os.path.exists('model.zip'):
    with zipfile.ZipFile('model.zip') as z:
        pkl_files = [n for n in z.namelist() if n.endswith('.pkl')]
        if not pkl_files:
            raise FileNotFoundError('No .pkl file found inside model.zip')
        # read first .pkl found
        model_bytes = z.read(pkl_files[0])
        model_data = pickle.loads(model_bytes)
elif os.path.exists('model.pkl'):
    with open('model.pkl', 'rb') as f:
        model_data = pickle.load(f)
else:
    raise FileNotFoundError('No model.zip or model.pkl found in the project directory')

if isinstance(model_data, dict) and 'model' in model_data:
    model = model_data['model']
    features = model_data.get('features')
else:
    model = model_data
    features = None

st.title("💳 Loan-Lens: Smart Approval System")
st.write("Analyze loan eligibility based on Credit Score, Income, and Profile.")

# --- INPUT SECTION ---
col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", 18, 100, 30)
    gender_txt = st.selectbox("Gender", ["Male", "Female", "Other"])
    marital_txt = st.selectbox("Marital Status", ["Married", "Single"])
    edu_txt = st.selectbox("Education", ["Master's", "Bachelor's", "High School", "Other"])

with col2:
    income = st.number_input("Annual Income ($)", value=40000)
    credit_score = st.slider("Credit Score", 300, 850, 650) # Typical range 300-850
    loan_amount = st.number_input("Loan Amount Requested ($)", value=15000)
    employ_txt = st.selectbox("Employment", ["Employed", "Self-employed", "Unemployed"])

# --- PREPARE INPUTS ---
# Keep text values for one-hot reconstruction; also create fallback numeric codes
gender = {'Male': 1, 'Female': 0, 'Other': 2}[gender_txt]  # fallback if model expects numeric
marital = {'Married': 1, 'Single': 0}.get(marital_txt, 0)
education = {"Master's": 3, "Bachelor's": 2, "High School": 1, "Other": 0}.get(edu_txt, 0)
employment = {'Employed': 2, 'Self-employed': 1, 'Unemployed': 0}.get(employ_txt, 0)  # fallback numeric code

# --- PREDICTION ---
if st.button("Analyze Application"):
    # Build a single-row DataFrame aligned with training features (if available)
    if features:
        input_df = pd.DataFrame(0, index=[0], columns=features)
        # set numeric features if present
        for col, val in {'age': age, 'annual_income': income, 'credit_score': credit_score, 'loan_amount': loan_amount}.items():
            if col in input_df.columns:
                input_df.at[0, col] = val
        # set categorical dummies (pandas.get_dummies used 'col_val' naming)
        cat_map = {'gender': gender_txt, 'marital_status': marital_txt, 'education_level': edu_txt, 'employment_status': employ_txt}
        for k, v in cat_map.items():
            col_name = f"{k}_{v}"
            if col_name in input_df.columns:
                input_df.at[0, col_name] = 1
        # predict
        try:
            prediction = model.predict(input_df)
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.write("Model features:", features)
            raise
    else:
        # Fallback: pass a simple list of numeric features as before
        input_data = [[age, gender, marital, education, income, employment, credit_score, loan_amount]]
        prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.success("✅ APPROVED")
        st.balloons()
    else:
        st.error("❌ REJECTED")

    # --- EXPLAINABILITY (The Winning Feature) ---
    st.subheader("💡 Analysis Report")
    if credit_score < 650:
        st.warning(f"⚠️ **Credit Score Issue:** Your score ({credit_score}) is below the recommended 650.")
    if employ_txt == "Unemployed":
        st.warning("⚠️ **Employment Status:** Unemployed status significantly lowers approval odds.")
    if loan_amount > (income * 0.5):
        st.warning("⚠️ **High Debt Risk:** You are asking for more than 50% of your annual income.")