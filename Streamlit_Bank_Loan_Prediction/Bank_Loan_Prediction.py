import streamlit as st
from PIL import Image
import pickle
from sklearn.preprocessing import StandardScaler

# Load the model
model = pickle.load(open('./Model/ML_Model.pkl', 'rb'))

# Initialize the scaler
scaler = StandardScaler()

def run():
    img1 = Image.open('bank.png')
    img1 = img1.resize((156, 145))
    st.image(img1, use_column_width=False)
    st.title("Bank Loan Prediction using Machine Learning")

    # Input Fields
    account_no = st.text_input('Account Number')
    if len(account_no) != 13 or not account_no.isdigit():
        st.warning("Account number must be a 13-digit numeric value.")
        st.stop()

    fn = st.text_input('Full Name')

    # Dropdown inputs
    gen = st.selectbox("Gender", ['Female', 'Male'])
    mar = st.selectbox("Marital Status", ['No', 'Yes'])
    dep = st.selectbox("Dependents", ['No', 'One', 'Two', 'More than Two'])
    edu = st.selectbox("Education", ['Not Graduate', 'Graduate'])
    emp = st.selectbox("Employment Status", ['Job', 'Business'])
    prop = st.selectbox("Property Area", ['Rural', 'Semi-Urban', 'Urban'])
    cred = st.selectbox("Credit Score", ['Between 300 to 500', 'Above 500'])
    
    # Numerical inputs
    mon_income = st.number_input("Applicant's Monthly Income($)", min_value=0)
    co_mon_income = st.number_input("Co-Applicant's Monthly Income($)", min_value=0)
    loan_amt = st.number_input("Loan Amount", min_value=0)

    # Loan Duration
    dur_display = ['2 Month', '6 Month', '8 Month', '1 Year', '16 Month']
    dur = st.selectbox("Loan Duration", dur_display)
    duration = {'2 Month': 60, '6 Month': 180, '8 Month': 240, '1 Year': 360, '16 Month': 480}[dur]

    # Prediction Logic
    if st.button("Submit"):
        # Encoding categorical features
        gen = 0 if gen == 'Female' else 1
        mar = 0 if mar == 'No' else 1
        dep_map = {'No': 0, 'One': 1, 'Two': 2, 'More than Two': 3}
        dep = dep_map[dep]
        edu = 0 if edu == 'Not Graduate' else 1
        emp = 0 if emp == 'Job' else 1
        prop_map = {'Rural': 0, 'Semi-Urban': 1, 'Urban': 2}
        prop = prop_map[prop]
        cred = 0 if cred == 'Between 300 to 500' else 1

        # Feature scaling for numerical inputs
        scaled_values = scaler.fit_transform([[mon_income, co_mon_income, loan_amt]])[0]

        # Feature list
        features = [[gen, mar, dep, edu, emp, scaled_values[0], scaled_values[1], scaled_values[2], duration, cred, prop]]
        
        # Predict with probability threshold
        proba = model.predict_proba(features)[0][1]

        if proba > 0.4:  # Adjust threshold as needed
            st.success(f"Hello: {fn} || Account number: {account_no} || Congratulations! You will get the loan from the bank.")
        else:
            st.error(f"Hello: {fn} || Account number: {account_no} || According to our calculations, you will not get the loan from the bank.")

run()
