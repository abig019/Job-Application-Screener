import os
import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Auto-train if model doesn't exist on Streamlit Cloud
if not os.path.exists('model.pkl'):
    import subprocess
    subprocess.run(['python', 'train.py'])

@st.cache_resource
def load_all():
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('encoders.pkl', 'rb') as f:
        encoders = pickle.load(f)
    with open('feature_names.pkl', 'rb') as f:
        features = pickle.load(f)
    with open('model_name.pkl', 'rb') as f:
        model_name = pickle.load(f)
    return model, encoders, features, model_name

model, encoders, feature_names, model_name = load_all()

st.title("💼 Job Application Screener")
st.markdown("Enter your profile and find out your placement chances!")
st.caption(f"Powered by {model_name}")
st.divider()

col1, col2 = st.columns(2)

with col1:
    gender     = st.selectbox("👤 Gender", ['M', 'F'])
    ssc_p      = st.slider("📚 SSC (10th) %",    40.0, 100.0, 70.0)
    hsc_p      = st.slider("📚 HSC (12th) %",    40.0, 100.0, 70.0)
    hsc_s      = st.selectbox("🏫 HSC Stream", ['Commerce', 'Science', 'Arts'])

with col2:
    degree_p   = st.slider("🎓 Degree %",         40.0, 100.0, 70.0)
    degree_t   = st.selectbox("📖 Degree Type",   ['Sci&Tech', 'Comm&Mgmt', 'Others'])
    workex     = st.selectbox("💼 Work Experience", ['No', 'Yes'])
    etest_p    = st.slider("📝 Employability Test %", 40.0, 100.0, 70.0)

col3, _ = st.columns(2)
with col3:
    specialisation = st.selectbox("🎯 MBA Specialisation", ['Mkt&HR', 'Mkt&Fin'])
    mba_p          = st.slider("🎓 MBA %", 40.0, 100.0, 70.0)

st.divider()

if st.button("🔍 Check My Placement Chances", type="primary"):

    input_dict = {
        'gender'         : gender,
        'ssc_percentage' : ssc_p,
        'ssc_board'      : 'Central',
        'hsc_percentage' : hsc_p,
        'hsc_board'      : 'Central',
        'hsc_subject'    : hsc_s,
        'degree_percentage' : degree_p,
        'undergrad_degree'  : degree_t,
        'work_experience'   : workex,
        'emp_test_percentage': etest_p,
        'specialisation'    : specialisation,
        'mba_percent'       : mba_p
    }

    # Encode text columns
    for col, val in input_dict.items():
        if col in encoders:
            try:
                input_dict[col] = encoders[col].transform([val])[0]
            except:
                input_dict[col] = 0

    input_df    = pd.DataFrame([input_dict])[feature_names]
    prediction  = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0]

    placed_prob     = round(probability[1] * 100, 1)
    not_placed_prob = round(probability[0] * 100, 1)

    if prediction == 1:
        st.success(f"✅ Likely to be PLACED!  Confidence: {placed_prob}%")
    else:
        st.error(f"❌ Risk of NOT being placed.  Confidence: {not_placed_prob}%")

    col_a, col_b = st.columns(2)
    col_a.metric("✅ Placed",     f"{placed_prob}%")
    col_b.metric("❌ Not Placed", f"{not_placed_prob}%")

    st.subheader("💡 How to improve your chances")
    tips = []
    if ssc_p < 60:
        tips.append("📚 SSC percentage is low. Strong academics matter to recruiters.")
    if hsc_p < 60:
        tips.append("📚 HSC percentage is low. Highlight other strengths.")
    if degree_p < 60:
        tips.append("🎓 Degree percentage is low. Focus on projects and certifications.")
    if workex == 'No':
        tips.append("💼 No work experience. Try internships — even 1 month helps.")
    if etest_p < 60:
        tips.append("📝 Employability test score is low. Practice aptitude tests online.")
    if mba_p < 60:
        tips.append("🎓 MBA percentage is low. Focus on your specialisation skills.")

    if tips:
        for tip in tips:
            st.info(tip)
    else:
        st.success("🌟 Strong profile! Keep it up.")

    st.subheader("📊 What matters most for placement?")
    if hasattr(model, 'feature_importances_'):
        imp_df = pd.DataFrame({
            'Factor'    : feature_names,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
        st.bar_chart(imp_df.set_index('Factor'))