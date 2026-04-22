import streamlit as st # type: ignore
import pandas as pd # type: ignore
import numpy as np # type: ignore
import pickle

# ══════════════════════════════════════════════════════════
# PAGE CONFIG — must be first streamlit command
# ══════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Job Application Screener",
    page_icon="💼",
    layout="wide"
)

# ══════════════════════════════════════════════════════════
# CUSTOM CSS — makes it look professional
# ══════════════════════════════════════════════════════════

st.markdown("""
<style>
    .stApp { background-color: #f0f2f6 !important; }

    /* Force all text dark */
    .stApp, .stApp p, .stApp div, .stApp span, .stApp label {
        color: #1e293b !important;
    }

    [data-testid="stMetric"] {
        background: white;
        border-radius: 12px;
        padding: 16px;
        border: 1px solid #e2e8f0;
    }
    [data-testid="stMetricValue"] { color: #1e293b !important; }
    [data-testid="stMetricLabel"] { color: #64748b !important; }
    [data-testid="stMetricDelta"] { color: #16a34a !important; }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
            
            /* Fix dropdown selected value color */
[data-baseweb="select"] span,
[data-baseweb="select"] div,
[data-baseweb="tag"] span,
div[data-baseweb="select"] > div {
    color: #1e293b !important;
    background-color: white !important;
}

/* Fix dropdown options list */
[data-baseweb="popover"] li,
[data-baseweb="menu"] li,
[data-baseweb="option"] {
    color: #1e293b !important;
    background-color: white !important;
}

[data-baseweb="option"]:hover {
    background-color: #f1f5f9 !important;
    color: #1e293b !important;
}

/* Fix number input text */
input[type="number"],
input[type="text"] {
    color: #1e293b !important;
    background-color: white !important;
}

/* Fix selectbox container */
.stSelectbox > div > div {
    background-color: white !important;
    color: #1e293b !important;
}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
# LOAD MODEL — cached so it only loads once
# ══════════════════════════════════════════════════════════

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

# WHY @st.cache_resource?
# Without it, Streamlit reloads the model on EVERY interaction.
# With it, model loads once and stays in memory.
# For larger models this saves seconds per click.

model, encoders, feature_names, model_name = load_all()

# ══════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════

st.markdown("## 💼 Job Application Screener")
st.markdown(f"Predicts placement chances using **{model_name}** · Trained on 215 real MBA placement records · Accuracy **88.4%**")
st.divider()


# ══════════════════════════════════════════════════════════
# TWO COLUMN LAYOUT — inputs left, results right
# ══════════════════════════════════════════════════════════

left_col, right_col = st.columns([1, 1], gap="large")

# ══════════════════════════════════════════════════════════
# LEFT — INPUT FORM
# ══════════════════════════════════════════════════════════

with left_col:
    st.markdown("### 📋 Your Profile")

    # ── Academic scores ──────────────────────────────────
    st.markdown("**Academic scores**")

    col1, col2 = st.columns(2)
    with col1:
        ssc_percentage  = st.number_input("SSC (10th) %",   min_value=40.0, max_value=100.0, value=70.0, step=0.1)
        degree_percentage = st.number_input("Degree %",     min_value=40.0, max_value=100.0, value=70.0, step=0.1)
    with col2:
        hsc_percentage  = st.number_input("HSC (12th) %",   min_value=40.0, max_value=100.0, value=70.0, step=0.1)
        mba_percent     = st.number_input("MBA %",          min_value=40.0, max_value=100.0, value=70.0, step=0.1)

    emp_test_percentage = st.slider("Employability Test %", min_value=40.0, max_value=100.0, value=70.0, step=0.5)

    st.markdown("---")

    # ── Background ────────────────────────────────────────
    st.markdown("**Background**")

    col3, col4 = st.columns(2)
    with col3:
        gender         = st.selectbox("Gender",          ["M", "F"])
        hsc_subject    = st.selectbox("HSC Stream",      ["Science", "Commerce", "Arts"])
        undergrad_degree = st.selectbox("Degree Type",   ["Sci&Tech", "Comm&Mgmt", "Others"])
    with col4:
        work_experience = st.selectbox("Work Experience", ["No", "Yes"])
        specialisation  = st.selectbox("MBA Specialisation", ["Mkt&Fin", "Mkt&HR"])
        ssc_board       = st.selectbox("SSC Board",       ["Central", "Others"])

    hsc_board = st.selectbox("HSC Board", ["Central", "Others"])

    st.markdown("---")

    predict_btn = st.button("🔍 Check My Placement Chances", type="primary", use_container_width=True)

# ══════════════════════════════════════════════════════════
# RIGHT — RESULTS
# ══════════════════════════════════════════════════════════

with right_col:
    st.markdown("### 📊 Your Results")

    if not predict_btn:
        # Placeholder before user clicks
        st.info("👈 Fill in your profile and click **Check My Placement Chances**")

        # Show model stats as context
        st.markdown("**About this model**")
        m1, m2, m3 = st.columns(3)
        m1.metric("Accuracy",  "88.4%")
        m2.metric("Precision", "90.6%")
        m3.metric("Recall",    "93.6%")

        st.caption(f"Algorithm: {model_name} · Dataset: 215 MBA students · Test set: 43 rows")

    else:
        # ── BUILD INPUT ───────────────────────────────────
        input_dict = {
            'gender'             : gender,
            'ssc_percentage'     : ssc_percentage,
            'ssc_board'          : ssc_board,
            'hsc_percentage'     : hsc_percentage,
            'hsc_board'          : hsc_board,
            'hsc_subject'        : hsc_subject,
            'degree_percentage'  : degree_percentage,
            'undergrad_degree'   : undergrad_degree,
            'work_experience'    : work_experience,
            'emp_test_percentage': emp_test_percentage,
            'specialisation'     : specialisation,
            'mba_percent'        : mba_percent
        }

        # Encode text columns using saved encoders
        for col, val in input_dict.items():
            if col in encoders:
                try:
                    input_dict[col] = int(encoders[col].transform([val])[0])
                except:
                    input_dict[col] = 0

        # Build dataframe in exact column order model expects
        input_df = pd.DataFrame([input_dict])[feature_names]

        # ── PREDICT ───────────────────────────────────────
        prediction  = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0]

        placed_pct     = round(float(probability[1]) * 100, 1)
        not_placed_pct = round(float(probability[0]) * 100, 1)

        # ── RESULT BANNER ─────────────────────────────────
        if prediction == 1:
            st.success(f"✅  Likely to be PLACED")
        else:
            st.error(f"❌  Risk of NOT being placed")

        # ── PROBABILITY GAUGE ─────────────────────────────
        st.markdown("**Placement probability**")

        col_a, col_b = st.columns(2)
        col_a.metric("✅ Placed",     f"{placed_pct}%",
                     delta=f"{placed_pct - 50:.1f}% vs 50/50")
        col_b.metric("❌ Not Placed", f"{not_placed_pct}%")

        # Visual probability bar
        bar_color = "#16a34a" if prediction == 1 else "#dc2626"
        st.markdown(f"""
        <div style="background:#e2e8f0; border-radius:8px; height:20px; margin:8px 0 16px;">
            <div style="background:{bar_color}; width:{placed_pct}%;
                        height:20px; border-radius:8px;
                        display:flex; align-items:center;
                        padding-left:8px; color:white; font-size:12px; font-weight:600;">
                {placed_pct}%
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.divider()

        # ── SCORE BREAKDOWN ───────────────────────────────
        st.markdown("**Your score breakdown**")

        scores = {
            "SSC (10th)":       ssc_percentage,
            "HSC (12th)":       hsc_percentage,
            "Degree":           degree_percentage,
            "MBA":              mba_percent,
            "Employability Test": emp_test_percentage,
        }

        for label, score in scores.items():
            color = "#16a34a" if score >= 70 else ("#f59e0b" if score >= 60 else "#dc2626")
            icon  = "✅" if score >= 70 else ("⚠️" if score >= 60 else "❌")
            st.markdown(f"""
            <div style="display:flex; justify-content:space-between;
                        align-items:center; padding:6px 0;
                        border-bottom:1px solid #f1f5f9;">
                <span style="font-size:14px;">{icon} {label}</span>
                <span style="font-weight:600; color:{color}; font-size:14px;">{score}%</span>
            </div>
            """, unsafe_allow_html=True)

        st.divider()

        # ── PERSONALISED TIPS ─────────────────────────────
        st.markdown("**💡 How to improve your chances**")

        tips = []
        if ssc_percentage  < 60: tips.append(("📚", "SSC percentage is below 60. Mention strong projects to compensate."))
        if hsc_percentage  < 60: tips.append(("📚", "HSC percentage is below 60. Highlight extracurriculars and skills."))
        if degree_percentage < 60: tips.append(("🎓", "Degree percentage is low. Build strong project portfolio instead."))
        if mba_percent     < 60: tips.append(("🎓", "MBA percentage is below 60. Focus on your specialisation domain."))
        if emp_test_percentage < 60: tips.append(("📝", "Employability test score is low. Practice aptitude tests on IndiaBIX."))
        if work_experience == "No": tips.append(("💼", "No work experience. Even a 1-month internship changes this significantly."))

        if tips:
            for icon, tip in tips:
                st.markdown(f"""
                <div style="background:#fefce8; border-left:3px solid #f59e0b;
                            border-radius:6px; padding:10px 12px; margin-bottom:8px;
                            font-size:14px;">
                    {icon} {tip}
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="background:#f0fdf4; border-left:3px solid #16a34a;
                        border-radius:6px; padding:10px 12px; font-size:14px;">
                🌟 Strong profile across all areas. Keep it up!
            </div>
            """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════════

st.divider()
st.caption("Built with Streamlit · Logistic Regression · Scikit-learn · Dataset: MBA Placement Records")