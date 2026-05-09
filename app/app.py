
# ================================================================
#//* Heart Attack Expert System — Streamlit Dashboard
#//*   Full UI/UX overhaul with:
#//*     - Dynamic Height/Weight/BMI inputs
#//*     - Plotly Speedometer Gauge for risk visualisation
#//*     - Tabbed categorised layout (Personal Info / Medical History / Lifestyle)
#//*     - What-If Simulation (smoking cessation + BMI reduction)
#//*     - Clinical Insights engine based on probability tier
#//*   Architecture: AdamW + Swish preserved in expert_model.h5
# ================================================================


import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import plotly.graph_objects as go
import joblib
import os
from inference_mapper import load_schema, prepare_patient_vector

# ─────────────────────────────────────────────────────────────
# 1. Page Configuration
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Heart Attack Expert System",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─────────────────────────────────────────────────────────────
# 2. Custom CSS — Premium Dark Theme
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp { background: linear-gradient(135deg, #0d1117 0%, #161b22 50%, #0d1117 100%); }

.hero-card {
    background: linear-gradient(135deg, rgba(220,38,38,0.15), rgba(239,68,68,0.05));
    border: 1px solid rgba(220,38,38,0.3);
    border-radius: 16px;
    padding: 2rem;
    margin-bottom: 1.5rem;
}
.bmi-card {
    background: linear-gradient(135deg, rgba(99,102,241,0.15), rgba(99,102,241,0.05));
    border: 1px solid rgba(99,102,241,0.3);
    border-radius: 12px;
    padding: 1rem 1.5rem;
    text-align: center;
    margin-top: 0.5rem;
}
.insight-card {
    border-radius: 12px;
    padding: 1rem 1.2rem;
    margin: 0.4rem 0;
}
.insight-high  { background: rgba(220,38,38,0.12);  border-left: 4px solid #dc2626; }
.insight-med   { background: rgba(234,179,8,0.12);  border-left: 4px solid #eab308; }
.insight-low   { background: rgba(34,197,94,0.12);  border-left: 4px solid #22c55e; }
.whatif-card {
    background: rgba(99,102,241,0.1);
    border: 1px solid rgba(99,102,241,0.25);
    border-radius: 12px;
    padding: 1rem 1.4rem;
    margin: 0.5rem 0;
}
.section-label {
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #6b7280;
    margin-bottom: 0.25rem;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# 3. Resource Loading (cached)
# ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_resources():
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path  = os.path.join(BASE_DIR, 'models', 'expert_model.h5')
    scaler_path = os.path.join(BASE_DIR, 'models', 'scaler.joblib')
    schema_path = os.path.join(BASE_DIR, 'models', 'model_schema.json')
    try:
        #//! compile=False preserves AdamW + Swish activations without
        #//! re-compiling; avoids optimizer state mismatch on load.
        model  = tf.keras.models.load_model(model_path, compile=False)
        scaler = joblib.load(scaler_path)
        schema = load_schema(schema_path)
        return None, model, scaler, schema
    except Exception as e:
        return f"Error loading resources: {str(e)}", None, None, None

error_msg, expert_model, scaler, schema = load_resources()

if error_msg:
    st.error(error_msg)
    st.stop()

# ─────────────────────────────────────────────────────────────
# 4. Helper — Plotly Speedometer Gauge
# ─────────────────────────────────────────────────────────────
def make_gauge(prob_pct: float) -> go.Figure:
    """
    Renders a Plotly speedometer gauge for the predicted risk percentage.
    Color bands:  0–30 % → Green | 31–60 % → Yellow | 61–100 % → Red
    """
    #//? Gauge steps are defined in reversed order so Plotly renders correctly.
    if prob_pct <= 30:
        needle_color = "#22c55e"
        risk_label   = "LOW RISK"
    elif prob_pct <= 60:
        needle_color = "#eab308"
        risk_label   = "MODERATE RISK"
    else:
        needle_color = "#ef4444"
        risk_label   = "HIGH RISK"

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=prob_pct,
        number={"suffix": "%", "font": {"size": 38, "color": needle_color, "family": "Inter"}},
        title={"text": f"<b>{risk_label}</b>", "font": {"size": 16, "color": "#e5e7eb"}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#374151",
                     "tickfont": {"color": "#9ca3af"}},
            "bar": {"color": needle_color, "thickness": 0.25},
            "bgcolor": "rgba(0,0,0,0)",
            "borderwidth": 0,
            "steps": [
                {"range": [0,  30],  "color": "rgba(34,197,94,0.18)"},
                {"range": [30, 60],  "color": "rgba(234,179,8,0.18)"},
                {"range": [60, 100], "color": "rgba(239,68,68,0.18)"},
            ],
            "threshold": {
                "line": {"color": needle_color, "width": 4},
                "thickness": 0.8,
                "value": prob_pct,
            },
        },
    ))
    fig.update_layout(
        height=280,
        margin=dict(l=20, r=20, t=40, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        font={"family": "Inter"},
    )
    return fig

# ─────────────────────────────────────────────────────────────
# 5. Helper — What-If Simulation
# ─────────────────────────────────────────────────────────────
def run_whatif(base_inputs: dict, schema, scaler, model) -> dict:
    """
    Simulates risk reduction for two counterfactual scenarios:
      - Scenario A: SmokerStatus changed to 'Never smoked'
      - Scenario B: BMI reduced to 25 (healthy range)
    Returns a dict with original probability and both scenario probabilities.
    """
    def _predict(inputs):
        vec, _ = prepare_patient_vector(inputs, schema, scaler)
        return float(model.predict(vec, verbose=0)[0][0]) * 100

    original = _predict(base_inputs)

    # Scenario A — Smoking cessation (ordinal 0 = 'Never smoked')
    inputs_a = base_inputs.copy()
    inputs_a['SmokerStatus'] = 0
    prob_a = _predict(inputs_a)

    # Scenario B — BMI reduction to 25
    inputs_b = base_inputs.copy()
    inputs_b['BMI'] = 25.0
    inputs_b['WeightInKilograms'] = 25.0 * (base_inputs.get('HeightInMeters', 1.70) ** 2)
    prob_b = _predict(inputs_b)

    return {"original": original, "no_smoke": prob_a, "bmi25": prob_b}

# ─────────────────────────────────────────────────────────────
# 6. Helper — Clinical Insights
# ─────────────────────────────────────────────────────────────
def get_clinical_insights(prob: float, inputs: dict) -> list[dict]:
    """
    Returns tier-appropriate clinical advice based on predicted probability
    and specific patient inputs (angina, sleep, smoker, BMI).
    Each item: {"icon": str, "text": str, "tier": "high"|"med"|"low"}
    """
    tips = []

    if prob > 0.60:
        tips.append({"icon": "🫀", "tier": "high",
                     "text": "Consult a Cardiologist immediately for a full cardiac evaluation."})
        tips.append({"icon": "🚑", "tier": "high",
                     "text": "Consider an ECG and stress test — do not delay."})
        if inputs.get("HadAngina") == "Yes":
            tips.append({"icon": "💊", "tier": "high",
                         "text": "Angina detected — discuss nitrate therapy with your physician."})

    elif prob > 0.30:
        tips.append({"icon": "🩺", "tier": "med",
                     "text": "Schedule a preventive cardiology check-up within 3 months."})
        tips.append({"icon": "🏃", "tier": "med",
                     "text": "Aim for 150 min/week of moderate aerobic exercise."})

    else:
        tips.append({"icon": "✅", "tier": "low",
                     "text": "Risk is low. Maintain your healthy habits!"})
        tips.append({"icon": "🥗", "tier": "low",
                     "text": "Continue a heart-healthy diet rich in omega-3 fatty acids."})

    # Sleep-specific
    sleep = inputs.get("SleepHours", 7)
    if float(sleep) < 6:
        tips.append({"icon": "😴", "tier": "med",
                     "text": f"You sleep only {sleep}h/night. Target 7–9 hours to reduce cardiac risk."})

    # Smoking-specific — SmokerStatus is now an ordinal int (0=never, 1=former, 2=some days, 3=every day)
    smoker_ordinal = int(inputs.get("SmokerStatus", 0))
    if smoker_ordinal >= 2:
        tips.append({"icon": "🚭", "tier": "high",
                     "text": "Active smoker detected — quitting can cut cardiac risk by up to 50% within 1 year."})

    # BMI-specific
    bmi_val = float(inputs.get("BMI", 25))
    if bmi_val >= 30:
        tips.append({"icon": "⚖️", "tier": "med",
                     "text": f"BMI {bmi_val:.1f} indicates obesity. A 5% weight loss can improve cardiac metrics."})
    elif bmi_val >= 25:
        tips.append({"icon": "⚖️", "tier": "low",
                     "text": f"BMI {bmi_val:.1f} is slightly elevated. Small lifestyle tweaks can bring it under 25."})

    return tips

# ─────────────────────────────────────────────────────────────
# 7. Header
# ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-card">
    <h1 style="margin:0;color:#f9fafb;font-size:1.9rem;">❤️ Heart Attack Expert System</h1>
    <p style="margin:0.4rem 0 0;color:#9ca3af;font-size:0.95rem;">
        AI-powered cardiac risk assessment using a Deep Neural Network (Swish · AdamW · SMOTE)
    </p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# 8. Tabbed Input Layout
# ─────────────────────────────────────────────────────────────
tab_personal, tab_medical, tab_lifestyle = st.tabs(
    ["👤  Personal Info", "📋  Medical History", "🏃  Lifestyle"]
)

# ── Tab 1: Personal Info ──────────────────────────────────────
with tab_personal:
    c1, c2, c3 = st.columns(3)

    with c1:
        age = st.selectbox("Age Category", [
            "Age 18 to 24", "Age 25 to 29", "Age 30 to 34",
            "Age 35 to 39", "Age 40 to 44", "Age 45 to 49",
            "Age 50 to 54", "Age 55 to 59", "Age 60 to 64",
            "Age 65 to 69", "Age 70 to 74", "Age 75 to 79",
            "Age 80 or older"
        ], index=7, key="age_select")
        sex = st.selectbox("Biological Sex", ["Female", "Male"], key="sex_select")

    with c2:
        #//? Height & Weight are collected independently so all three
        #//? (H, W, BMI) can be passed as model features. BMI is derived
        #//? automatically and shown as read-only feedback.
        height_m = st.slider("Height (meters)", 1.40, 2.20, 1.70, step=0.01,
                             key="height_slider")
        weight_kg = st.slider("Weight (kg)", 30.0, 200.0, 80.0, step=0.5,
                              key="weight_slider")

    with c3:
        bmi_computed = weight_kg / (height_m ** 2)
        if bmi_computed < 18.5:
            bmi_cat, bmi_col = "Underweight", "#60a5fa"
        elif bmi_computed < 25:
            bmi_cat, bmi_col = "Normal", "#22c55e"
        elif bmi_computed < 30:
            bmi_cat, bmi_col = "Overweight", "#eab308"
        else:
            bmi_cat, bmi_col = "Obese", "#ef4444"

        st.markdown(f"""
        <div class="bmi-card">
            <div class="section-label">Calculated BMI</div>
            <div style="font-size:2.4rem;font-weight:700;color:{bmi_col};">{bmi_computed:.1f}</div>
            <div style="color:{bmi_col};font-weight:600;font-size:0.9rem;">{bmi_cat}</div>
            <div style="color:#6b7280;font-size:0.78rem;margin-top:0.3rem;">
                Height {height_m:.2f} m · Weight {weight_kg:.1f} kg
            </div>
        </div>
        """, unsafe_allow_html=True)

        gen_health = st.selectbox("General Health", [
            "Excellent", "Very good", "Good", "Fair", "Poor"
        ], index=1, key="gen_health_select")

# ── Tab 2: Medical History ───────────────────────────────────
with tab_medical:
    m1, m2, m3 = st.columns(3)

    with m1:
        angina       = st.selectbox("Had Angina (Chest Pain)?",     ["No", "Yes"], key="angina")
        stroke       = st.selectbox("Had Stroke?",                   ["No", "Yes"], key="stroke")
        diabetic     = st.selectbox("Diabetic Status", [
            "No",
            "No, pre-diabetes or borderline diabetes",
            "Yes",
            "Yes, but only during pregnancy (female)"
        ], key="diabetic")

    with m2:
        kidney       = st.selectbox("Had Kidney Disease?",           ["No", "Yes"], key="kidney")
        arthritis    = st.selectbox("Had Arthritis?",                 ["No", "Yes"], key="arthritis")

    with m3:
        copd         = st.selectbox("Had COPD?",                      ["No", "Yes"], key="copd")
        depression   = st.selectbox("Had Depressive Disorder?",       ["No", "Yes"], key="depression")
        chest_scan   = st.selectbox("Had Chest CT Scan?",             ["No", "Yes"], key="chest_scan")

    st.divider()
    d1, d2, d3 = st.columns(3)
    with d1:
        diff_walking  = st.selectbox("Difficulty Walking?",           ["No", "Yes"], key="diff_walk")
        diff_dress    = st.selectbox("Difficulty Dressing/Bathing?",  ["No", "Yes"], key="diff_dress")
    with d2:
        diff_conc     = st.selectbox("Difficulty Concentrating?",     ["No", "Yes"], key="diff_conc")
        diff_errands  = st.selectbox("Difficulty Running Errands?",   ["No", "Yes"], key="diff_errands")
    with d3:
        phys_days     = st.slider("Physically Unhealthy Days (last 30)", 0, 30, 0, key="phys_days")
        ment_days     = st.slider("Mentally Unhealthy Days (last 30)",   0, 30, 0, key="ment_days")

# ── Tab 3: Lifestyle ─────────────────────────────────────────
with tab_lifestyle:
    l1, l2, l3 = st.columns(3)

    with l1:
        smoker       = st.selectbox("Smoker Status", [
            "Never smoked",
            "Former smoker",
            "Current smoker - now smokes some days",
            "Current smoker - now smokes every day"
        ], key="smoker")
        ecigarette   = st.selectbox("E-Cigarette Usage", [
            "Never used e-cigarettes in my entire life",
            "Not at all (right now)",
            "Use them some days",
            "Use them every day"
        ], key="ecigarette")

    with l2:
        physical_act = st.selectbox("Physically Active?",             ["Yes", "No"], key="phys_act")
        sleep_hours  = st.slider("Sleep Hours per Night", 1, 24, 7,   key="sleep")

    with l3:
        last_checkup = st.selectbox("Last Medical Checkup", [
            "Within past year (anytime less than 12 months ago)",
            "Within past 2 years (1 year but less than 2 years ago)",
            "Within past 5 years (2 years but less than 5 years ago)",
            "5 or more years ago"
        ], key="checkup")

# ─────────────────────────────────────────────────────────────
# 9. Analysis Trigger  (Reset + Run)
# ─────────────────────────────────────────────────────────────
DEFAULTS = {
    'age_select':        'Age 55 to 59',
    'sex_select':        'Female',
    'height_slider':     1.70,
    'weight_slider':     81.0,
    'gen_health_select': 'Very good',
    'angina':            'No',
    'stroke':            'No',
    'diabetic':          'No',
    'kidney':            'No',
    'arthritis':         'No',
    'copd':              'No',
    'depression':        'No',
    'chest_scan':        'No',
    'diff_walk':         'No',
    'diff_dress':        'No',
    'diff_conc':         'No',
    'diff_errands':      'No',
    'phys_days':         0,
    'ment_days':         0,
    'smoker':            'Never smoked',
    'ecigarette':        'Never used e-cigarettes in my entire life',
    'phys_act':          'Yes',
    'sleep':             7,
    'checkup':           'Within past year (anytime less than 12 months ago)',
}

def reset_all():
    """Writes all widget defaults to session_state before Streamlit re-renders."""
    for key, val in DEFAULTS.items():
        st.session_state[key] = val

st.divider()
reset_col, run_col, _ = st.columns([1, 1, 2])
with reset_col:
    st.button('🔄 Reset All', on_click=reset_all, use_container_width=True, key='reset_btn')
with run_col:
    run_analysis = st.button("🚀 Run AI Analysis", use_container_width=True, key="run_btn")

# ─────────────────────────────────────────────────────────────
# 10. Results Panel
# ─────────────────────────────────────────────────────────────
if run_analysis:

    # Build the raw user inputs dictionary
    user_inputs = {
        'AgeCategory':            age,
        'Sex':                    sex,
        'BMI':                    round(bmi_computed, 2),
        'HeightInMeters':         round(height_m, 2),
        'WeightInKilograms':      round(weight_kg, 1),
        'GeneralHealth':   {'Excellent': 0, 'Very good': 1, 'Good': 2,
                            'Fair': 3, 'Poor': 4}.get(gen_health, 0),
        'SleepHours':             float(sleep_hours),
        'PhysicalHealthDays':     float(phys_days),
        'MentalHealthDays':       float(ment_days),
        # Ordinal-encoded features (0–3 severity scale)
        'SmokerStatus':    {'Never smoked': 0, 'Former smoker': 1,
                            'Current smoker - now smokes some days': 2,
                            'Current smoker - now smokes every day': 3}.get(smoker, 0),
        'ECigaretteUsage': {'Never used e-cigarettes in my entire life': 0,
                            'Not at all (right now)': 1,
                            'Use them some days': 2,
                            'Use them every day': 3}.get(ecigarette, 0),
        'PhysicalActivities':     physical_act,
        'LastCheckupTime':        last_checkup,
        'HadAngina':              angina,
        'HadStroke':              stroke,
        'HadDiabetes':            diabetic,
        'HadKidneyDisease':       kidney,
        'HadArthritis':           arthritis,
        'HadCOPD':                copd,
        'HadDepressiveDisorder':  depression,
        'ChestScan':              chest_scan,
        'DifficultyWalking':      diff_walking,
        'DifficultyDressingBathing': diff_dress,
        'DifficultyConcentrating':   diff_conc,
        'DifficultyErrands':         diff_errands,
    }

    with st.spinner("🧠 Analysing cardiac risk profile..."):
        input_scaled, _ = prepare_patient_vector(user_inputs, schema, scaler)
        prob = float(expert_model.predict(input_scaled, verbose=0)[0][0])
        prob_pct = prob * 100

    # ── Layout: Gauge + Insights ────────────────────────────
    gauge_col, insight_col = st.columns([1, 1])

    with gauge_col:
        st.subheader("📊 Risk Assessment")
        st.plotly_chart(make_gauge(prob_pct), use_container_width=True, key="gauge_chart")

    with insight_col:
        st.subheader("🩺 Clinical Insights")
        tier_css = {"high": "insight-high", "med": "insight-med", "low": "insight-low"}
        for tip in get_clinical_insights(prob, user_inputs):
            css = tier_css.get(tip["tier"], "insight-low")
            st.markdown(
                f'<div class="insight-card {css}">{tip["icon"]} {tip["text"]}</div>',
                unsafe_allow_html=True
            )

    # ── What-If Analysis (only when risk is elevated) ───────
    if prob > 0.30:
        st.divider()
        st.subheader("🔬 Simulate Health Changes")
        st.caption("See how lifestyle modifications could reduce your predicted risk.")

        with st.spinner("Running simulations..."):
            wif = run_whatif(user_inputs, schema, scaler, expert_model)

        wif_col1, wif_col2 = st.columns(2)

        with wif_col1:
            delta_smoke = wif['original'] - wif['no_smoke']
            #//? Positive delta = risk dropped (good). Negative delta = risk increased (bad).
            if delta_smoke > 0:
                smoke_icon, smoke_word, smoke_color = "📉", "drop", "#22c55e"
            else:
                smoke_icon, smoke_word, smoke_color = "📈", "increase", "#ef4444"
            st.markdown(f"""
            <div class="whatif-card">
                <div class="section-label">Scenario A — Stop Smoking</div>
                <div style="font-size:1.5rem;font-weight:700;color:#a78bfa;">
                    {wif['no_smoke']:.1f}%
                </div>
                <div style="color:{smoke_color};font-size:0.85rem;font-weight:600;">
                    {smoke_icon} {abs(delta_smoke):.1f}% {smoke_word} vs current risk ({wif['original']:.1f}%)
                </div>
            </div>
            """, unsafe_allow_html=True)

        with wif_col2:
            delta_bmi = wif['original'] - wif['bmi25']
            if delta_bmi > 0:
                bmi_icon, bmi_word, bmi_color = "📉", "drop", "#22c55e"
            else:
                bmi_icon, bmi_word, bmi_color = "📈", "increase", "#ef4444"
            st.markdown(f"""
            <div class="whatif-card">
                <div class="section-label">Scenario B — Reduce BMI to 25</div>
                <div style="font-size:1.5rem;font-weight:700;color:#a78bfa;">
                    {wif['bmi25']:.1f}%
                </div>
                <div style="color:{bmi_color};font-size:0.85rem;font-weight:600;">
                    {bmi_icon} {abs(delta_bmi):.1f}% {bmi_word} vs current risk ({wif['original']:.1f}%)
                </div>
            </div>
            """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# 11. Footer
# ─────────────────────────────────────────────────────────────
st.divider()
# st.caption("❤️ Heart Attack Expert System · Developed by Mohamed Mustafa AbdelAziz · O6U · 2025")
st.caption("⚕️ *For educational purposes only. Not a substitute for professional medical advice.*")