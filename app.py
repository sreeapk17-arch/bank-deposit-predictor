import streamlit as st
import pandas as pd
import joblib
import numpy as np

# ── PAGE CONFIG ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DepositIQ · Bank Term Prediction",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── GLOBAL CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&display=swap');

/* ── Base reset */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* ── App background */
.stApp {
    background: #0a0f1e;
    background-image:
        radial-gradient(ellipse 80% 60% at 50% -10%, rgba(56,189,248,0.12) 0%, transparent 60%),
        radial-gradient(ellipse 50% 40% at 90% 80%, rgba(99,102,241,0.10) 0%, transparent 60%);
}

/* ── Hide Streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 0 !important; max-width: 100% !important; }

/* ── HERO HEADER */
.hero {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 60%, #0f172a 100%);
    border-bottom: 1px solid rgba(56,189,248,0.18);
    padding: 2.4rem 3.5rem 2rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 300px; height: 300px;
    background: radial-gradient(circle, rgba(56,189,248,0.15) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-badge {
    display: inline-block;
    background: rgba(56,189,248,0.12);
    border: 1px solid rgba(56,189,248,0.35);
    color: #38bdf8;
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    padding: 0.3rem 0.9rem;
    border-radius: 999px;
    margin-bottom: 0.85rem;
}
.hero-title {
    font-family: 'DM Serif Display', serif;
    font-size: 2.6rem;
    color: #f1f5f9;
    line-height: 1.15;
    margin: 0 0 0.5rem;
    letter-spacing: -0.02em;
}
.hero-title span { color: #38bdf8; font-style: italic; }
.hero-sub {
    color: #94a3b8;
    font-size: 0.93rem;
    font-weight: 300;
    max-width: 540px;
    line-height: 1.6;
}
.hero-stats {
    display: flex;
    gap: 2rem;
    margin-top: 1.6rem;
}
.hero-stat { text-align: left; }
.hero-stat-val {
    font-family: 'DM Serif Display', serif;
    font-size: 1.5rem;
    color: #f1f5f9;
}
.hero-stat-label {
    font-size: 0.72rem;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

/* ── MAIN BODY */
.main-body {
    padding: 2.5rem 3.5rem;
    display: flex;
    gap: 2rem;
}

/* ── PANEL CARD */
.panel-card {
    background: rgba(15,23,42,0.85);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 18px;
    padding: 2rem 2.2rem;
    backdrop-filter: blur(12px);
    box-shadow: 0 4px 40px rgba(0,0,0,0.4);
}
.panel-label {
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #475569;
    margin-bottom: 0.3rem;
}
.panel-title {
    font-family: 'DM Serif Display', serif;
    font-size: 1.35rem;
    color: #e2e8f0;
    margin-bottom: 1.6rem;
    padding-bottom: 1rem;
    border-bottom: 1px solid rgba(255,255,255,0.06);
}

/* ── INPUT GROUPS */
.input-group-title {
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #38bdf8;
    margin: 1.4rem 0 0.7rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.input-group-title::after {
    content: '';
    flex: 1;
    height: 1px;
    background: rgba(56,189,248,0.15);
}

/* ── Streamlit widget overrides */
div[data-testid="stNumberInput"] label,
div[data-testid="stSelectbox"] label,
div[data-testid="stSlider"] label {
    color: #94a3b8 !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.02em;
    font-family: 'DM Sans', sans-serif !important;
}

div[data-testid="stNumberInput"] input,
div[data-testid="stSelectbox"] select {
    background: rgba(30,41,59,0.8) !important;
    border: 1px solid rgba(100,116,139,0.3) !important;
    border-radius: 10px !important;
    color: #e2e8f0 !important;
    font-family: 'DM Sans', sans-serif !important;
}
div[data-testid="stNumberInput"] input:focus,
div[data-testid="stSelectbox"] select:focus {
    border-color: #38bdf8 !important;
    box-shadow: 0 0 0 3px rgba(56,189,248,0.15) !important;
}

/* ── PREDICT BUTTON */
div[data-testid="stButton"] > button {
    width: 100%;
    background: linear-gradient(135deg, #0ea5e9 0%, #6366f1 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.85rem 1.5rem !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.95rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.04em !important;
    cursor: pointer !important;
    transition: all 0.2s ease !important;
    margin-top: 1.2rem !important;
    box-shadow: 0 4px 20px rgba(14,165,233,0.35) !important;
}
div[data-testid="stButton"] > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 28px rgba(14,165,233,0.5) !important;
}

/* ── RESULT CARDS */
.result-positive {
    background: linear-gradient(135deg, rgba(16,185,129,0.12) 0%, rgba(5,150,105,0.06) 100%);
    border: 1px solid rgba(16,185,129,0.35);
    border-radius: 16px;
    padding: 1.8rem 2rem;
    text-align: center;
    animation: fadeSlide 0.5s ease;
}
.result-negative {
    background: linear-gradient(135deg, rgba(239,68,68,0.12) 0%, rgba(185,28,28,0.06) 100%);
    border: 1px solid rgba(239,68,68,0.35);
    border-radius: 16px;
    padding: 1.8rem 2rem;
    text-align: center;
    animation: fadeSlide 0.5s ease;
}
@keyframes fadeSlide {
    from { opacity: 0; transform: translateY(12px); }
    to   { opacity: 1; transform: translateY(0); }
}
.result-icon { font-size: 2.8rem; margin-bottom: 0.5rem; }
.result-verdict {
    font-family: 'DM Serif Display', serif;
    font-size: 1.7rem;
    margin: 0.3rem 0 0.5rem;
}
.result-verdict-pos { color: #34d399; }
.result-verdict-neg { color: #f87171; }
.result-detail { color: #94a3b8; font-size: 0.88rem; }

/* ── PROBABILITY BAR */
.prob-container {
    background: rgba(15,23,42,0.85);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 14px;
    padding: 1.4rem 1.6rem;
    margin-top: 1.2rem;
    animation: fadeSlide 0.6s ease 0.1s both;
}
.prob-label-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.7rem;
}
.prob-label { color: #64748b; font-size: 0.78rem; font-weight: 500; }
.prob-value {
    font-family: 'DM Serif Display', serif;
    font-size: 1.4rem;
    color: #f1f5f9;
}
.prob-track {
    background: rgba(30,41,59,0.8);
    border-radius: 999px;
    height: 8px;
    overflow: hidden;
}
.prob-fill {
    height: 100%;
    border-radius: 999px;
    transition: width 0.8s cubic-bezier(0.25,1,0.5,1);
}
.prob-fill-pos { background: linear-gradient(90deg, #10b981, #34d399); }
.prob-fill-neg { background: linear-gradient(90deg, #ef4444, #f87171); }

/* ── BREAKDOWN METRICS */
.metrics-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0.9rem;
    margin-top: 1.2rem;
    animation: fadeSlide 0.7s ease 0.2s both;
}
.metric-chip {
    background: rgba(30,41,59,0.6);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 12px;
    padding: 1rem 1.1rem;
}
.metric-chip-label { color: #475569; font-size: 0.72rem; font-weight: 500; margin-bottom: 0.3rem; }
.metric-chip-val { color: #e2e8f0; font-size: 1.05rem; font-weight: 600; }

/* ── IDLE STATE */
.idle-state {
    text-align: center;
    padding: 3rem 1rem;
    color: #334155;
}
.idle-icon { font-size: 3.5rem; margin-bottom: 1rem; opacity: 0.4; }
.idle-text { font-size: 0.9rem; }

/* ── DIVIDER */
.divider {
    height: 1px;
    background: rgba(255,255,255,0.06);
    margin: 1.5rem 0;
}

/* ── STREAMLIT COLUMN GAP FIX */
[data-testid="column"] { padding: 0 0.6rem !important; }
</style>
""", unsafe_allow_html=True)

# ── LOAD ASSETS ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_assets():
    model   = joblib.load('model.pkl')
    scaler  = joblib.load('scaler.pkl')
    encoders = joblib.load('encoders.pkl')
    return model, scaler, encoders

try:
    model, scaler, encoders = load_assets()
    assets_ok = True
except Exception as e:
    assets_ok = False
    load_err = str(e)

# ── HERO HEADER ────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-badge">ML · Banking Intelligence</div>
    <div class="hero-title">Deposit<span>IQ</span></div>
    <div class="hero-sub">
        Real-time machine learning predictions to identify customers likely to subscribe
        to a bank term deposit — powered by a trained Random Forest model.
    </div>
    <div class="hero-stats">
        <div class="hero-stat">
            <div class="hero-stat-val">Random Forest</div>
            <div class="hero-stat-label">Model Architecture</div>
        </div>
        <div class="hero-stat">
            <div class="hero-stat-val">100 Trees</div>
            <div class="hero-stat-label">Ensemble Size</div>
        </div>
        <div class="hero-stat">
            <div class="hero-stat-val">6 Features</div>
            <div class="hero-stat-label">Input Variables</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── ERROR STATE ────────────────────────────────────────────────────────────────
if not assets_ok:
    st.error(f"⚠️ Could not load model files. Make sure model.pkl, scaler.pkl, and encoders.pkl are in the same directory.\n\n`{load_err}`")
    st.stop()

# ── BODY LAYOUT ───────────────────────────────────────────────────────────────
st.markdown('<div style="padding: 2.5rem 2.5rem 0;">', unsafe_allow_html=True)

left_col, right_col = st.columns([1, 1], gap="large")

# ── LEFT PANEL: INPUTS ────────────────────────────────────────────────────────
with left_col:
    st.markdown("""
    <div class="panel-card">
        <div class="panel-label">Step 1</div>
        <div class="panel-title">Customer Profile</div>
    </div>
    """, unsafe_allow_html=True)

    with st.container():
        # Numeric group
        st.markdown('<div class="input-group-title">📊 Financial & Contact Data</div>', unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            age = st.number_input("Age", min_value=18, max_value=100, value=35, step=1)
        with c2:
            balance = st.number_input("Account Balance (€)", min_value=-10000, max_value=200000, value=1500, step=100)

        duration = st.slider(
            "Last Call Duration (seconds)",
            min_value=0, max_value=4918, value=250, step=10,
            help="Duration of the last marketing call in seconds"
        )

        # Categorical group
        st.markdown('<div class="input-group-title" style="margin-top:1.6rem;">👤 Demographic Details</div>', unsafe_allow_html=True)

        c3, c4 = st.columns(2)
        with c3:
            job       = st.selectbox("Job Category", options=sorted(encoders['job'].classes_))
        with c4:
            marital   = st.selectbox("Marital Status", options=sorted(encoders['marital'].classes_))

        education = st.selectbox("Education Level", options=sorted(encoders['education'].classes_))

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        predict_btn = st.button("⚡  Run Prediction", use_container_width=True)

# ── RIGHT PANEL: OUTPUT ───────────────────────────────────────────────────────
with right_col:
    st.markdown("""
    <div class="panel-card">
        <div class="panel-label">Step 2</div>
        <div class="panel-title">Prediction Result</div>
    </div>
    """, unsafe_allow_html=True)

    if not predict_btn:
        st.markdown("""
        <div class="idle-state">
            <div class="idle-icon">🔮</div>
            <div class="idle-text">Fill in the customer profile on the left<br>and click <strong style="color:#38bdf8">Run Prediction</strong> to see results.</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        # ── Build input dataframe
        input_df = pd.DataFrame({
            'age':       [age],
            'balance':   [balance],
            'duration':  [duration],
            'job':       [job],
            'marital':   [marital],
            'education': [education],
        })

        # ── Encode categoricals
        for col in ['job', 'marital', 'education']:
            input_df[col] = encoders[col].transform(input_df[col])

        # ── Scale
        input_scaled = scaler.transform(input_df)

        # ── Predict
        prediction  = model.predict(input_scaled)[0]
        proba       = model.predict_proba(input_scaled)[0]
        prob_pos    = proba[1]
        prob_neg    = proba[0]

        # ── Verdict card
        if prediction == 1:
            st.markdown(f"""
            <div class="result-positive">
                <div class="result-icon">✅</div>
                <div class="result-verdict result-verdict-pos">Will Subscribe</div>
                <div class="result-detail">This customer is likely to accept a term deposit offer.</div>
            </div>
            """, unsafe_allow_html=True)
            fill_class = "prob-fill-pos"
            bar_width  = f"{prob_pos * 100:.1f}%"
            prob_show  = prob_pos
        else:
            st.markdown(f"""
            <div class="result-negative">
                <div class="result-icon">❌</div>
                <div class="result-verdict result-verdict-neg">Will Not Subscribe</div>
                <div class="result-detail">This customer is unlikely to accept a term deposit offer.</div>
            </div>
            """, unsafe_allow_html=True)
            fill_class = "prob-fill-neg"
            bar_width  = f"{prob_neg * 100:.1f}%"
            prob_show  = prob_neg

        # ── Probability bar
        pct_label = f"{prob_show * 100:.1f}%"
        st.markdown(f"""
        <div class="prob-container">
            <div class="prob-label-row">
                <span class="prob-label">CONFIDENCE SCORE</span>
                <span class="prob-value">{pct_label}</span>
            </div>
            <div class="prob-track">
                <div class="prob-fill {fill_class}" style="width:{bar_width};"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Metrics breakdown
        call_min   = duration // 60
        call_sec   = duration % 60
        call_label = f"{call_min}m {call_sec}s"
        bal_label  = f"€{balance:,}"

        st.markdown(f"""
        <div class="metrics-grid">
            <div class="metric-chip">
                <div class="metric-chip-label">Subscribe Probability</div>
                <div class="metric-chip-val">{prob_pos * 100:.1f}%</div>
            </div>
            <div class="metric-chip">
                <div class="metric-chip-label">Decline Probability</div>
                <div class="metric-chip-val">{prob_neg * 100:.1f}%</div>
            </div>
            <div class="metric-chip">
                <div class="metric-chip-label">Call Duration</div>
                <div class="metric-chip-val">{call_label}</div>
            </div>
            <div class="metric-chip">
                <div class="metric-chip-label">Account Balance</div>
                <div class="metric-chip-val">{bal_label}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Feature importance note
        st.markdown("""
        <div style="margin-top:1.2rem; padding:1rem 1.2rem;
                    background:rgba(56,189,248,0.05);
                    border:1px solid rgba(56,189,248,0.12);
                    border-radius:12px; color:#64748b; font-size:0.8rem; line-height:1.6;
                    animation: fadeSlide 0.8s ease 0.3s both;">
            💡 <strong style="color:#38bdf8;">Model Insight:</strong>
            Call duration, account balance, and job category are typically the strongest
            predictors of term deposit subscription in this model.
        </div>
        """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# ── FOOTER ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center; padding: 2rem; color:#1e293b; font-size:0.75rem;">
    DepositIQ · Powered by Random Forest · For internal business use only
</div>
""", unsafe_allow_html=True)