from datetime import datetime
import streamlit as st
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.inference import predict_default_risk

# =============================================================================
# CONFIG
# =============================================================================
st.set_page_config(
    page_title="CreditSight · Risk Analysis",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# SESSION STATE
# =============================================================================
if "page" not in st.session_state:
    st.session_state.page = "home"

# =============================================================================
# CSS
# =============================================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

#MainMenu, footer, header {
    visibility: hidden;
}

.stApp {
    background-color: #F1F5F9;
}

.block-container {
    padding-top: 0rem;
    padding-bottom: 2rem;
    max-width: 100%;
}

[data-testid="stSidebar"] {
    background: #0F172A;
}
[data-testid="stSidebar"] * {
    color: #E2E8F0 !important;
}

.hero-box {
    background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%);
    border-radius: 22px;
    padding: 56px 48px;
    color: white;
    margin-bottom: 32px;
}

.hero-badge {
    display: inline-block;
    padding: 8px 14px;
    border-radius: 999px;
    font-size: 12px;
    font-weight: 700;
    color: #93C5FD;
    background: rgba(59,130,246,0.12);
    border: 1px solid rgba(147,197,253,0.2);
    margin-bottom: 22px;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

.hero-title {
    font-size: 58px;
    line-height: 1.05;
    font-weight: 800;
    letter-spacing: -0.03em;
    margin-bottom: 18px;
    max-width: 900px;
}

.hero-sub {
    font-size: 20px;
    line-height: 1.7;
    color: #CBD5E1;
    max-width: 1050px;
    margin-bottom: 28px;
}

.chip {
    display: inline-block;
    padding: 10px 14px;
    margin-right: 10px;
    margin-bottom: 10px;
    border-radius: 12px;
    background: rgba(255,255,255,0.07);
    border: 1px solid rgba(255,255,255,0.08);
    color: #E2E8F0;
    font-size: 13px;
    font-weight: 500;
}

.section-label {
    font-size: 13px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #64748B;
    margin-top: 8px;
    margin-bottom: 14px;
}

.card {
    background: white;
    border: 1px solid #E2E8F0;
    border-radius: 18px;
    padding: 24px;
    height: 100%;
}

.card-icon {
    font-size: 24px;
    margin-bottom: 12px;
}

.card-title {
    font-size: 24px;
    font-weight: 700;
    color: #0F172A;
    margin-bottom: 10px;
}

.card-text {
    font-size: 15px;
    color: #64748B;
    line-height: 1.8;
}

.metric-card {
    background: white;
    border: 1px solid #E2E8F0;
    border-radius: 16px;
    padding: 20px;
}

.metric-label {
    font-size: 12px;
    font-weight: 700;
    color: #94A3B8;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-bottom: 8px;
}

.metric-value {
    font-size: 32px;
    font-weight: 800;
    color: #0F172A;
}

.metric-sub {
    font-size: 12px;
    color: #94A3B8;
    margin-top: 6px;
}

.result-card {
    background: white;
    border: 1px solid #E2E8F0;
    border-radius: 18px;
    padding: 28px;
    height: 100%;
}

.result-title {
    font-size: 13px;
    font-weight: 700;
    color: #64748B;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-bottom: 18px;
}

.score-number {
    font-size: 64px;
    font-weight: 800;
    color: #0F172A;
    line-height: 1;
    margin-bottom: 8px;
}

.score-sub {
    font-size: 13px;
    color: #64748B;
    margin-bottom: 20px;
}

.badge-green, .badge-orange, .badge-red {
    display: inline-block;
    padding: 10px 14px;
    border-radius: 12px;
    font-size: 14px;
    font-weight: 700;
    margin-bottom: 8px;
}

.badge-green {
    background: #ECFDF5;
    color: #059669;
    border: 1px solid #A7F3D0;
}

.badge-orange {
    background: #FFFBEB;
    color: #D97706;
    border: 1px solid #FCD34D;
}

.badge-red {
    background: #FEF2F2;
    color: #DC2626;
    border: 1px solid #FECACA;
}

.factor-row {
    padding: 14px 0;
    border-bottom: 1px solid #F1F5F9;
}

.factor-name {
    font-size: 15px;
    font-weight: 700;
    color: #0F172A;
}

.factor-desc {
    font-size: 13px;
    color: #64748B;
    margin-bottom: 8px;
}

.disclaimer {
    margin-top: 24px;
    padding: 16px 18px;
    background: #F8FAFC;
    border: 1px solid #E2E8F0;
    border-radius: 12px;
    font-size: 12px;
    color: #64748B;
    line-height: 1.7;
}

.topbar-wrap {
    background: white;
    border-bottom: 1px solid #E2E8F0;
    padding: 18px 32px;
    margin-bottom: 30px;
}

.topbar-title {
    font-size: 18px;
    font-weight: 800;
    color: #0F172A;
}

.topbar-sub {
    font-size: 12px;
    color: #94A3B8;
}

.stButton > button {
    border-radius: 12px;
    height: 46px;
    font-weight: 700;
    border: 1px solid #CBD5E1;
}

.main-launch-btn button {
    background: #0F172A !important;
    color: white !important;
    border: none !important;
    height: 54px !important;
    font-size: 16px !important;
}
            
/* Texte sliders (labels) */
label {
    color: #0F172A !important;
}

/* Titres (Borrower profile / Loan exposure) */
h4 {
    color: #0F172A !important;
}

/* Valeurs au-dessus des sliders */
[data-baseweb="slider"] div {
    color: #0F172A !important;
}
            
</style>
""", unsafe_allow_html=True)

# =============================================================================
# HELPERS
# =============================================================================
def go_to_home():
    st.session_state.page = "home"

def go_to_analysis():
    st.session_state.page = "analysis"

def render_topbar():
    now = datetime.now().strftime("%d %b %Y · %H:%M")
    left, right = st.columns([4, 1])
    with left:
        st.markdown(
            """
            <div class="topbar-wrap">
                <div class="topbar-title">🏦 CreditSight</div>
                <div class="topbar-sub">Risk Intelligence Platform</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with right:
        st.markdown(
            f"""
            <div style="padding: 22px 6px 0 0; text-align:right;">
                <div style="font-size:12px; color:#94A3B8;">Credit decision support interface</div>
                <div style="font-size:12px; color:#94A3B8;">{now}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

def risk_level(prob):
    if prob < 0.25:
        return "Approve", "badge-green", "Low-risk profile — the application can proceed under standard review."
    elif prob < 0.55:
        return "Manual Review", "badge-orange", "Moderate-risk profile — additional assessment is recommended."
    else:
        return "Decline", "badge-red", "High-risk profile — probability of default is materially elevated."

def factor_status(value, low, high, reverse=False):
    if reverse:
        if value >= high:
            return "Favorable"
        elif value >= low:
            return "Moderate"
        return "Adverse"
    else:
        if value <= low:
            return "Favorable"
        elif value <= high:
            return "Moderate"
        return "Adverse"
    
def get_explanation(factor, status):
    explanations = {
        "fico": {
            "Favorable": "Strong credit history indicating reliable repayment behavior.",
            "Moderate": "Average credit profile — may require additional review.",
            "Adverse": "Low credit score indicating higher default risk."
        },
        "dti": {
            "Favorable": "Low debt-to-income ratio suggests healthy financial balance.",
            "Moderate": "Moderate leverage — manageable but should be monitored.",
            "Adverse": "High debt burden relative to income, increasing default risk."
        },
        "employment": {
            "Favorable": "Stable employment history supports repayment capacity.",
            "Moderate": "Moderate job stability — some uncertainty remains.",
            "Adverse": "Limited employment history indicating higher income instability."
        },
        "lines": {
            "Favorable": "Balanced credit usage indicates controlled exposure.",
            "Moderate": "Moderate credit exposure — acceptable but not optimal.",
            "Adverse": "High number of credit lines may signal overexposure risk."
        }
    }
    return explanations[factor][status]

# =============================================================================
# HOME PAGE
# =============================================================================
def render_home():
    render_topbar()

    st.markdown(
        """
        <div class="hero-box">
            <div class="hero-badge">Enterprise Credit Decisioning</div>
            <div class="hero-title">Smarter credit risk decisions, in seconds.</div>
            <div class="hero-sub">
                CreditSight helps lending teams assess borrower profiles, surface key risk signals,
                and support faster, more consistent underwriting decisions through an intuitive,
                analyst-grade scoring experience.
            </div>
            <div>
                <span class="chip">Real-time borrower scoring</span>
                <span class="chip">Transparent risk drivers</span>
                <span class="chip">Decision support workflow</span>
                <span class="chip">Analyst-ready interface</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    c1, c2, c3 = st.columns([1, 1.2, 1])
    with c2:
        st.markdown('<div class="main-launch-btn">', unsafe_allow_html=True)
        if st.button("Launch Risk Analysis", use_container_width=True):
            go_to_analysis()
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
        st.caption("Access the scoring interface and evaluate a borrower profile.")

    st.markdown('<div class="section-label">Platform capabilities</div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="card">
            <div class="card-icon">⚡</div>
            <div class="card-title">Instant scoring</div>
            <div class="card-text">
                Generate a borrower risk assessment in real time using a structured decision workflow designed for rapid evaluation.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="card">
            <div class="card-icon">🔎</div>
            <div class="card-title">Explainable signals</div>
            <div class="card-text">
                Highlight the main drivers behind the final score, from credit quality to leverage and employment stability.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="card">
            <div class="card-icon">✅</div>
            <div class="card-title">Decision support</div>
            <div class="card-text">
                Support approval, manual review, and rejection workflows with a consistent, risk-based recommendation layer.
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="section-label" style="margin-top:26px;">Decision engine overview</div>', unsafe_allow_html=True)
    left, right = st.columns([1.2, 0.8])

    with left:
        st.markdown("""
        <div class="card">
            <div class="card-title">Built for modern underwriting workflows</div>
            <div class="card-text">
                CreditSight is designed to give lending and risk teams a fast, structured, and explainable
                view of borrower quality. The platform combines financial profile inputs, leverage indicators,
                and credit signals into a clear decision-support experience suitable for high-volume credit
                assessment environments.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with right:
        model_name = "Selected credit risk model"
        model_status = "Production-ready scoring engine"

        st.markdown(f"""
        <div class="card">
            <div class="card-title">Scoring engine</div>
            <div style="margin-top:16px; padding:16px; background:#F8FAFC; border:1px solid #E2E8F0; border-radius:12px;">
                <div class="metric-label">Status</div>
                <div style="font-size:18px; font-weight:700; color:#0F172A;">{model_name}</div>
                <div class="metric-sub">{model_status}</div>
            </div>
            <div style="margin-top:14px; padding:16px; background:#F8FAFC; border:1px solid #E2E8F0; border-radius:12px;">
                <div class="metric-label">Output</div>
                <div style="font-size:18px; font-weight:700; color:#0F172A;">Probability of default</div>
                <div class="metric-sub">Displayed as a risk score with actionable recommendation tiers.</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

# =============================================================================
# ANALYSIS PAGE
# =============================================================================
def render_analysis():
    render_topbar()

    st.markdown('<div class="section-label">Borrower summary</div>', unsafe_allow_html=True)

    # Valeurs par défaut
    default_fico = 638
    default_income = 70_000
    default_years_employed = 5
    default_loan_amt = 4_160
    default_total_debt = 8_719
    default_credit_lines = 1

    # Etat initial
    if "fico_score" not in st.session_state:
        st.session_state.fico_score = default_fico
    if "income" not in st.session_state:
        st.session_state.income = default_income
    if "years_employed" not in st.session_state:
        st.session_state.years_employed = default_years_employed
    if "loan_amt" not in st.session_state:
        st.session_state.loan_amt = default_loan_amt
    if "total_debt" not in st.session_state:
        st.session_state.total_debt = default_total_debt
    if "credit_lines" not in st.session_state:
        st.session_state.credit_lines = default_credit_lines
    if "analysis_run" not in st.session_state:
        st.session_state.analysis_run = False

    fico_score = st.session_state.fico_score
    income = st.session_state.income
    years_employed = st.session_state.years_employed
    loan_amt = st.session_state.loan_amt
    total_debt = st.session_state.total_debt
    credit_lines = st.session_state.credit_lines

    dti_pct = round(total_debt / max(income, 1) * 100, 1)

    m1, m2, m3, m4 = st.columns(4)

    with m1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">FICO Score</div>
            <div class="metric-value">{fico_score}</div>
            <div class="metric-sub">{'Above portfolio average' if fico_score > 638 else 'Below portfolio average'}</div>
        </div>
        """, unsafe_allow_html=True)

    with m2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">DTI Ratio</div>
            <div class="metric-value">{dti_pct}%</div>
            <div class="metric-sub">Recommended threshold: &lt; 35%</div>
        </div>
        """, unsafe_allow_html=True)

    with m3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Annual Income</div>
            <div class="metric-value">${income:,}</div>
            <div class="metric-sub">Outstanding loan: ${loan_amt:,}</div>
        </div>
        """, unsafe_allow_html=True)

    with m4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Employment</div>
            <div class="metric-value">{years_employed} yrs</div>
            <div class="metric-sub">{credit_lines} open credit line(s)</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    left, right = st.columns([1, 1])

    with left:
        st.markdown("""
        <div class="result-card">
            <div class="result-title">Borrower input form</div>
        """, unsafe_allow_html=True)

        with st.form("borrower_form"):
            st.markdown("#### Borrower profile")
            fico_score_input = st.slider("FICO Score", 408, 850, st.session_state.fico_score)
            income_input = st.slider("Annual Income ($)", 1_000, 150_000, st.session_state.income, step=500)
            years_employed_input = st.slider("Employment Length (years)", 0, 10, st.session_state.years_employed)

            st.markdown("#### Loan exposure")
            loan_amt_input = st.slider("Outstanding Loan Amount ($)", 47, 10_751, st.session_state.loan_amt, step=50)
            total_debt_input = st.slider("Total Debt ($)", 32, 43_689, st.session_state.total_debt, step=100)
            credit_lines_input = st.slider("Open Credit Lines", 0, 5, st.session_state.credit_lines)

            c1, c2 = st.columns([1, 1])
            with c1:
                submitted = st.form_submit_button("▶ Run Analysis", use_container_width=True)
            with c2:
                back_clicked = st.form_submit_button("← Back to Overview", use_container_width=True)

        st.markdown("</div>", unsafe_allow_html=True)

        if back_clicked:
            go_to_home()
            st.rerun()

        if submitted:
            st.session_state.fico_score = fico_score_input
            st.session_state.income = income_input
            st.session_state.years_employed = years_employed_input
            st.session_state.loan_amt = loan_amt_input
            st.session_state.total_debt = total_debt_input
            st.session_state.credit_lines = credit_lines_input
            st.session_state.analysis_run = True
            st.rerun()

    with right:
        if not st.session_state.analysis_run:
            st.markdown("""
            <div class="result-card">
                <div class="result-title">Risk analysis output</div>
                <div style="text-align:center; padding:50px 10px; color:#94A3B8;">
                    <div style="font-size:42px; margin-bottom:12px;">📋</div>
                    <div style="font-size:16px; font-weight:600; color:#64748B;">
                        Fill in the borrower profile and click <strong>Run Analysis</strong>.
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            fico_score = st.session_state.fico_score
            income = st.session_state.income
            years_employed = st.session_state.years_employed
            loan_amt = st.session_state.loan_amt
            total_debt = st.session_state.total_debt
            credit_lines = st.session_state.credit_lines

            dti_pct = round(total_debt / max(income, 1) * 100, 1)

            input_data = {
                "credit_lines_outstanding": credit_lines,
                "loan_amt_outstanding": loan_amt,
                "total_debt_outstanding": total_debt,
                "income": income,
                "years_employed": years_employed,
                "fico_score": fico_score
            }

            result = predict_default_risk(input_data)
            proba = result["default_probability"]

            st.markdown(
                f"""
                <div style="font-size:12px; color:#0F172A; margin-bottom:10px;">
                    Raw probability: {proba:.6f} <br>
                    Prediction: {result["prediction"]}
                </div>
                """,
                unsafe_allow_html=True
            )

            pct = round(proba * 100, 1)

            if proba < 0.25:
                badge_class = "badge-green"
                decision = "Approve"
                decision_sub = "Low-risk profile — the application can proceed under standard review."
            elif proba < 0.55:
                badge_class = "badge-orange"
                decision = "Manual Review"
                decision_sub = "Moderate-risk profile — additional assessment is recommended."
            else:
                badge_class = "badge-red"
                decision = "Decline"
                decision_sub = "High-risk profile — probability of default is materially elevated."

            def factor_status(value, low, high, reverse=False):
                if reverse:
                    if value >= high:
                        return "Favorable"
                    elif value >= low:
                        return "Moderate"
                    return "Adverse"
                else:
                    if value <= low:
                        return "Favorable"
                    elif value <= high:
                        return "Moderate"
                    return "Adverse"

            fico_status = factor_status(fico_score, 580, 700, reverse=True)
            dti_status = factor_status(dti_pct, 20, 40)
            emp_status = factor_status(years_employed, 2, 5, reverse=True)
            lines_status = factor_status(credit_lines, 2, 4)

            st.markdown(f"""
            <div class="result-card">
                <div class="result-title">Risk score</div>
                <div class="score-number">{pct}%</div>
                <div class="score-sub">Probability of default</div>
                <div class="{badge_class}">Recommendation: {decision}</div>
                <div style="font-size:14px; color:#64748B; margin-top:8px; margin-bottom:20px;">
                    {decision_sub}
                </div>
            """, unsafe_allow_html=True)

            st.progress(min(max(proba, 0.0), 1.0))

            st.markdown(f"""
            <div class="result-card">

            <div class="result-title" style="margin-top:0;">Risk factor breakdown</div>

            <div class="factor-row">
                <div class="factor-name">FICO Score</div>
                <div class="factor-desc">{fico_score} · Portfolio average: 638</div>
                <div style="font-size:13px; font-weight:700; color:#334155;">{fico_status}</div>
                <div style="font-size:12px; color:#64748B; margin-top:4px;">
                    {get_explanation("fico", fico_status)}
                </div>
            </div>

            <div class="factor-row">
                <div class="factor-name">Debt-to-Income Ratio</div>
                <div class="factor-desc">{dti_pct}% · Recommended threshold: &lt; 35%</div>
                <div style="font-size:13px; font-weight:700; color:#334155;">{dti_status}</div>
                <div style="font-size:12px; color:#64748B; margin-top:4px;">
                    {get_explanation("dti", dti_status)}
                </div>
            </div>

            <div class="factor-row">
                <div class="factor-name">Employment Stability</div>
                <div class="factor-desc">{years_employed} year(s) · Tenure indicator</div>
                <div style="font-size:13px; font-weight:700; color:#334155;">{emp_status}</div>
                <div style="font-size:12px; color:#64748B; margin-top:4px;">
                    {get_explanation("employment", emp_status)}
                </div>
            </div>

            <div class="factor-row" style="border-bottom:none;">
                <div class="factor-name">Open Credit Lines</div>
                <div class="factor-desc">{credit_lines} line(s) · Exposure footprint</div>
                <div style="font-size:13px; font-weight:700; color:#334155;">{lines_status}</div>
                <div style="font-size:12px; color:#64748B; margin-top:4px;">
                    {get_explanation("lines", lines_status)}
                </div>
            </div>

            </div>
            """, unsafe_allow_html=True)

    st.markdown("""
    <div class="disclaimer">
        <strong>Disclaimer:</strong> This interface provides automated risk decision support for internal use.
        Outputs are intended to assist underwriting and risk review workflows and should be used alongside
        established credit governance and human oversight procedures.
    </div>
    """, unsafe_allow_html=True)

# =============================================================================
# ROUTER
# =============================================================================
if st.session_state.page == "home":
    render_home()
else:
    render_analysis()