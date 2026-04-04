import streamlit as st
import numpy as np
import pickle
import os
from datetime import datetime

st.set_page_config(
    page_title="CreditSight · Risk Analysis",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
#MainMenu, footer { visibility: hidden; }

/* ── Page background ── */
.stApp { background: #F0F2F5; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #0F1B2D !important;
    border-right: none;
}
[data-testid="stSidebar"] * { color: #CBD5E1 !important; }
[data-testid="stSidebar"] .stSlider label { font-size: 12px !important; color: #94A3B8 !important; }
[data-testid="stSidebar"] .stSlider > div > div > div > div { background: #3B82F6 !important; }

/* ── Main content ── */
.block-container { padding: 0 !important; max-width: 100% !important; }

/* ── Topbar ── */
.topbar {
    background: #FFFFFF;
    border-bottom: 1px solid #E2E8F0;
    padding: 14px 32px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 0;
}
.topbar-brand {
    display: flex;
    align-items: center;
    gap: 10px;
}
.brand-icon {
    width: 32px; height: 32px;
    background: #1E40AF;
    border-radius: 8px;
    display: flex; align-items: center; justify-content: center;
    font-size: 16px;
}
.brand-name {
    font-size: 16px;
    font-weight: 700;
    color: #0F172A;
    letter-spacing: -0.01em;
}
.brand-sub {
    font-size: 11px;
    color: #94A3B8;
    font-weight: 400;
}
.topbar-meta {
    font-size: 12px;
    color: #94A3B8;
    text-align: right;
    line-height: 1.6;
}

/* ── Page content ── */
.page-content { padding: 28px 32px; }

/* ── Section title ── */
.section-title {
    font-size: 13px;
    font-weight: 600;
    color: #64748B;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    margin: 0 0 14px;
}

/* ── KPI Cards ── */
.kpi-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 14px;
    margin-bottom: 24px;
}
.kpi-card {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 10px;
    padding: 18px 20px;
}
.kpi-label {
    font-size: 11px;
    font-weight: 600;
    color: #94A3B8;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-bottom: 6px;
}
.kpi-value {
    font-size: 24px;
    font-weight: 700;
    color: #0F172A;
    font-family: 'JetBrains Mono', monospace;
    line-height: 1;
}
.kpi-delta {
    font-size: 11px;
    color: #94A3B8;
    margin-top: 4px;
}

/* ── Result layout ── */
.result-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 16px;
}

/* ── Score card ── */
.score-card {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 10px;
    padding: 28px 24px;
}
.score-header {
    font-size: 12px;
    font-weight: 600;
    color: #64748B;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-bottom: 20px;
}

/* Jauge SVG wrapper */
.gauge-wrap { text-align: center; margin: 8px 0 16px; }

.decision-box {
    border-radius: 8px;
    padding: 14px 18px;
    display: flex;
    align-items: center;
    gap: 12px;
    margin-top: 16px;
}
.decision-approved { background: #ECFDF5; border: 1px solid #A7F3D0; }
.decision-review   { background: #FFFBEB; border: 1px solid #FCD34D; }
.decision-rejected { background: #FEF2F2; border: 1px solid #FECACA; }
.decision-dot {
    width: 10px; height: 10px;
    border-radius: 50%;
    flex-shrink: 0;
}
.dot-green  { background: #10B981; }
.dot-orange { background: #F59E0B; }
.dot-red    { background: #EF4444; }
.decision-label {
    font-size: 14px;
    font-weight: 600;
    color: #0F172A;
}
.decision-sub {
    font-size: 12px;
    color: #64748B;
    margin-top: 1px;
}

/* ── Breakdown card ── */
.breakdown-card {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 10px;
    padding: 24px;
}
.factor-item {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 10px 0;
    border-bottom: 1px solid #F1F5F9;
}
.factor-item:last-child { border-bottom: none; }
.factor-icon {
    width: 32px; height: 32px;
    border-radius: 8px;
    display: flex; align-items: center; justify-content: center;
    font-size: 14px;
    flex-shrink: 0;
}
.fi-blue   { background: #EFF6FF; }
.fi-green  { background: #ECFDF5; }
.fi-orange { background: #FFFBEB; }
.fi-red    { background: #FEF2F2; }
.factor-info { flex: 1; }
.factor-name { font-size: 13px; font-weight: 500; color: #0F172A; }
.factor-desc { font-size: 11px; color: #94A3B8; margin-top: 1px; }
.factor-bar-wrap {
    width: 90px;
    background: #F1F5F9;
    border-radius: 3px;
    height: 5px;
    overflow: hidden;
}
.factor-bar { height: 100%; border-radius: 3px; }
.bar-green  { background: #10B981; }
.bar-orange { background: #F59E0B; }
.bar-red    { background: #EF4444; }
.factor-tag {
    font-size: 10px;
    font-weight: 600;
    padding: 2px 8px;
    border-radius: 4px;
    min-width: 68px;
    text-align: center;
}
.tag-good { background: #ECFDF5; color: #059669; }
.tag-mod  { background: #FFFBEB; color: #D97706; }
.tag-bad  { background: #FEF2F2; color: #DC2626; }

/* ── Disclaimer ── */
.disclaimer {
    margin-top: 24px;
    padding: 14px 18px;
    background: #F8FAFC;
    border: 1px solid #E2E8F0;
    border-radius: 8px;
    font-size: 11px;
    color: #94A3B8;
    line-height: 1.6;
}

/* ── Sidebar labels ── */
.sidebar-section {
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #475569;
    margin: 20px 0 10px;
    padding: 0 4px;
}
</style>
""", unsafe_allow_html=True)


# ── Modèle ────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    if os.path.exists("model.pkl"):
        with open("model.pkl", "rb") as f:
            return pickle.load(f)
    return None

model = load_model()


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding: 24px 16px 8px;">
        <div style="font-size:18px; font-weight:700; color:#F1F5F9; letter-spacing:-0.01em;">
            🏦 CreditSight
        </div>
        <div style="font-size:11px; color:#475569; margin-top:2px;">
            Risk Analysis Platform
        </div>
    </div>
    <hr style="border:none; border-top:1px solid #1E293B; margin: 16px 0;">
    """, unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section">Profil emprunteur</div>', unsafe_allow_html=True)

    fico_score     = st.slider("Score FICO", 408, 850, 638,
                               help="Score de crédit FICO (408–850)")
    income         = st.slider("Revenus annuels ($)", 1_000, 150_000, 70_000, step=500)
    years_employed = st.slider("Ancienneté (ans)", 0, 10, 5)

    st.markdown('<div class="sidebar-section">Données du prêt</div>', unsafe_allow_html=True)

    loan_amt     = st.slider("Encours du prêt ($)", 47, 10_751, 4_160, step=50)
    total_debt   = st.slider("Dette totale ($)", 32, 43_689, 8_719, step=100)
    credit_lines = st.slider("Lignes de crédit ouvertes", 0, 5, 1)

    st.markdown("<br>", unsafe_allow_html=True)
    analyze = st.button("▶  Lancer l'analyse", use_container_width=True)

    st.markdown("""
    <hr style="border:none;border-top:1px solid #1E293B;margin:20px 0 12px;">
    <div style="font-size:10px;color:#334155;padding:0 4px;line-height:1.6;">
        Modèle v1.0 · Dataset: Loan_Data<br>
        Taux de défaut observé: 18.5%
    </div>
    """, unsafe_allow_html=True)


# ── Topbar ────────────────────────────────────────────────────────────────────
now = datetime.now().strftime("%d %b %Y · %H:%M")
st.markdown(f"""
<div class="topbar">
    <div class="topbar-brand">
        <div class="brand-icon">🏦</div>
        <div>
            <div class="brand-name">CreditSight</div>
            <div class="brand-sub">Risk Analysis Platform</div>
        </div>
    </div>
    <div class="topbar-meta">
        Analyse individuelle · Prêts personnels<br>{now}
    </div>
</div>
""", unsafe_allow_html=True)


# ── Calculs préliminaires ─────────────────────────────────────────────────────
dti_pct  = round(total_debt / max(income, 1) * 100, 1)
lti_pct  = round(loan_amt   / max(income, 1) * 100, 1)
debt_inc = round(total_debt / max(income, 1), 2)

# ── KPI Cards ─────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="page-content">
<div class="section-title">Vue d'ensemble du dossier</div>
<div class="kpi-grid">
    <div class="kpi-card">
        <div class="kpi-label">Score FICO</div>
        <div class="kpi-value">{fico_score}</div>
        <div class="kpi-delta">{'↑ Supérieur à la moyenne' if fico_score > 638 else '↓ Inférieur à la moyenne'}</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-label">Ratio DTI</div>
        <div class="kpi-value">{dti_pct}%</div>
        <div class="kpi-delta">Seuil acceptable : &lt; 35%</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-label">Revenus annuels</div>
        <div class="kpi-value">${income:,}</div>
        <div class="kpi-delta">Encours : ${loan_amt:,}</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-label">Ancienneté</div>
        <div class="kpi-value">{years_employed} ans</div>
        <div class="kpi-delta">{credit_lines} ligne(s) de crédit</div>
    </div>
</div>
""", unsafe_allow_html=True)


# ── Résultat ──────────────────────────────────────────────────────────────────
if analyze:
    features = np.array([[credit_lines, loan_amt, total_debt,
                          income, years_employed, fico_score]])

    if model is not None:
        proba = float(model.predict_proba(features)[0][1])
    else:
        fn    = (850 - fico_score) / (850 - 408)
        dti_r = min(total_debt / max(income, 1), 1.2) / 1.2
        lti   = min(loan_amt   / max(income, 1), 0.5) / 0.5
        proba = float(np.clip(
            fn * 0.40 + dti_r * 0.28 + lti * 0.15
            + (credit_lines / 5) * 0.10
            + (1 - years_employed / 10) * 0.07,
            0.02, 0.97
        ))
        st.info("⚠️ Modèle non chargé · Score indicatif uniquement · Placez model.pkl à la racine")

    pct = round(proba * 100, 1)

    # Décision
    if proba < 0.25:
        dec_class = "decision-approved"
        dot_class = "dot-green"
        dec_label = "Recommandation : Approbation"
        dec_sub   = "Profil à faible risque — le dossier peut être validé."
        gauge_col = "#10B981"
    elif proba < 0.55:
        dec_class = "decision-review"
        dot_class = "dot-orange"
        dec_label = "Recommandation : Révision manuelle"
        dec_sub   = "Profil à risque modéré — analyse complémentaire recommandée."
        gauge_col = "#F59E0B"
    else:
        dec_class = "decision-rejected"
        dot_class = "dot-red"
        dec_label = "Recommandation : Refus"
        dec_sub   = "Profil à risque élevé — probabilité de défaut significative."
        gauge_col = "#EF4444"

    # Facteurs
    def tag(val, low, high, reverse=False):
        if reverse:
            if val >= high: return "good", "tag-good", "bar-green", val / (high * 1.2)
            if val >= low:  return "mod",  "tag-mod",  "bar-orange", val / (high * 1.2)
            return "bad", "tag-bad", "bar-red", val / (high * 1.2)
        else:
            if val <= low:  return "good", "tag-good", "bar-green", val / (high * 1.2)
            if val <= high: return "mod",  "tag-mod",  "bar-orange", val / (high * 1.2)
            return "bad", "tag-bad", "bar-red", min(val / (high * 1.2), 1.0)

    labels = {"good": "Favorable", "mod": "Modéré", "bad": "Défavorable"}

    fk, fc, fb, fp = tag(fico_score,     580, 700, reverse=True)
    dk, dc, db, dp = tag(dti_pct,        20,  40)
    ek, ec, eb, ep = tag(years_employed, 2,   5,  reverse=True)
    lk, lc, lb, lp = tag(credit_lines,   2,   4)

    angle = int(proba * 180)

    # Gauge SVG demi-cercle
    gauge_svg = f"""
    <svg viewBox="0 0 200 110" width="220" xmlns="http://www.w3.org/2000/svg">
      <path d="M 20 100 A 80 80 0 0 1 180 100" fill="none" stroke="#E2E8F0" stroke-width="14" stroke-linecap="round"/>
      <path d="M 20 100 A 80 80 0 0 1 180 100" fill="none" stroke="{gauge_col}" stroke-width="14"
            stroke-linecap="round" stroke-dasharray="251.2" stroke-dashoffset="{int(251.2 * (1 - proba))}"/>
      <text x="100" y="88" text-anchor="middle" font-family="JetBrains Mono,monospace"
            font-size="28" font-weight="700" fill="#0F172A">{pct}%</text>
      <text x="100" y="104" text-anchor="middle" font-family="Inter,sans-serif"
            font-size="9" fill="#94A3B8" letter-spacing="1">PROBABILITÉ DE DÉFAUT</text>
      <text x="22" y="112" font-family="Inter,sans-serif" font-size="8" fill="#CBD5E1">0%</text>
      <text x="168" y="112" font-family="Inter,sans-serif" font-size="8" fill="#CBD5E1">100%</text>
    </svg>
    """

    st.markdown(f"""
    <div class="section-title" style="margin-top:0;">Résultat de l'analyse</div>
    <div class="result-grid">
        <div class="score-card">
            <div class="score-header">Score de risque</div>
            <div class="gauge-wrap">{gauge_svg}</div>
            <div class="decision-box {dec_class}">
                <div class="decision-dot {dot_class}"></div>
                <div>
                    <div class="decision-label">{dec_label}</div>
                    <div class="decision-sub">{dec_sub}</div>
                </div>
            </div>
        </div>

        <div class="breakdown-card">
            <div class="score-header">Analyse des facteurs de risque</div>

            <div class="factor-item">
                <div class="factor-icon fi-blue">📊</div>
                <div class="factor-info">
                    <div class="factor-name">Score FICO</div>
                    <div class="factor-desc">{fico_score} · Moyenne portefeuille : 638</div>
                </div>
                <div class="factor-bar-wrap">
                    <div class="factor-bar {fb}" style="width:{min(int(fp*100),100)}%"></div>
                </div>
                <div class="factor-tag {fc}">{labels[fk]}</div>
            </div>

            <div class="factor-item">
                <div class="factor-icon fi-orange">💳</div>
                <div class="factor-info">
                    <div class="factor-name">Ratio dette / revenu</div>
                    <div class="factor-desc">{dti_pct}% · Seuil recommandé : &lt; 35%</div>
                </div>
                <div class="factor-bar-wrap">
                    <div class="factor-bar {db}" style="width:{min(int(dp*100),100)}%"></div>
                </div>
                <div class="factor-tag {dc}">{labels[dk]}</div>
            </div>

            <div class="factor-item">
                <div class="factor-icon fi-green">💼</div>
                <div class="factor-info">
                    <div class="factor-name">Ancienneté professionnelle</div>
                    <div class="factor-desc">{years_employed} an(s) · Stabilité de l'emploi</div>
                </div>
                <div class="factor-bar-wrap">
                    <div class="factor-bar {eb}" style="width:{min(int(ep*100),100)}%"></div>
                </div>
                <div class="factor-tag {ec}">{labels[ek]}</div>
            </div>

            <div class="factor-item">
                <div class="factor-icon fi-blue">🔗</div>
                <div class="factor-info">
                    <div class="factor-name">Lignes de crédit ouvertes</div>
                    <div class="factor-desc">{credit_lines} ligne(s) · Risque d'exposition</div>
                </div>
                <div class="factor-bar-wrap">
                    <div class="factor-bar {lb}" style="width:{min(int(lp*100),100)}%"></div>
                </div>
                <div class="factor-tag {lc}">{labels[lk]}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

else:
    st.markdown("""
    <div style="text-align:center; padding: 60px 0; color:#94A3B8;">
        <div style="font-size:40px; margin-bottom:16px;">📋</div>
        <div style="font-size:15px; font-weight:500; color:#64748B;">
            Renseignez le profil dans le panneau de gauche<br>et cliquez sur <strong>Lancer l'analyse</strong>.
        </div>
    </div>
    """, unsafe_allow_html=True)

# ── Disclaimer ────────────────────────────────────────────────────────────────
st.markdown("""
    <div class="disclaimer">
        <strong>Avertissement :</strong> Cette analyse est générée automatiquement par un modèle de machine learning
        à des fins d'aide à la décision uniquement. Elle ne constitue pas une décision de crédit définitive et ne
        remplace pas l'évaluation d'un analyste qualifié. Les prédictions sont basées sur des données historiques
        et peuvent ne pas refléter l'ensemble des facteurs de risque individuels.
        Conformément à la réglementation en vigueur, toute décision de crédit doit être validée par un responsable
        habilité. · <em>CreditSight Risk Platform v1.0</em>
    </div>
</div>
""", unsafe_allow_html=True)
