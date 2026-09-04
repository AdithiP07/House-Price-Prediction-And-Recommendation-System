"""
AI-Powered House Price Prediction & Smart Property Recommendation System
Streamlit Application — Production-Ready Academic Submission
"""

# Suppress Streamlit use_container_width deprecation noise
import warnings
warnings.filterwarnings("ignore", message=".*use_container_width.*")

import os
import json
import warnings
import numpy as np
import pandas as pd
import streamlit as st
import joblib

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# PAGE CONFIG — must be first Streamlit call
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="🏠 House Price AI",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
/* Google Font */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* App background */
.stApp {
    background: linear-gradient(135deg, #0b091f 0%, #1e1b4b 50%, #15132d 100%) !important;
    color: #ffffff !important;
}

/* Sidebar Container */
[data-testid="stSidebar"] {
    background: #110e2e !important;
    border-right: 1px solid rgba(255, 255, 255, 0.2) !important;
}

/* All Sidebar Text & Labels — High Contrast Bright White */
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] h4,
[data-testid="stSidebar"] .stMarkdown {
    color: #ffffff !important;
    font-weight: 600 !important;
}

/* Sidebar Radio & Checkbox Labels */
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p,
[data-testid="stSidebar"] .stCheckbox label span,
[data-testid="stSidebar"] .stRadio label span {
    color: #ffffff !important;
    font-weight: 500 !important;
}

/* Form Controls & Selectboxes: Solid dark background with bright white text */
div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div,
.stSelectbox div[role="combobox"],
input, select {
    background-color: #231f4e !important;
    color: #ffffff !important;
    border: 1.5px solid #6366f1 !important;
    border-radius: 10px !important;
}

/* Value displayed inside selectbox */
div[data-baseweb="select"] span,
div[data-baseweb="select"] div {
    color: #ffffff !important;
    font-weight: 600 !important;
}

/* Dropdown popover list & options: High contrast dark background with crisp white text */
div[data-baseweb="popover"],
div[data-baseweb="popover"] ul,
div[data-baseweb="menu"],
ul[role="listbox"] {
    background-color: #1a173d !important;
    border: 1px solid #6366f1 !important;
    border-radius: 10px !important;
}

li[role="option"],
li[role="option"] span,
div[data-baseweb="popover"] li {
    background-color: #1a173d !important;
    color: #ffffff !important;
    font-size: 0.95rem !important;
    font-weight: 600 !important;
    padding: 0.5rem 1rem !important;
}

li[role="option"]:hover,
li[role="option"][aria-selected="true"] {
    background-color: #4f46e5 !important;
    color: #ffffff !important;
}

/* Slider values & numbers */
.stSlider [data-testid="stTickBarMin"],
.stSlider [data-testid="stTickBarMax"],
.stSlider div {
    color: #ffffff !important;
    font-weight: 600 !important;
}

/* Main header */
.main-header {
    background: linear-gradient(90deg, #4f46e5 0%, #7c3aed 100%);
    padding: 2.2rem 2rem;
    border-radius: 20px;
    text-align: center;
    margin-bottom: 2rem;
    box-shadow: 0 15px 45px rgba(79, 70, 229, 0.4);
}
.main-header h1 { color: #ffffff !important; font-size: 2.3rem; font-weight: 800; margin: 0; }
.main-header p  { color: #f1f5f9 !important; font-size: 1.05rem; margin: 0.5rem 0 0; font-weight: 500; }

/* Metric cards with high visibility */
.metric-card {
    background: rgba(30, 27, 75, 0.85);
    border: 1.5px solid rgba(167, 139, 250, 0.35);
    border-radius: 16px;
    padding: 1.4rem 1.2rem;
    text-align: center;
    backdrop-filter: blur(8px);
    transition: transform 0.2s, box-shadow 0.2s;
}
.metric-card:hover { transform: translateY(-4px); box-shadow: 0 12px 40px rgba(0,0,0,0.5); }
.metric-label  { font-size: 0.82rem; color: #cbd5e1 !important; text-transform: uppercase; letter-spacing: 1px; font-weight: 700; }
.metric-value  { font-size: 2rem; font-weight: 800; color: #ffffff !important; margin: 0.3rem 0; }
.metric-sub    { font-size: 0.85rem; color: #94a3b8 !important; font-weight: 600; }

/* Price result hero */
.price-hero {
    background: linear-gradient(135deg, #4338ca, #6b21a8);
    border: 2px solid rgba(167, 139, 250, 0.5);
    border-radius: 20px;
    padding: 2.5rem;
    text-align: center;
    margin: 1.5rem 0;
    box-shadow: 0 16px 48px rgba(79, 70, 229, 0.5);
    animation: fadeInUp 0.6s ease;
}
@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(20px); }
    to   { opacity: 1; transform: translateY(0); }
}
.price-hero h2     { color: #f8fafc !important; font-size: 1.2rem; font-weight: 600; margin: 0 0 0.5rem; }
.price-hero .price { color: #ffffff !important; font-size: 3.2rem; font-weight: 800; margin: 0.3rem 0; text-shadow: 0 4px 12px rgba(0,0,0,0.4); }
.price-hero .sub   { color: #e2e8f0 !important; font-size: 1.05rem; font-weight: 500; }

/* Tier badge */
.tier-badge {
    display: inline-block;
    padding: 0.45rem 1.4rem;
    border-radius: 30px;
    font-size: 0.95rem;
    font-weight: 700;
    margin-top: 1rem;
}
.tier-Budget     { background: rgba(16, 185, 129, 0.25); color: #34d399 !important; border: 1.5px solid #34d399; }
.tier-Mid-Range  { background: rgba(245, 158, 11, 0.25); color: #fbbf24 !important; border: 1.5px solid #fbbf24; }
.tier-Luxury     { background: rgba(239, 68, 68, 0.25);  color: #f87171 !important; border: 1.5px solid #f87171; }

/* Property cards — sharp, readable text */
.prop-card {
    background: #181534;
    border: 1.5px solid rgba(167, 139, 250, 0.3);
    border-radius: 16px;
    padding: 1.3rem 1.5rem;
    margin-bottom: 1rem;
    transition: all 0.25s;
    position: relative;
}
.prop-card:hover {
    border-color: #a78bfa;
    background: #201c44;
    transform: translateX(4px);
}
.prop-rank {
    position: absolute; top: 1rem; right: 1rem;
    background: linear-gradient(135deg, #4f46e5, #7c3aed);
    color: #ffffff !important; font-size: 0.78rem; font-weight: 800;
    padding: 0.3rem 0.8rem; border-radius: 30px;
}
.prop-location { font-size: 1.15rem; font-weight: 800; color: #ffffff !important; }
.prop-price    { font-size: 1.65rem; font-weight: 800; color: #38bdf8 !important; margin: 0.3rem 0; }
.prop-specs    { font-size: 0.92rem; color: #cbd5e1 !important; font-weight: 500; }
.prop-match    { font-size: 0.88rem; color: #4ade80 !important; margin-top: 0.4rem; font-weight: 700; }

/* Section header */
.section-header {
    display: flex; align-items: center; gap: 0.7rem;
    border-bottom: 2px solid rgba(167, 139, 250, 0.5);
    padding-bottom: 0.6rem; margin: 2rem 0 1.2rem;
}
.section-header h3 { color: #ffffff !important; font-size: 1.25rem; font-weight: 700; margin: 0; }

/* Leaderboard table */
thead tr th { background: #2e2860 !important; color: #ffffff !important; font-weight: 700 !important; }
tbody tr td { color: #ffffff !important; font-size: 0.95rem !important; }
tbody tr:hover td { background: rgba(167, 139, 250, 0.2) !important; }

/* Predict button */
div[data-testid="stForm"] button[kind="primaryFormSubmit"],
.stButton > button {
    background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.8rem 2rem !important;
    font-size: 1.1rem !important;
    font-weight: 800 !important;
    width: 100% !important;
    transition: all 0.3s !important;
    box-shadow: 0 6px 24px rgba(79, 70, 229, 0.6) !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 10px 32px rgba(79, 70, 229, 0.8) !important;
}

/* Tab styling — distinct, bright tabs */
.stTabs [role="tab"] { color: #cbd5e1 !important; font-weight: 600 !important; font-size: 1.05rem !important; }
.stTabs [role="tab"][aria-selected="true"] { color: #ffffff !important; border-bottom: 3px solid #818cf8 !important; }

/* Info / Alert boxes */
.stAlert { background: rgba(30, 27, 75, 0.9) !important; border: 1px solid #6366f1 !important; border-radius: 12px !important; color: #ffffff !important; }
.stAlert p { color: #ffffff !important; font-weight: 500 !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# PATHS — resolve relative to this file so it
# works both locally and on Streamlit Cloud
# ─────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)   # project root

MODELS_DIR   = os.path.join(_ROOT, "models")
MODEL_PATH   = os.path.join(MODELS_DIR, "house_price_model.pkl")
KMEANS_PATH  = os.path.join(MODELS_DIR, "kmeans_model.pkl")
PCA_PATH     = os.path.join(MODELS_DIR, "pca_model.pkl")
CATALOG_PATH = os.path.join(MODELS_DIR, "property_catalog.pkl")
METRICS_PATH = os.path.join(MODELS_DIR, "metrics_summary.json")

LOCATIONS = [
    "Whitefield", "Indiranagar", "Koramangala", "HSR Layout",
    "Electronic City", "Bellandur", "Marathahalli", "Hebbal",
    "Rajajinagar", "Sarjapur Road", "Jayanagar", "Malleshwaram",
    "Yelahanka", "Banashankari", "Thanisandra",
]

ALL_AMENITIES = [
    "Gym", "Swimming Pool", "Clubhouse", "24/7 Security", "Power Backup",
    "Children's Play Area", "Covered Parking", "Landscaped Garden",
    "Tennis Court", "High-Speed Elevators", "Intercom", "Jogging Track",
]

FURNISHING_OPTIONS = ["Semi-Furnished", "Furnished", "Unfurnished"]

# ─────────────────────────────────────────────
# LOAD ARTIFACTS (cached)
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_resource(show_spinner=False)
def load_kmeans():
    return joblib.load(KMEANS_PATH)

@st.cache_resource(show_spinner=False)
def load_catalog():
    return joblib.load(CATALOG_PATH)

@st.cache_resource(show_spinner=False)
def load_metrics():
    with open(METRICS_PATH, "r") as f:
        return json.load(f)

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def build_input_df(area, bhk, bathrooms, prop_age, location,
                   furnishing, amenities_list):
    """Build a single-row DataFrame matching the pipeline's expected input."""
    luxury_score = len(amenities_list) / len(ALL_AMENITIES)
    return pd.DataFrame([{
        "Area":             area,
        "BHK":              bhk,
        "Bathrooms":        bathrooms,
        "Property Age":     prop_age,
        "luxury_score":     luxury_score,
        "Location":         location,
        "Furnishing Status": furnishing,
    }])


def get_market_tier(kmeans_bundle, area, price, luxury_score):
    """Use K-Means cluster to assign a market tier label."""
    try:
        kmeans  = kmeans_bundle["kmeans"]
        scaler  = kmeans_bundle["scaler"]
        lmap    = kmeans_bundle["labels_map"]
        pt      = scaler.transform([[area, price, luxury_score]])
        cid     = int(kmeans.predict(pt)[0])
        return lmap.get(cid, "Mid-Range")
    except Exception:
        if price >= 20_000_000:
            return "Luxury"
        elif price >= 10_000_000:
            return "Mid-Range"
        else:
            return "Budget"


def format_price(rupees):
    if rupees >= 1e7:
        return f"₹ {rupees/1e7:.2f} Cr"
    return f"₹ {rupees/1e5:.1f} L"


def recommend_properties(catalog, area, bhk, location, n=5):
    """Return top-N similar properties using multi-attribute scoring."""
    df = catalog.copy()

    # Similarity components (all normalised 0→1, higher = better match)
    df["score_loc"]  = (df["Location"] == location).astype(float) * 0.40
    df["score_bhk"]  = np.maximum(0, 1 - np.abs(df["BHK"] - bhk) / 4) * 0.25
    df["score_area"] = np.maximum(0, 1 - np.abs(df["Area"] - area) / area) * 0.35
    df["similarity"] = df["score_loc"] + df["score_bhk"] + df["score_area"]

    top = df.nlargest(n, "similarity").reset_index(drop=True)
    return top


def price_per_sqft_label(pps):
    return f"₹ {int(pps):,} / sq.ft"

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 1rem 0 2rem;'>
        <div style='font-size:3rem;'>🏠</div>
        <div style='font-size:1.1rem; font-weight:700; color:#c4b5fd;'>House Price AI</div>
        <div style='font-size:0.8rem; color:rgba(255,255,255,0.5); margin-top:0.3rem;'>
            Bangalore Real Estate Intelligence
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("#### 📍 Location")
    location = st.selectbox("Select Location", LOCATIONS, label_visibility="collapsed")

    st.markdown("#### 🏢 Property Specs")
    area = st.slider("Area (sq. ft.)", 400, 5000, 1200, 50)
    bhk  = st.selectbox("BHK (Bedrooms)", [1, 2, 3, 4, 5], index=1)
    baths = st.selectbox("Bathrooms", [1, 2, 3, 4, 5], index=1)

    st.markdown("#### 📅 Property Age")
    prop_age = st.slider("Age (years)", 0, 30, 5)

    st.markdown("#### 🛋️ Furnishing")
    furnishing = st.radio("Status", FURNISHING_OPTIONS, horizontal=False, label_visibility="collapsed")

    st.markdown("#### ✨ Amenities")
    amenities_selected = []
    col1s, col2s = st.columns(2)
    for i, am in enumerate(ALL_AMENITIES):
        col = col1s if i % 2 == 0 else col2s
        if col.checkbox(am, value=(am in ["Gym", "24/7 Security", "Power Backup"]), key=f"am_{i}"):
            amenities_selected.append(am)

    st.markdown("---")
    predict_btn = st.button("🔮  Predict & Recommend", use_container_width=True)

# ─────────────────────────────────────────────
# MAIN HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div class='main-header'>
    <h1>🏠 AI-Powered House Price Prediction</h1>
    <p>Smart Property Valuation & Recommendation System · Bangalore Real Estate · XGBoost + Deep Learning</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab1, tab2 = st.tabs(["🔮 Prediction & Recommendations", "📊 Model Analytics"])

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 1 — PREDICTION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab1:
    # Quick summary cards
    luxury_score_current = len(amenities_selected) / len(ALL_AMENITIES)
    c1, c2, c3, c4 = st.columns(4)
    def _card(col, label, value, sub=""):
        col.markdown(f"""
        <div class='metric-card'>
            <div class='metric-label'>{label}</div>
            <div class='metric-value'>{value}</div>
            <div class='metric-sub'>{sub}</div>
        </div>""", unsafe_allow_html=True)
    _card(c1, "Area", f"{area:,} sq.ft", "Built-up area")
    _card(c2, "Configuration", f"{bhk} BHK / {baths} Bath", location)
    _card(c3, "Property Age", f"{prop_age} yrs", furnishing)
    _card(c4, "Amenities", f"{len(amenities_selected)}/{len(ALL_AMENITIES)}", f"Luxury score: {luxury_score_current:.0%}")

    st.markdown("<br>", unsafe_allow_html=True)

    if predict_btn:
        with st.spinner("🔮 Analysing property features…"):
            try:
                model   = load_model()
                kmeans_bundle = load_kmeans()
                catalog = load_catalog()

                # Build input & predict
                input_df       = build_input_df(area, bhk, baths, prop_age,
                                                location, furnishing, amenities_selected)
                predicted_price = float(model.predict(input_df)[0])
                predicted_price = max(predicted_price, 1_500_000)

                pps   = predicted_price / area
                tier  = get_market_tier(kmeans_bundle, area, predicted_price, luxury_score_current)
                tier_cls = tier.replace(" ", "-")

                # ── Price Hero ──
                st.markdown(f"""
                <div class='price-hero'>
                    <h2>Estimated Market Value</h2>
                    <div class='price'>{format_price(predicted_price)}</div>
                    <div class='sub'>{price_per_sqft_label(pps)} &nbsp;|&nbsp; {location} &nbsp;|&nbsp; {bhk} BHK</div>
                    <span class='tier-badge tier-{tier_cls}'>{tier} Segment</span>
                </div>
                """, unsafe_allow_html=True)

                # ── Price breakdown columns ──
                r1, r2, r3 = st.columns(3)
                _card(r1, "Predicted Price",      format_price(predicted_price), "Best Estimate")
                _card(r2, "Price / sq.ft",        f"₹ {int(pps):,}", "Market rate")
                _card(r3, "Market Segment",        tier, "Based on K-Means cluster")

                # ── Property Details ──
                st.markdown("""
                <div class='section-header'>
                    <span style='font-size:1.4rem'>🏡</span>
                    <h3>Property Summary</h3>
                </div>""", unsafe_allow_html=True)
                d1, d2 = st.columns(2)
                with d1:
                    st.markdown(f"""
                    | Feature | Value |
                    |---------|-------|
                    | 📍 Location | {location} |
                    | 📐 Area | {area:,} sq. ft. |
                    | 🛏 BHK | {bhk} Bedrooms |
                    | 🚿 Bathrooms | {baths} |
                    """)
                with d2:
                    st.markdown(f"""
                    | Feature | Value |
                    |---------|-------|
                    | 📅 Property Age | {prop_age} years |
                    | 🛋️ Furnishing | {furnishing} |
                    | ✨ Amenities | {len(amenities_selected)} features |
                    | 💎 Luxury Score | {luxury_score_current:.0%} |
                    """)

                # ── Recommendations ──
                st.markdown("""
                <div class='section-header'>
                    <span style='font-size:1.4rem'>🔍</span>
                    <h3>Top 5 Similar Properties</h3>
                </div>""", unsafe_allow_html=True)

                recs = recommend_properties(catalog, area, bhk, location, n=5)

                for rank, (_, row) in enumerate(recs.iterrows(), start=1):
                    match_pct = int(row["similarity"] * 100)
                    amenity_count = str(row.get("Amenities", "")).count(",") + 1 \
                                    if pd.notna(row.get("Amenities", "")) and row.get("Amenities", "") != "" else 0
                    prop_cat = row.get("property_category", "—")
                    rec_price = format_price(row["Price"])
                    rec_pps   = f"₹ {int(row.get('price_per_sqft', row['Price']/row['Area'])):,}/sq.ft"
                    match_bar = "█" * (match_pct // 10) + "░" * (10 - match_pct // 10)

                    st.markdown(f"""
                    <div class='prop-card'>
                        <span class='prop-rank'>#{rank}</span>
                        <div class='prop-location'>📍 {row['Location']}</div>
                        <div class='prop-price'>{rec_price}</div>
                        <div class='prop-specs'>
                            🏠 {int(row['Area']):,} sq.ft &nbsp;|&nbsp;
                            🛏 {int(row['BHK'])} BHK &nbsp;|&nbsp;
                            🚿 {int(row['Bathrooms'])} Bath &nbsp;|&nbsp;
                            📅 {int(row.get('Property Age',0))} yrs &nbsp;|&nbsp;
                            ✨ {amenity_count} amenities &nbsp;|&nbsp;
                            {rec_pps}
                        </div>
                        <div class='prop-match'>
                            ✅ Match Score: {match_pct}%  {match_bar}
                            &nbsp;|&nbsp; Segment: {prop_cat}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

            except Exception as e:
                st.error(f"⚠️ Prediction error: {e}")
                st.info("Ensure the model artifacts are present in the `models/` directory.")

    else:
        # ── Placeholder / landing ──
        st.markdown("""
        <div style='text-align:center; padding: 3rem 1rem; color: rgba(255,255,255,0.45);'>
            <div style='font-size:5rem; margin-bottom:1rem;'>🏡</div>
            <div style='font-size:1.2rem; font-weight:600; color:rgba(255,255,255,0.6);'>
                Configure your property details in the sidebar
            </div>
            <div style='font-size:0.95rem; margin-top:0.5rem;'>
                then click <strong style='color:#a78bfa;'>Predict & Recommend</strong> to see your AI valuation
            </div>
        </div>
        """, unsafe_allow_html=True)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 2 — MODEL ANALYTICS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab2:
    try:
        metrics = load_metrics()
        lb      = metrics.get("leaderboard", [])
        fi      = metrics.get("feature_importances", [])
        best_m  = metrics.get("best_model", "XGBoost")

        # ── Leaderboard ──
        st.markdown("""
        <div class='section-header'>
            <span style='font-size:1.4rem'>🏆</span>
            <h3>Model Performance Leaderboard</h3>
        </div>""", unsafe_allow_html=True)

        lb_df = pd.DataFrame(lb)
        if not lb_df.empty:
            lb_df = lb_df.rename(columns={
                "R2_Score": "R² Score", "CV_R2_Mean": "CV R² Mean",
                "CV_R2_Std": "CV R² Std",
                "MAE": "MAE (₹)", "RMSE": "RMSE (₹)", "MSE": "MSE"
            })
            # Highlight best model
            def _highlight_best(row):
                color = "background: rgba(167,139,250,0.2);" if row.get("Model") == best_m else ""
                return [color] * len(row)

            cols_order = ["Model", "R² Score", "CV R² Mean", "CV R² Std", "MAE (₹)", "MSE", "RMSE (₹)"]
            cols_order = [c for c in cols_order if c in lb_df.columns]
            lb_display = lb_df[cols_order].copy()
            lb_display["R² Score"]   = lb_display["R² Score"].apply(lambda x: f"{x:.4f}")
            lb_display["CV R² Mean"] = lb_display["CV R² Mean"].apply(lambda x: f"{x:.4f}" if isinstance(x, float) else x)
            lb_display["MAE (₹)"]   = lb_display["MAE (₹)"].apply(lambda x: f"₹ {x:,.0f}")
            lb_display["RMSE (₹)"]  = lb_display["RMSE (₹)"].apply(lambda x: f"₹ {x:,.0f}")
            lb_display["MSE"]        = lb_display["MSE"].apply(lambda x: f"{x:.2e}" if isinstance(x, float) else x)
            st.dataframe(lb_display, use_container_width=True, hide_index=True)

            st.success(f"🏆 **Best Model: {best_m}** — selected based on highest R² Score and CV performance.")

        # ── Best-model metric cards ──
        if lb:
            best_row = next((r for r in lb if r["Model"] == best_m), lb[0])
            st.markdown("""
            <div class='section-header'>
                <span style='font-size:1.4rem'>📈</span>
                <h3>Best Model Evaluation Metrics</h3>
            </div>""", unsafe_allow_html=True)
            mc1, mc2, mc3, mc4, mc5 = st.columns(5)
            _card(mc1, "MAE",      f"₹ {best_row['MAE']/1e5:.1f} L",      "Mean Abs Error")
            _card(mc2, "RMSE",     f"₹ {best_row['RMSE']/1e5:.1f} L",     "Root Mean Sq Error")
            _card(mc3, "R² Score", f"{best_row['R2_Score']:.4f}",          "Coefficient of Det.")
            _card(mc4, "CV R²",    f"{best_row['CV_R2_Mean']:.4f}",        "5-Fold Cross-Val")
            _card(mc5, "CV Std",   f"± {best_row['CV_R2_Std']:.4f}",       "Stability")

        # ── Feature Importances ──
        if fi:
            st.markdown("""
            <div class='section-header'>
                <span style='font-size:1.4rem'>🔬</span>
                <h3>Feature Importances (XGBoost)</h3>
            </div>""", unsafe_allow_html=True)
            fi_df = pd.DataFrame(fi).head(12)
            fi_df["Importance %"] = (fi_df["Importance"] * 100).round(2)

            import matplotlib.pyplot as plt
            import matplotlib
            matplotlib.rcParams["font.family"] = "DejaVu Sans"
            fig, ax = plt.subplots(figsize=(9, 5))
            fig.patch.set_facecolor("#1a1a2e")
            ax.set_facecolor("#1a1a2e")
            bars = ax.barh(fi_df["Feature"][::-1], fi_df["Importance %"][::-1],
                           color="#a78bfa", edgecolor="#764ba2", linewidth=0.5)
            ax.set_xlabel("Importance (%)", color="white")
            ax.tick_params(colors="white")
            for spine in ax.spines.values():
                spine.set_edgecolor((1.0, 1.0, 1.0, 0.2))
            ax.set_title("Top Feature Importances", color="white", fontweight="bold", pad=12)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        # ── K-Means cluster ──
        st.markdown("""
        <div class='section-header'>
            <span style='font-size:1.4rem'>🎯</span>
            <h3>K-Means Market Segmentation</h3>
        </div>""", unsafe_allow_html=True)
        ks1, ks2, ks3 = st.columns(3)
        _card(ks1, "💚 Budget Segment",    "< ₹ 60 L",      "Electronic City, Yelahanka")
        _card(ks2, "🟡 Mid-Range Segment", "₹ 60L – 1.5 Cr","HSR Layout, Bellandur")
        _card(ks3, "🔴 Luxury Segment",    "> ₹ 1.5 Cr",    "Indiranagar, Koramangala")

    except Exception as e:
        st.warning(f"Could not load model analytics: {e}")


