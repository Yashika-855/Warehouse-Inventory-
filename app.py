# ── MUST be the absolute first Streamlit call ─────────────────────────────────
import streamlit as st
st.set_page_config(
    page_title="Retail Demand Forecasting",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Standard imports (after set_page_config) ──────────────────────────────────
import pandas as pd
import joblib
import os
import sys

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@500;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
#MainMenu, footer, header { visibility: hidden; }
.stApp { background: #0f1117; color: #e2e8f0; }

.hero {
    background: linear-gradient(135deg, #1e293b 0%, #0f172a 60%, #1a1040 100%);
    border: 1px solid #334155; border-radius: 16px;
    padding: 2.4rem 2.5rem 2rem; margin-bottom: 2rem; position: relative; overflow: hidden;
}
.hero::before {
    content: ""; position: absolute; top: -60px; right: -60px;
    width: 220px; height: 220px;
    background: radial-gradient(circle, rgba(99,102,241,0.18) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-label { font-size:0.72rem; font-weight:700; letter-spacing:0.14em;
    text-transform:uppercase; color:#818cf8; margin-bottom:0.4rem; }
.hero-title { font-size:2rem; font-weight:700; color:#f1f5f9; line-height:1.2; margin:0 0 0.6rem; }
.hero-sub   { font-size:0.9rem; color:#94a3b8; margin:0; max-width:520px; }

.sec { font-size:0.7rem; font-weight:700; letter-spacing:0.14em; text-transform:uppercase;
    color:#6366f1; margin:2rem 0 0.8rem; padding-bottom:0.4rem; border-bottom:1px solid #1e293b; }

div[data-testid="stNumberInput"] input,
div[data-testid="stTextInput"]   input {
    background:#0f172a !important; border:1px solid #334155 !important;
    border-radius:8px !important; color:#e2e8f0 !important;
    font-family:'JetBrains Mono',monospace !important; font-size:0.88rem !important; }
div[data-testid="stSelectbox"] > div > div {
    background:#0f172a !important; border:1px solid #334155 !important;
    border-radius:8px !important; color:#e2e8f0 !important; }
label { color:#94a3b8 !important; font-size:0.82rem !important; font-weight:500 !important; }

div[data-testid="stButton"] > button[kind="primary"] {
    background: linear-gradient(135deg,#6366f1,#8b5cf6) !important;
    border:none !important; border-radius:10px !important; color:#fff !important;
    font-weight:700 !important; font-size:1rem !important; padding:0.75rem 2.5rem !important;
    box-shadow:0 4px 24px rgba(99,102,241,0.35) !important; }

.result-box {
    background: linear-gradient(135deg,#1a2744,#0f172a);
    border:1px solid #6366f1; border-radius:14px;
    padding:2rem 2.4rem; text-align:center; margin-top:1.5rem; }
.result-label { font-size:0.75rem; font-weight:700; letter-spacing:0.12em;
    text-transform:uppercase; color:#818cf8; margin-bottom:0.4rem; }
.result-value { font-family:'JetBrains Mono',monospace; font-size:3.2rem;
    font-weight:700; color:#a5b4fc; line-height:1; }
.result-unit  { font-size:0.85rem; color:#64748b; margin-top:0.5rem; }

hr { border-color:#1e293b !important; margin:1.5rem 0 !important; }
div[data-testid="stAlert"] { border-radius:10px !important; font-size:0.88rem !important; }
</style>
""", unsafe_allow_html=True)


# ── Model loading (cached so it only runs once) ───────────────────────────────
MODEL_PATH = "inventory_forecast_model.joblib"

@st.cache_resource(show_spinner="Loading model…")
def load_model():
    """Safe model loader — returns (model, error_message)."""
    if not os.path.exists(MODEL_PATH):
        return None, (
            f"Model file **`{MODEL_PATH}`** not found in the repo root.\n\n"
            "Make sure `inventory_forecast_model.joblib` is committed to your GitHub repo "
            "alongside `app.py`."
        )
    try:
        mdl = joblib.load(MODEL_PATH)
        return mdl, None
    except AttributeError as e:
        return None, (
            f"**Pickle / version mismatch.**\n\n"
            f"The model was saved with a different Python or scikit-learn version.\n\n"
            f"- Runtime: `Python {sys.version}`\n"
            f"- Fix: retrain and re-save the model in **this same environment**, then recommit.\n\n"
            f"Detail: `{e}`"
        )
    except Exception as e:
        return None, f"Unexpected error loading model: `{e}`"


# ── Hero banner ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div class="hero-label">📦 Retail Intelligence</div>
  <div class="hero-title">Inventory Demand Forecasting</div>
  <p class="hero-sub">
    Enter product and store details to predict units sold for any given day.
    Powered by a trained machine-learning model.
  </p>
</div>
""", unsafe_allow_html=True)


# ── Load model (deferred — after hero renders so users see something) ─────────
model, load_error = load_model()

if load_error:
    st.error("⚠️ Model could not be loaded", icon="🚫")
    st.markdown(load_error)
    with st.expander("📋 How to fix — step by step"):
        st.markdown("""
**1. Find your scikit-learn version** (run in the environment where you trained the model):
```python
import sklearn, joblib
print(sklearn.__version__)          # e.g. 1.4.2
joblib.dump(model, 'inventory_forecast_model.joblib')
```

**2. Pin that version in `requirements.txt`:**
```
streamlit>=1.35.0
pandas>=2.0.0
scikit-learn==1.4.2          # ← use YOUR version here
joblib>=1.3.0
numpy>=1.24.0
```

**3. Commit both `app.py` and `requirements.txt` to GitHub, then redeploy.**
""")
    st.stop()

st.success("✅ Model loaded and ready", icon="🟢")
st.markdown("---")


# ── Input form ────────────────────────────────────────────────────────────────
st.markdown('<div class="sec">🏪 Store &amp; Product</div>', unsafe_allow_html=True)
c1, c2, c3 = st.columns(3)

with c1:
    store_id       = st.text_input("Store ID", value="S001")
    category       = st.selectbox("Category", ["Groceries", "Electronics", "Clothing", "Furniture"])
    inventory_level = st.number_input("Inventory Level", min_value=0, value=200, step=1)
    units_ordered  = st.number_input("Units Ordered", min_value=0, value=150, step=1)

with c2:
    product_id     = st.text_input("Product ID", value="P001")
    region         = st.selectbox("Region", ["North", "South", "East", "West"])
    demand_forecast = st.number_input("Demand Forecast", min_value=0, value=120, step=1)
    price          = st.number_input("Price ($)", min_value=0.0, value=25.50, step=0.01, format="%.2f")

with c3:
    weather        = st.selectbox("Weather Condition", ["Sunny", "Rainy", "Cloudy"])
    seasonality    = st.selectbox("Seasonality", ["Summer", "Winter", "Spring", "Autumn"])
    discount       = st.number_input("Discount (%)", min_value=0.0, max_value=100.0, value=10.0, step=0.5)
    competitor_price = st.number_input("Competitor Price ($)", min_value=0.0, value=24.80, step=0.01, format="%.2f")

st.markdown('<div class="sec">📅 Promotion &amp; Date</div>', unsafe_allow_html=True)
d1, d2, d3, d4, d5 = st.columns(5)

with d1:
    promotion  = st.selectbox("Holiday / Promotion",
                              options=[0, 1], format_func=lambda x: "Yes" if x else "No")
with d2:
    year       = st.number_input("Year",        min_value=2000, max_value=2100, value=2026, step=1)
with d3:
    month      = st.number_input("Month",       min_value=1,    max_value=12,   value=7,    step=1)
with d4:
    day        = st.number_input("Day",         min_value=1,    max_value=31,   value=15,   step=1)
with d5:
    dayofweek  = st.number_input("Day of Week", min_value=0,    max_value=6,    value=2,    step=1,
                                 help="0 = Monday … 6 = Sunday")

st.markdown("---")

# ── Predict ───────────────────────────────────────────────────────────────────
_, btn_col, _ = st.columns([2, 1, 2])
with btn_col:
    predict = st.button("🔮  Predict Demand", type="primary", use_container_width=True)

if predict:
    input_data = pd.DataFrame([{
        "Store ID":           store_id,
        "Product ID":         product_id,
        "Category":           category,
        "Region":             region,
        "Inventory Level":    int(inventory_level),
        "Units Ordered":      int(units_ordered),
        "Demand Forecast":    int(demand_forecast),
        "Price":              float(price),
        "Discount":           float(discount),
        "Weather Condition":  weather,
        "Holiday/Promotion":  int(promotion),
        "Competitor Pricing": float(competitor_price),
        "Seasonality":        seasonality,
        "Year":               int(year),
        "Month":              int(month),
        "Day":                int(day),
        "DayOfWeek":          int(dayofweek),
    }])

    try:
        with st.spinner("Running prediction…"):
            pred = model.predict(input_data)[0]

        st.markdown(f"""
<div class="result-box">
  <div class="result-label">Predicted Units Sold</div>
  <div class="result-value">{pred:,.1f}</div>
  <div class="result-unit">{category} · {region} region · {year}-{month:02d}-{day:02d}</div>
</div>
""", unsafe_allow_html=True)

        st.markdown('<div class="sec">📋 Input Summary</div>', unsafe_allow_html=True)
        st.dataframe(input_data.T.rename(columns={0: "Value"}), use_container_width=True)

    except ValueError as e:
        st.error(f"**Column mismatch:** `{e}`", icon="❌")
        st.info("The input columns must exactly match what the model was trained on — "
                "same names, same order, same data types.")
    except Exception as e:
        st.error(f"**Prediction failed:** `{e}`", icon="❌")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption("Retail Demand Forecasting · scikit-learn + Streamlit · © 2026")
