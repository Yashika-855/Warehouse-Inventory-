import streamlit as st
import pandas as pd
import joblib
import os
import sys

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Retail Demand Forecasting",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

/* Base */
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* Hide default Streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }

/* Page background */
.stApp { background: #0f1117; color: #e2e8f0; }

/* Hero banner */
.hero {
    background: linear-gradient(135deg, #1e293b 0%, #0f172a 60%, #1a1040 100%);
    border: 1px solid #334155;
    border-radius: 16px;
    padding: 2.5rem 2.5rem 2rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: "";
    position: absolute;
    top: -60px; right: -60px;
    width: 220px; height: 220px;
    background: radial-gradient(circle, rgba(99,102,241,0.18) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-label {
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #818cf8;
    margin-bottom: 0.5rem;
}
.hero-title {
    font-size: 2rem;
    font-weight: 700;
    color: #f1f5f9;
    line-height: 1.2;
    margin: 0 0 0.6rem;
}
.hero-sub {
    font-size: 0.92rem;
    color: #94a3b8;
    margin: 0;
    max-width: 520px;
}

/* Section heading */
.section-head {
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #6366f1;
    margin: 2rem 0 1rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid #1e293b;
}

/* Input card */
.input-card {
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1rem;
}

/* Streamlit widget tweaks */
div[data-testid="stNumberInput"] input,
div[data-testid="stTextInput"] input {
    background: #0f172a !important;
    border: 1px solid #334155 !important;
    border-radius: 8px !important;
    color: #e2e8f0 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.88rem !important;
}
div[data-testid="stSelectbox"] > div > div {
    background: #0f172a !important;
    border: 1px solid #334155 !important;
    border-radius: 8px !important;
    color: #e2e8f0 !important;
}
label { color: #94a3b8 !important; font-size: 0.82rem !important; font-weight: 500 !important; }

/* Primary button */
div[data-testid="stButton"] > button[kind="primary"] {
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
    border: none !important;
    border-radius: 10px !important;
    color: #fff !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    padding: 0.75rem 2.5rem !important;
    letter-spacing: 0.02em !important;
    transition: opacity 0.2s !important;
    box-shadow: 0 4px 24px rgba(99,102,241,0.3) !important;
}
div[data-testid="stButton"] > button[kind="primary"]:hover { opacity: 0.88 !important; }

/* Result box */
.result-box {
    background: linear-gradient(135deg, #1a2744, #0f172a);
    border: 1px solid #6366f1;
    border-radius: 14px;
    padding: 2rem 2.4rem;
    text-align: center;
    margin-top: 1.5rem;
}
.result-label {
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #818cf8;
    margin-bottom: 0.4rem;
}
.result-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 3rem;
    font-weight: 700;
    color: #a5b4fc;
    line-height: 1;
}
.result-unit {
    font-size: 0.85rem;
    color: #64748b;
    margin-top: 0.4rem;
}

/* Alert overrides */
div[data-testid="stAlert"] {
    border-radius: 10px !important;
    font-size: 0.88rem !important;
}

/* Divider */
hr { border-color: #1e293b !important; margin: 1.5rem 0 !important; }
</style>
""", unsafe_allow_html=True)


# ── Model loading ─────────────────────────────────────────────────────────────
MODEL_PATH = "inventory_forecast_model.joblib"

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None, (
            f"Model file **`{MODEL_PATH}`** not found. "
            "Place it in the same directory as `app.py` and restart."
        )
    try:
        return joblib.load(MODEL_PATH), None
    except AttributeError as e:
        return None, (
            f"**Pickle compatibility error.** The model was saved with a different "
            f"Python / scikit-learn version.\n\n"
            f"- Your runtime: `Python {sys.version}`\n"
            f"- Fix: retrain and re-save the model in this environment, then re-upload "
            f"`{MODEL_PATH}`.\n\n"
            f"Technical detail: `{e}`"
        )
    except Exception as e:
        return None, f"Unexpected error loading model: `{e}`"


model, load_error = load_model()


# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div class="hero-label">📦 Retail Intelligence</div>
  <div class="hero-title">Inventory Demand Forecasting</div>
  <p class="hero-sub">
    Enter product and store details below to predict units sold for any given day.
    Forecasts are powered by a trained machine-learning model.
  </p>
</div>
""", unsafe_allow_html=True)


# ── Model error state ─────────────────────────────────────────────────────────
if load_error:
    st.error("⚠️ Model could not be loaded", icon="🚫")
    st.markdown(load_error)
    with st.expander("How to fix this"):
        st.markdown("""
**Step 1 — Check scikit-learn version in your training environment:**
```python
import sklearn, joblib
print(sklearn.__version__)
joblib.dump(model, 'inventory_forecast_model.joblib')
```

**Step 2 — Pin the version in `requirements.txt`:**
```
scikit-learn==<your_version>
joblib>=1.3
```

**Step 3 — Re-upload `inventory_forecast_model.joblib` and restart the app.**
""")
    st.stop()

st.success("✅ Model loaded and ready", icon="🟢")
st.markdown("---")


# ── Input form ────────────────────────────────────────────────────────────────
st.markdown('<div class="section-head">Store &amp; Product</div>', unsafe_allow_html=True)
col1, col2, col3 = st.columns(3)

with col1:
    store_id = st.text_input("Store ID", value="S001", placeholder="e.g. S001")
    category = st.selectbox("Category", ["Groceries", "Electronics", "Clothing", "Furniture"])
    inventory_level = st.number_input("Inventory Level", min_value=0, value=200, step=1)
    units_ordered = st.number_input("Units Ordered", min_value=0, value=150, step=1)

with col2:
    product_id = st.text_input("Product ID", value="P001", placeholder="e.g. P001")
    region = st.selectbox("Region", ["North", "South", "East", "West"])
    demand_forecast = st.number_input("Demand Forecast", min_value=0, value=120, step=1)
    price = st.number_input("Price ($)", min_value=0.0, value=25.50, step=0.01, format="%.2f")

with col3:
    weather = st.selectbox("Weather Condition", ["Sunny", "Rainy", "Cloudy"])
    seasonality = st.selectbox("Seasonality", ["Summer", "Winter", "Spring", "Autumn"])
    discount = st.number_input("Discount (%)", min_value=0.0, max_value=100.0, value=10.0, step=0.5)
    competitor_price = st.number_input("Competitor Price ($)", min_value=0.0, value=24.80, step=0.01, format="%.2f")

st.markdown('<div class="section-head">Promotion &amp; Date</div>', unsafe_allow_html=True)
col4, col5, col6, col7, col8 = st.columns(5)

with col4:
    promotion = st.selectbox("Holiday / Promotion", options=[0, 1],
                             format_func=lambda x: "Yes (1)" if x else "No (0)")
with col5:
    year = st.number_input("Year", min_value=2000, max_value=2100, value=2026, step=1)
with col6:
    month = st.number_input("Month", min_value=1, max_value=12, value=7, step=1)
with col7:
    day = st.number_input("Day", min_value=1, max_value=31, value=15, step=1)
with col8:
    dayofweek = st.number_input("Day of Week", min_value=0, max_value=6, value=2,
                                help="0 = Monday … 6 = Sunday")

st.markdown("---")

# ── Predict button ────────────────────────────────────────────────────────────
_, btn_col, _ = st.columns([2, 1, 2])
with btn_col:
    predict = st.button("🔮  Predict Demand", type="primary", use_container_width=True)

if predict:
    input_data = pd.DataFrame([{
        "Store ID":           store_id,
        "Product ID":         product_id,
        "Category":           category,
        "Region":             region,
        "Inventory Level":    inventory_level,
        "Units Ordered":      units_ordered,
        "Demand Forecast":    demand_forecast,
        "Price":              price,
        "Discount":           discount,
        "Weather Condition":  weather,
        "Holiday/Promotion":  promotion,
        "Competitor Pricing": competitor_price,
        "Seasonality":        seasonality,
        "Year":               year,
        "Month":              month,
        "Day":                day,
        "DayOfWeek":          dayofweek,
    }])

    try:
        prediction = model.predict(input_data)[0]
        st.markdown(f"""
<div class="result-box">
  <div class="result-label">Predicted Units Sold</div>
  <div class="result-value">{prediction:,.1f}</div>
  <div class="result-unit">units · {category} · {region} region</div>
</div>
""", unsafe_allow_html=True)

        # Quick summary table
        st.markdown('<div class="section-head">Input Summary</div>', unsafe_allow_html=True)
        summary = input_data.T.rename(columns={0: "Value"})
        st.dataframe(summary, use_container_width=True)

    except ValueError as e:
        st.error(f"**Column mismatch:** {e}", icon="❌")
        st.info("Make sure the input features match exactly what the model was trained on. "
                "Check column names, order, and data types.")
    except Exception as e:
        st.error(f"**Prediction failed:** {e}", icon="❌")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption("Retail Demand Forecasting · Powered by scikit-learn & Streamlit")
