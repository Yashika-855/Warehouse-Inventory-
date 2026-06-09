import streamlit as st
import pandas as pd
import joblib
import os
import sys

st.title("Retail Inventory Demand Forecasting")

# --- Robust model loading ---
MODEL_PATH = "inventory_forecast_model.joblib"

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None, f"Model file '{MODEL_PATH}' not found. Please upload it to your repo root."
    try:
        model = joblib.load(MODEL_PATH)
        return model, None
    except AttributeError as e:
        return None, (
            f"**Pickle compatibility error:** The model was saved with a different Python/sklearn version.\n\n"
            f"**Your runtime:** Python {sys.version}\n\n"
            f"**Fix:** Re-train or re-save the model using the same Python and scikit-learn "
            f"versions as this deployment, then re-upload `{MODEL_PATH}`.\n\n"
            f"Technical detail: `{e}`"
        )
    except Exception as e:
        return None, f"Unexpected error loading model: `{e}`"

model, load_error = load_model()

if load_error:
    st.error("⚠️ Could not load model")
    st.markdown(load_error)
    st.info(
        "**Quick fix steps:**\n"
        "1. In your training environment, run:\n"
        "```python\n"
        "import joblib, sklearn\n"
        "print(sklearn.__version__)  # note this version\n"
        "joblib.dump(model, 'inventory_forecast_model.joblib')\n"
        "```\n"
        "2. Add `scikit-learn==<your_version>` to `requirements.txt`\n"
        "3. Re-deploy"
    )
    st.stop()

st.success("✅ Model loaded successfully")

# --- Input form ---
st.header("Enter Product Details")

col1, col2 = st.columns(2)

with col1:
    store_id = st.text_input("Store ID", "S001")
    product_id = st.text_input("Product ID", "P001")
    category = st.selectbox("Category", ["Groceries", "Electronics", "Clothing", "Furniture"])
    region = st.selectbox("Region", ["North", "South", "East", "West"])
    inventory_level = st.number_input("Inventory Level", value=200)
    units_ordered = st.number_input("Units Ordered", value=150)
    demand_forecast = st.number_input("Demand Forecast", value=120)
    price = st.number_input("Price", value=25.5)
    discount = st.number_input("Discount (%)", value=10)

with col2:
    weather = st.selectbox("Weather Condition", ["Sunny", "Rainy", "Cloudy"])
    promotion = st.selectbox("Holiday/Promotion", [0, 1])
    competitor_price = st.number_input("Competitor Pricing", value=24.8)
    seasonality = st.selectbox("Seasonality", ["Summer", "Winter", "Spring", "Autumn"])
    year = st.number_input("Year", value=2026, min_value=2000, max_value=2100)
    month = st.number_input("Month", value=7, min_value=1, max_value=12)
    day = st.number_input("Day", value=15, min_value=1, max_value=31)
    dayofweek = st.number_input("Day Of Week (0=Mon, 6=Sun)", value=2, min_value=0, max_value=6)

# --- Prediction ---
if st.button("Predict Demand", type="primary"):
    input_data = pd.DataFrame({
        "Store ID": [store_id],
        "Product ID": [product_id],
        "Category": [category],
        "Region": [region],
        "Inventory Level": [inventory_level],
        "Units Ordered": [units_ordered],
        "Demand Forecast": [demand_forecast],
        "Price": [price],
        "Discount": [discount],
        "Weather Condition": [weather],
        "Holiday/Promotion": [promotion],
        "Competitor Pricing": [competitor_price],
        "Seasonality": [seasonality],
        "Year": [year],
        "Month": [month],
        "Day": [day],
        "DayOfWeek": [dayofweek]
    })

    try:
        prediction = model.predict(input_data)
        st.success(f"🎯 Predicted Units Sold: **{prediction[0]:.2f}**")
    except Exception as e:
        st.error(f"Prediction failed: `{e}`")
        st.info("Check that your input columns match what the model was trained on.")
