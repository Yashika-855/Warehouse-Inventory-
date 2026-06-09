import streamlit as st
import pandas as pd
import pickle

# Load trained model
with open("inventory_forecast_model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("Retail Inventory Demand Forecasting")

st.header("Enter Product Details")

store_id = st.text_input("Store ID", "S001")
product_id = st.text_input("Product ID", "P001")

category = st.selectbox(
    "Category",
    ["Groceries", "Electronics", "Clothing", "Furniture"]
)

region = st.selectbox(
    "Region",
    ["North", "South", "East", "West"]
)

inventory_level = st.number_input("Inventory Level", value=200)
units_ordered = st.number_input("Units Ordered", value=150)
demand_forecast = st.number_input("Demand Forecast", value=120)

price = st.number_input("Price", value=25.5)
discount = st.number_input("Discount (%)", value=10)

weather = st.selectbox(
    "Weather Condition",
    ["Sunny", "Rainy", "Cloudy"]
)

promotion = st.selectbox(
    "Holiday/Promotion",
    [0, 1]
)

competitor_price = st.number_input(
    "Competitor Pricing",
    value=24.8
)

seasonality = st.selectbox(
    "Seasonality",
    ["Summer", "Winter", "Spring", "Autumn"]
)

year = st.number_input("Year", value=2026)
month = st.number_input("Month", value=7)
day = st.number_input("Day", value=15)
dayofweek = st.number_input("Day Of Week", value=2)

if st.button("Predict Demand"):

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

    prediction = model.predict(input_data)

    st.success(
        f"Predicted Units Sold: {prediction[0]:.2f}"
    )
