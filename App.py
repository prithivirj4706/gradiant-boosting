import streamlit as st
from model import AdSpendModel

st.set_page_config(page_title="Revenue Predictor", layout="centered")

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #141E30, #243B55);
    color: white;
}
.title {
    text-align: center;
    font-size: 40px;
    font-weight: bold;
    color: #00FFD1;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="title">Ad Spend Revenue Predictor</p>', unsafe_allow_html=True)

ad_spend = st.number_input("Ad Spend ($)", min_value=0.0)

season = st.selectbox(
    "Season",
    ["Spring", "Summer", "Autumn", "Winter"]
)

if st.button("Predict Revenue"):
    model = AdSpendModel()
    result = model.predict(ad_spend, season)
    st.success(f"Predicted Revenue: ${result}")