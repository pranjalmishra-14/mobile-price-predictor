import streamlit as st
import numpy as np
import pickle
import matplotlib.pyplot as plt

st.markdown("""
<style>
.main {
    background-color: #0E1117;
}
h1 {
    text-align: center;
}
.stButton>button {
    background-color: #4CAF50;
    color: white;
    font-size: 18px;
    border-radius: 8px;
}
</style>
""", unsafe_allow_html=True)

st.sidebar.markdown("## ⚙ Configuration Panel")
st.sidebar.markdown("Adjust mobile specifications below:")


# Page configuration
st.set_page_config(page_title="Mobile Price Predictor", layout="wide")


# Load model and scaler
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# Load accuracy
with open("accuracy.txt", "r") as f:
    accuracy = f.read()

# Title Section
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>📱 Mobile Price Range Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Predict smartphone price category using Machine Learning</p>", unsafe_allow_html=True)
st.markdown("---")
st.markdown(f"<h4 style='text-align:center;'>Model Accuracy: {accuracy}%</h4>", unsafe_allow_html=True)


# Create 2 columns layout
col1, col2 = st.columns(2)

with col1:
    st.subheader("🔋 Hardware Specs")

    battery_power = st.slider("Battery Power", 500, 5000, 2000)
    ram = st.slider("RAM (MB)", 1000, 8000, 4000)
    int_memory = st.slider("Internal Memory (GB)", 2, 256, 32)
    n_cores = st.slider("Number of Cores", 1, 8, 4)
    mobile_wt = st.slider("Mobile Weight (grams)", 80, 300, 150)

with col2:
    st.subheader("📸 Display & Connectivity")

    px_height = st.slider("Pixel Height", 0, 2000, 1000)
    px_width = st.slider("Pixel Width", 0, 2000, 1500)
    talk_time = st.slider("Talk Time (hours)", 1, 30, 10)

    four_g = st.selectbox("5G Support", ["NO","YES" ])
    four_g = 1 if four_g == "YES" else 0
    three_g = st.selectbox("4G Support", ["NO", "YES"])
    three_g = 1 if three_g == "YES" else 0
    wifi = st.selectbox("WiFi Support", ["NO", "YES"])
    wifi = 1 if wifi == "YES" else 0
    touch_screen = st.selectbox("Touch Screen", ["NO", "YES"])
    touch_screen = 1 if touch_screen == "YES" else 0
    dual_sim = st.selectbox("Dual SIM", ["NO", "YES"])
    dual_sim = 1 if dual_sim == "YES" else 0

    fc = st.slider("Front Camera (MP)", 0, 20, 5)
    pc = st.slider("Primary Camera (MP)", 0, 50, 12)
    clock_speed = st.slider("Clock Speed", 0.5, 3.0, 2.0)
    m_dep = st.slider("Mobile Depth", 0.1, 1.0, 0.5)
    blue = st.selectbox("Bluetooth", ["NO", "YES"])
    blue = 1 if blue == "YES" else 0

st.markdown("---")

# Prediction Button
if st.button("🚀 Predict Price Range"):

    input_data = np.array([[battery_power, blue, clock_speed, dual_sim, fc,
                            four_g, int_memory, m_dep, mobile_wt, n_cores,
                            pc, px_height, px_width, ram, 
                            10, 5,
                            talk_time, three_g, touch_screen, wifi]])

    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)

    # Optional confidence
    try:
        proba = model.predict_proba(input_scaled)
        confidence = np.max(proba) * 100
    except:
        confidence = None

    price_dict = {
        0: "Low Cost",
        1: "Medium Cost",
        2: "High Cost",
        3: "Premium / Very High Cost"
    }

    predicted_label = price_dict[prediction[0]]

    st.success(f"### Predicted Price Range: {predicted_label}")

    if confidence:
        st.info(f"Prediction Confidence: {confidence:.2f}%")

    # ----------------------------
    # 📄 Generate Report
    # ----------------------------

    report = f"""
    📱 MOBILE PRICE PREDICTION REPORT
    -------------------------------------

    🔋 Battery Power: {battery_power} mAh
    🧠 RAM: {ram} MB
    💾 Internal Memory: {int_memory} GB
    ⚙ Cores: {n_cores}
    📦 Weight: {mobile_wt} g
    📷 Front Camera: {fc} MP
    📸 Primary Camera: {pc} MP
    🖥 Resolution: {px_height} x {px_width}
    📡 5G Support: {"YES" if four_g else "NO"}
    📡 4G Support: {"YES" if three_g else "NO"}
    📶 WiFi: {"YES" if wifi else "NO"}
    📲 Touch Screen: {"YES" if touch_screen else "NO"}
    🔵 Bluetooth: {"YES" if blue else "NO"}

    -------------------------------------
    💰 Predicted Category: {predicted_label}
    """

    if confidence:
        report += f"\n📊 Confidence: {confidence:.2f}%"

    # Download Button
    st.download_button(
        label="📥 Download Prediction Report",
        data=report,
        file_name="mobile_price_report.txt",
        mime="text/plain"
    )

# Create 2 tabs instead of 2 checkboxes (cleaner UI)
tab1, tab2 = st.tabs(["All Features", "Top 5 Features"])

importances = model.feature_importances_
features = ["battery_power", "blue", "clock_speed", "dual_sim", "fc",
            "four_g", "int_memory", "m_dep", "mobile_wt", "n_cores",
            "pc", "px_height", "px_width", "ram", "sc_h", "sc_w",
            "talk_time", "three_g", "touch_screen", "wifi"]

# -------- All Features Tab --------
with tab1:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.barh(features, importances)
    ax.invert_yaxis()
    ax.set_title("Feature Importance (All)")
    st.pyplot(fig)

# -------- Top 5 Features Tab --------
with tab2:
    feature_importance = sorted(zip(features, importances),
                                key=lambda x: x[1], reverse=True)[:5]

    names = [x[0] for x in feature_importance]
    values = [x[1] for x in feature_importance]

    fig, ax = plt.subplots()
    ax.barh(names, values)
    ax.invert_yaxis()
    ax.set_title("Top 5 Important Features")
    st.pyplot(fig)

st.markdown("---")
st.subheader("📱 Device Summary")

summary_col1, summary_col2, summary_col3 = st.columns(3)

with summary_col1:
    st.metric("RAM (MB)", ram)

with summary_col2:
    st.metric("Battery Power", battery_power)

with summary_col3:
    st.metric("Storage (GB)", int_memory)

st.markdown("""
<hr>
<p style='text-align:center; font-size:14px;'>
Developed using ML and Streamlit
</p>
""", unsafe_allow_html=True)
