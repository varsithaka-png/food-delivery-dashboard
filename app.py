import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# ================= UI STYLING =================
st.markdown("""
<style>
.kpi-card {
    padding: 20px;
    border-radius: 12px;
    color: white;
    text-align: center;
    font-size: 18px;
    font-weight: bold;
}

.kpi-title {
    font-size: 16px;
    opacity: 0.8;
}

.kpi-value {
    font-size: 28px;
    margin-top: 10px;
}

.card1 { background: linear-gradient(135deg, #667eea, #764ba2); }
.card2 { background: linear-gradient(135deg, #f7971e, #ffd200); }
.card3 { background: linear-gradient(135deg, #00c6ff, #0072ff); }
</style>
""", unsafe_allow_html=True)

# Page config
st.set_page_config(page_title="Smart Food Delivery Dashboard", layout="wide")

# Title
st.title("Smart Food Delivery Analytics and Prediction System")

# Load data
df = pd.read_csv("data.csv")
df.columns = df.columns.str.strip().str.replace(r'[^A-Za-z0-9]', '', regex=True)

# ================= SIDEBAR =================
st.sidebar.header("Filter Data")

restaurant = st.sidebar.selectbox(
    "Select Restaurant",
    ["All"] + list(df["Restaurant"].unique())
)

if restaurant != "All":
    df = df[df["Restaurant"] == restaurant]

if df.empty:
    st.warning("No data available for selected filter")
    st.stop()

# ================= KPI CARDS =================
st.markdown("---")
st.subheader("Key Metrics")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(f"""
    <div class="kpi-card card1">
        <div class="kpi-title">Total Orders</div>
        <div class="kpi-value">📦 {len(df)}</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="kpi-card card2">
        <div class="kpi-title">Avg Delivery Time</div>
        <div class="kpi-value">⏱ {round(df['DeliveryTime'].mean(), 2)} min</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="kpi-card card3">
        <div class="kpi-title">Avg Rating</div>
        <div class="kpi-value">⭐ {round(df['Rating'].mean(), 2)}</div>
    </div>
    """, unsafe_allow_html=True)

# ================= CHARTS =================
st.markdown("---")

col_left, col_right = st.columns(2)

with col_left:
    st.subheader("Peak Order Time")
    peak = df["OrderTime"].value_counts().sort_index()
    st.bar_chart(peak)

with col_right:
    st.subheader("Popular Food Items")
    food = df["FoodItem"].value_counts()
    st.bar_chart(food)

# Distribution
st.subheader("Delivery Time Distribution")
fig, ax = plt.subplots()
ax.hist(df["DeliveryTime"], bins=10)
ax.set_title("Delivery Time Distribution")
ax.set_xlabel("Minutes")
ax.set_ylabel("Orders")
st.pyplot(fig)

# ================= MACHINE LEARNING =================
st.markdown("---")
st.subheader("Machine Learning Model")
st.write("Model Used: Random Forest Classifier")

# Load full dataset
full_df = pd.read_csv("data.csv")
full_df.columns = full_df.columns.str.strip().str.replace(r'[^A-Za-z0-9]', '', regex=True)

# Convert categorical
full_df["TrafficLevel"] = full_df["TrafficLevel"].map({
    "Low": 0, "Medium": 1, "High": 2
})

full_df["Weather"] = full_df["Weather"].map({
    "Clear": 0, "Cloudy": 1, "Rainy": 2
})

# Target
full_df["Delay"] = full_df["DeliveryTime"].apply(lambda x: 1 if x > 40 else 0)

# Features
X = full_df[[
    "OrderTime",
    "Distance",
    "TrafficLevel",
    "Weather",
    "RiderExperience"
]]
y = full_df["Delay"]

# Train
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
st.write("Model Accuracy:", round(accuracy * 100, 2), "%")

# Confusion Matrix
st.subheader("Confusion Matrix")

cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

cm_df = pd.DataFrame(
    cm,
    index=["Actual On Time", "Actual Late"],
    columns=["Predicted On Time", "Predicted Late"]
)

st.dataframe(cm_df)

# ================= FEATURE IMPORTANCE =================
st.subheader("Feature Importance")

importance = model.feature_importances_
features = X.columns

imp_df = pd.DataFrame({
    "Feature": features,
    "Importance": importance
}).sort_values(by="Importance", ascending=False)

st.bar_chart(imp_df.set_index("Feature"))

# ================= PREDICTION =================
st.markdown("---")
st.subheader("Predict Delivery Status")

col1, col2 = st.columns(2)

with col1:
    order_time = st.number_input("Order Time (Hour)", 0, 23, 18)
    distance = st.number_input("Distance (km)", 1, 15, 5)

with col2:
    traffic = st.selectbox("Traffic Level", ["Low", "Medium", "High"])
    weather = st.selectbox("Weather", ["Clear", "Cloudy", "Rainy"])
    rider = st.slider("Rider Experience (1-5)", 1, 5, 3)

traffic_map = {"Low": 0, "Medium": 1, "High": 2}
weather_map = {"Clear": 0, "Cloudy": 1, "Rainy": 2}

if st.button("Predict"):
    result = model.predict([[
        order_time,
        distance,
        traffic_map[traffic],
        weather_map[weather],
        rider
    ]])

    if result[0] == 1:
        st.error("Delivery will be Late")
    else:
        st.success("Delivery will be On Time")

# ================= PERFORMANCE =================
st.markdown("---")
st.subheader("Restaurant Performance")

avg_time = df.groupby("Restaurant")["DeliveryTime"].mean()
st.bar_chart(avg_time)

# ================= INSIGHTS =================
st.subheader("Insights")

slow = df.loc[df["DeliveryTime"].idxmax()]
fast = df.loc[df["DeliveryTime"].idxmin()]

st.write("Slowest Delivery:", slow["Restaurant"], "-", slow["DeliveryTime"], "minutes")
st.write("Fastest Delivery:", fast["Restaurant"], "-", fast["DeliveryTime"], "minutes")