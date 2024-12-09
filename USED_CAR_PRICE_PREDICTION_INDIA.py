import streamlit as st
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Load the dataset
file_path = "Car_Data_Zscore.csv"  # Update this path if needed
data = pd.read_csv(file_path)

# Features and Target
X = data.drop("Price(Lakhs)", axis=1)
y = data["Price(Lakhs)"]

# Define preprocessing
categorical_features = ["Make", "Model", "Location", "Fuel_Type", "Transmission", "Owner_Type"]
numerical_features = ["Year", "Kilometers_Driven", "Mileage(KMPL)", "Engine(CC)", "Power(BHP)", "Seats"]

categorical_transformer = OneHotEncoder(handle_unknown="ignore")
numerical_transformer = StandardScaler()

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numerical_transformer, numerical_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)

# Define the pipeline
model = Pipeline(
    steps=[("preprocessor", preprocessor), ("regressor", RandomForestRegressor(random_state=42))]
)

# Train the model
model.fit(X, y)

# Streamlit App
st.title("Used Car Price Prediction")
make = st.selectbox("Make", options=data["Make"].unique())
filtered_models = data[data["Make"] == make]["Model"].unique()
model_name = st.selectbox("Model", options=filtered_models)
location = st.selectbox("Location", options=data["Location"].unique())
year = st.number_input("Year", min_value=1980, max_value=2023, step=1)
fuel_type = st.selectbox("Fuel Type", options=data["Fuel_Type"].unique())
kilometers_driven = st.number_input("Kilometers Driven", min_value=0, step=100)
transmission = st.selectbox("Transmission", options=data["Transmission"].unique())
owner_type = st.selectbox("Owner Type", options=data["Owner_Type"].unique())
mileage = st.number_input("Mileage (KMPL)", min_value=0.0, step=0.1)
engine = st.number_input("Engine (CC)", min_value=500.0, step=100.0)
power = st.number_input("Power (BHP)", min_value=20.0, step=5.0)
seats = st.number_input("Seats", min_value=2, max_value=10, step=1)

# Predict button
if st.button("Predict Price"):
    input_data = pd.DataFrame({
        "Make": [make],
        "Model": [model_name],
        "Location": [location],
        "Year": [year],
        "Fuel_Type": [fuel_type],
        "Kilometers_Driven": [kilometers_driven],
        "Transmission": [transmission],
        "Owner_Type": [owner_type],
        "Mileage(KMPL)": [mileage],
        "Engine(CC)": [engine],
        "Power(BHP)": [power],
        "Seats": [seats],
    })
    prediction = model.predict(input_data)
    st.success(f"Predicted Price: ₹{prediction[0]:,.2f} Lakhs")
