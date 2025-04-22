import streamlit as st
import pandas as pd
import joblib

# Load model and features
model = joblib.load("xgb_best_model_v2.pkl")
columns = joblib.load("xgb_feature_columns_v2.pkl")

st.set_page_config(page_title="Trumbull Home Price Estimator", layout="centered")
st.title("🏡 Trumbull County Home Price Estimator")

st.markdown("Predict the estimated sale price of a single-family home using machine learning and real sales data.")

# --- Input Section ---
col1, col2 = st.columns(2)

col1, col2 = st.columns(2)

with col1:
    acres = st.number_input("Lot Size (Acres)", value=0.48)
    year_built = st.number_input("Year Built", min_value=1900, max_value=2025, value=1962)
    bedrooms = st.slider("Bedrooms", 1, 6, 3)
    full_baths = st.slider("Full Baths", 1, 3, 1)

with col2:
    half_baths = st.slider("Half Baths", 0, 2, 0)
    square_ft = st.number_input("Square Footage", value=1438)
    appraised = st.number_input("Appraised Value ($)", value=148100)
    parcels = st.selectbox("Number of Parcels", [1, 2], index=0)




# --- Categorical Inputs ---
neighborhood_opts = [col for col in columns if col.startswith("Neighborhood_")]
school_opts = [col for col in columns if col.startswith("School_District_")]

# Default selection index based on matching strings
neighborhood_default = next((i for i, col in enumerate(neighborhood_opts) if "41202 - WARREN CITY" in col), 0)
school_default = next((i for i, col in enumerate(school_opts) if "WARREN CSD" in col), 0)

neighborhood = st.selectbox("Neighborhood", neighborhood_opts, index=neighborhood_default)
school = st.selectbox("School District", school_opts, index=school_default)


# --- Predict ---
if st.button("🔍 Predict Sale Price"):
    input_data = {col: 0 for col in columns}
    input_data.update({
        'Acres': acres,
        'Appraised_Value': appraised,
        'Year_Built': year_built,
        'Bedrooms': bedrooms,
        'Full_Baths': full_baths,
        'Half_Baths': half_baths,
        'Square_Ft': square_ft,
        'Number_Of_Parcels': parcels,
        neighborhood: 1,
        school: 1
    })

    df = pd.DataFrame([input_data])
    #df = df[columns]  #Fix for column error
    prediction = model.predict(df)[0]
    st.success(f"🏷️ Estimated Sale Price: ${prediction:,.2f}")
