import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

st.set_page_config(page_title="Car Analytics & Forecaster", layout="wide")

@st.cache_data
def load_and_preprocess_raw_data():
    try:
        df = pd.read_csv("car data.csv")
    except FileNotFoundError:
        st.error("⚠️ The file 'car data.csv' was not found. Please ensure it is saved in the same folder as this script.")
        st.stop()
        
    current_year = 2026
    df['Brand'] = df['Car_Name'].apply(lambda name: str(name).split()[0] if pd.notnull(name) else 'Unknown')
    df['Age'] = current_year - df['Year']
    df['Kms_Per_Year'] = df['Driven_kms'] / (df['Age'] + 1)
    df['Is_First_Owner'] = df['Owner'].apply(lambda x: 1 if x == 0 else 0)
    
    return df

@st.cache_resource
def train_pipeline_model(df):
    features = ['Brand', 'Age', 'Driven_kms', 'Present_Price', 'Fuel_Type', 'Selling_type', 'Transmission', 'Kms_Per_Year', 'Is_First_Owner']
    X = df[features]
    y = df['Selling_Price']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    numerical_cols = ['Age', 'Driven_kms', 'Present_Price', 'Kms_Per_Year', 'Is_First_Owner']
    categorical_cols = ['Brand', 'Fuel_Type', 'Selling_type', 'Transmission']
    
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])
    
    rf_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', RandomForestRegressor(n_estimators=100, random_state=42))
    ])
    
    rf_pipeline.fit(X_train, y_train)
    return rf_pipeline, X_test, y_test

df = load_and_preprocess_raw_data()
model, X_test, y_test = train_pipeline_model(df)

st.title("Car Market Intelligence & Predictive Engine")

analysis_mode = st.selectbox(
    "Select Dashboard Function",
    [
        "Price Comparison Across Years", 
        "Filter Cars According to Needs", 
        "Price Depreciation Analysis", 
        "Price Predictor Model",
        "Future Resale Value Forecaster"
    ]
)

if analysis_mode == "Price Comparison Across Years":
    st.subheader("Price Trends Over Production Years")
    yearly_data = df.groupby('Year')['Selling_Price'].mean().reset_index()
    
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(yearly_data['Year'].astype(str), yearly_data['Selling_Price'], marker='o', color='teal', linewidth=2)
    ax.set_xlabel("Year of Manufacture")
    ax.set_ylabel("Average Selling Price (Lakhs/Thousands)")
    ax.grid(True, linestyle='--', alpha=0.5)
    st.pyplot(fig)

elif analysis_mode == "Filter Cars According to Needs":
    st.subheader("Inventory Requirements Filter")
    
    c1, c2 = st.columns(2)
    with c1:
        transmission_choice = st.selectbox("Preferred Transmission", ["All", "Manual", "Automatic"])
    with c2:
        kms_choice = st.selectbox("Odometer Cutoff", ["All", "< 5000", "< 10000", "< 20000", "< 40000"])
        
    filtered = df.copy()
    if transmission_choice != "All":
        filtered = filtered[filtered['Transmission'] == transmission_choice]
        
    if kms_choice == "< 5000":
        filtered = filtered[filtered['Driven_kms'] < 5000]
    elif kms_choice == "< 10000":
        filtered = filtered[filtered['Driven_kms'] < 10000]
    elif kms_choice == "< 20000":
        filtered = filtered[filtered['Driven_kms'] < 20000]
    elif kms_choice == "< 40000":
        filtered = filtered[filtered['Driven_kms'] < 40000]
        
    st.write(f"Found {len(filtered)} vehicles matching your criteria.")
    st.dataframe(filtered)

elif analysis_mode == "Price Depreciation Analysis":
    st.subheader("Value Retention Evaluation")
    
    working_df = df.copy()
    working_df['Net_Depreciation'] = working_df['Present_Price'] - working_df['Selling_Price']
    working_df['Retention_Rate_Percentage'] = (working_df['Selling_Price'] / working_df['Present_Price']) * 100
    
    st.dataframe(working_df[['Car_Name', 'Year', 'Present_Price', 'Selling_Price', 'Net_Depreciation', 'Retention_Rate_Percentage']])
    
    fig, ax = plt.subplots(figsize=(10, 4))
    top_depreciated = working_df.nlargest(20, 'Net_Depreciation')
    ax.bar(top_depreciated['Car_Name'], top_depreciated['Net_Depreciation'], color='crimson', alpha=0.85)
    ax.set_xlabel("Vehicle Variant (Top 20 by Depreciation)")
    ax.set_ylabel("Absolute Margin Loss")
    plt.xticks(rotation=45, ha='right')
    st.pyplot(fig)

elif analysis_mode == "Price Predictor Model":
    st.subheader("Real-time Evaluation Engine")
    
    predictions = model.predict(X_test)
    r2 = r2_score(y_test, predictions)
    
    st.info(f"Current core model evaluation structural accuracy (R² Score): {r2:.4f}")
    
    with st.form("valuation_form"):
        car_name = st.text_input("Car Variant Title", value="Toyota Corolla")
        year = st.number_input("Manufacture Calendar Year", min_value=1990, max_value=2026, value=2018)
        present_price = st.number_input("Showroom Original Price (Present_Price)", min_value=0.0, value=15.0)
        driven_kms = st.number_input("Total Distance (Driven_kms)", min_value=0, value=40000)
        fuel = st.selectbox("Engine Fuel Type", ["Petrol", "Diesel", "CNG"])
        s_type = st.selectbox("Commercial Selling Channel", ["Dealer", "Individual"])
        trans = st.selectbox("Gearbox Configuration", ["Manual", "Automatic"])
        owners_count = st.selectbox("Historical Ownership Changes", [0, 1, 2, 3])
        
        execute = st.form_submit_button("Compute Valuations")
        
    if execute:
        current_year = 2026
        brand = str(car_name).split()[0]
        computed_age = current_year - year
        intensity = driven_kms / (computed_age + 1)
        first_owner_flag = 1 if owners_count == 0 else 0
        
        payload = pd.DataFrame({
            'Brand': [brand], 'Age': [computed_age], 'Driven_kms': [driven_kms],
            'Present_Price': [present_price], 'Fuel_Type': [fuel],
            'Selling_type': [s_type], 'Transmission': [trans],
            'Kms_Per_Year': [intensity], 'Is_First_Owner': [first_owner_flag]
        })
        
        out = model.predict(payload)[0]
        st.success(f"Estimated Market Realization Price: {out:,.2f}")

elif analysis_mode == "Future Resale Value Forecaster":
    st.subheader("Predictive Depreciation Forecast Based on Recent Years")
    
    c1, c2 = st.columns(2)
    with c1:
        f_name = st.text_input("Target Model", value="Toyota Corolla")
        f_year = st.number_input("Car Model Year", min_value=1990, max_value=2026, value=2022)
        f_present = st.number_input("Factory Sticker Price", min_value=0.0, value=15.0)
    with c2:
        f_kms = st.number_input("Odometer (Current Kms)", min_value=0, value=20000)
        f_annual = st.number_input("Expected Annual Utilization (Kms)", min_value=0, value=10000)
        f_fuel = st.selectbox("Fuel Category", ["Petrol", "Diesel", "CNG"])
        f_trans = st.selectbox("Transmission Strategy", ["Manual", "Automatic"])
        
    if st.button("Generate Trend Projection"):
        current_year = 2026
        brand = str(f_name).split()[0]
        
        projection_horizon = [2026, 2027, 2028, 2029, 2030]
        valuation_curve = []
        odometer_curve = []
        
        for timeline_year in projection_horizon:
            simulated_age = timeline_year - f_year
            simulated_kms = f_kms + (f_annual * (timeline_year - current_year))
            simulated_intensity = simulated_kms / (simulated_age + 1)
            
            odometer_curve.append(simulated_kms)
            
            evaluation_payload = pd.DataFrame({
                'Brand': [brand], 'Age': [simulated_age], 'Driven_kms': [simulated_kms],
                'Present_Price': [f_present], 'Fuel_Type': [f_fuel],
                'Selling_type': ['Dealer'], 'Transmission': [f_trans],
                'Kms_Per_Year': [simulated_intensity], 'Is_First_Owner': [1]
            })
            
            valuation_curve.append(model.predict(evaluation_payload)[0])
            
        summary_metrics = pd.DataFrame({
            'Target Year': projection_horizon,
            'Projected Total Odometer (Kms)': odometer_curve,
            'Estimated Residual Value': valuation_curve
        })
        
        st.dataframe(summary_metrics.style.format({'Estimated Residual Value': '{:,.2f}'}))
        
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(summary_metrics['Target Year'].astype(str), summary_metrics['Estimated Residual Value'], marker='s', color='purple', linewidth=2)
        ax.fill_between(summary_metrics['Target Year'].astype(str), summary_metrics['Estimated Residual Value'], color='purple', alpha=0.15)
        ax.set_xlabel("Forecast Projection Windows")
        ax.set_ylabel("Predicted Valuation Scale")
        ax.grid(True, linestyle=':', alpha=0.6)
        st.pyplot(fig)