import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

st.set_page_config(page_title="Iris Species Predictor", layout="wide")

@st.cache_data
def load_and_preprocess_raw_data():
    try:
        df = pd.read_csv("iris.csv")
    except FileNotFoundError:
        st.error("⚠️ The file 'iris.csv' was not found. Please ensure it is saved in the same folder as this script.")
        st.stop()
        
    df.columns = [col.lower().replace('.', '_').replace(' ', '_') for col in df.columns]
    
    if 'id' in df.columns:
        df = df.drop('id', axis=1)
        
    if 'variety' in df.columns:
        df = df.rename(columns={'variety': 'species'})
    elif 'class' in df.columns:
        df = df.rename(columns={'class': 'species'})
        
    return df

@st.cache_resource
def train_pipeline_model(df):
    features = [col for col in df.columns if col != 'species']
    X = df[features]
    y = df['species']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    rf_pipeline = Pipeline(steps=[
        ('scaler', StandardScaler()),
        ('model', RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    
    rf_pipeline.fit(X_train, y_train)
    return rf_pipeline, X_test, y_test, features

df = load_and_preprocess_raw_data()
model, X_test, y_test, feature_names = train_pipeline_model(df)

st.title("🌸 Iris Flower Intelligence & Predictive Engine")

analysis_mode = st.selectbox(
    "Select Dashboard Function",
    [
        "Dataset Overview & Averages", 
        "Filter by Species", 
        "Feature Scatter Analysis", 
        "Real-time Species Predictor"
    ]
)

if analysis_mode == "Dataset Overview & Averages":
    st.subheader("Average Measurements by Species")
    
    avg_data = df.groupby('species').mean().reset_index()
    st.dataframe(avg_data)
    
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(avg_data['species'], avg_data.iloc[:, 3], color=['#FF9999', '#66B2FF', '#99FF99'], edgecolor='black')
    ax.set_xlabel("Flower Species")
    ax.set_ylabel("Average Petal Length (cm)")
    ax.set_title("Petal Length Comparison Across Species")
    st.pyplot(fig)

elif analysis_mode == "Filter by Species":
    st.subheader("Dataset Explorer")
    
    species_choice = st.selectbox("Filter by specific species", ["All"] + list(df['species'].unique()))
        
    if species_choice != "All":
        filtered = df[df['species'] == species_choice]
    else:
        filtered = df
        
    st.write(f"Found {len(filtered)} records.")
    st.dataframe(filtered)

elif analysis_mode == "Feature Scatter Analysis":
    st.subheader("Sepal vs. Petal Dimensions")
    
    c1, c2 = st.columns(2)
    with c1:
        x_axis = st.selectbox("X-Axis Feature", feature_names, index=0)
    with c2:
        y_axis = st.selectbox("Y-Axis Feature", feature_names, index=2)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = {'Iris-setosa': 'red', 'Iris-versicolor': 'green', 'Iris-virginica': 'blue',
              'setosa': 'red', 'versicolor': 'green', 'virginica': 'blue'}
              
    for species in df['species'].unique():
        subset = df[df['species'] == species]
        color = colors.get(species, 'purple') 
        ax.scatter(subset[x_axis], subset[y_axis], label=species, color=color, alpha=0.7, edgecolors='k')
        
    ax.set_xlabel(x_axis.replace('_', ' ').title())
    ax.set_ylabel(y_axis.replace('_', ' ').title())
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)
    st.pyplot(fig)

elif analysis_mode == "Real-time Species Predictor":
    st.subheader("AI Classification Engine")
    
    predictions = model.predict(X_test)
    acc = accuracy_score(y_test, predictions)
    
    st.info(f"Current core model structural accuracy: {acc * 100:.2f}%")
    
    with st.form("classification_form"):
        st.write("Adjust the slider dimensions (in cm) to predict the flower species:")
        
        c1, c2 = st.columns(2)
        with c1:
            s_length = st.slider("Sepal Length (cm)", float(df.iloc[:,0].min()), float(df.iloc[:,0].max()), float(df.iloc[:,0].mean()))
            s_width = st.slider("Sepal Width (cm)", float(df.iloc[:,1].min()), float(df.iloc[:,1].max()), float(df.iloc[:,1].mean()))
        with c2:
            p_length = st.slider("Petal Length (cm)", float(df.iloc[:,2].min()), float(df.iloc[:,2].max()), float(df.iloc[:,2].mean()))
            p_width = st.slider("Petal Width (cm)", float(df.iloc[:,3].min()), float(df.iloc[:,3].max()), float(df.iloc[:,3].mean()))
            
        execute = st.form_submit_button("Predict Species")
        
    if execute:
        payload = pd.DataFrame([[s_length, s_width, p_length, p_width]], columns=feature_names)
        
        prediction = model.predict(payload)[0]
        st.success(f"### Predicted Species: 🌺 **{prediction.title()}**")