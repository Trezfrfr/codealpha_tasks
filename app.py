import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder, PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

st.set_page_config(page_title="Iris Predictor Pro", page_icon="🌿", layout="wide")

st.markdown("""
    <style>
    div[data-testid="metric-container"] {
        background-color: #f8f9fa;
        border: 1px solid #e0e0e0;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    </style>
""", unsafe_allow_html=True)

st.title("🌿 Iris Species Predictor Pro")
st.markdown("Adjust the flower measurements to predict the species, or view the model's underlying performance metrics.")
st.divider()

@st.cache_resource
def load_train_and_evaluate():
    df = sns.load_dataset('iris')
    df.rename(columns={
        'sepal_length': 'SepalLength', 'sepal_width': 'SepalWidth',
        'petal_length': 'PetalLength', 'petal_width': 'PetalWidth',
        'species': 'Species'
    }, inplace=True)
    
    X = df[['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth']]
    y = df['Species']
    
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
    
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('poly_features', PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)),
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    
    pipeline.fit(X_train, y_train)
    
    predictions = pipeline.predict(X_test)
    
    accuracy = accuracy_score(y_test, predictions)
    report_dict = classification_report(y_test, predictions, target_names=label_encoder.classes_, output_dict=True)
    cm = confusion_matrix(y_test, predictions)
    
    return pipeline, label_encoder, accuracy, report_dict, cm

pipeline, label_encoder, accuracy, report_dict, cm = load_train_and_evaluate()

tab1, tab2 = st.tabs(["🎯 Live Prediction", "📊 Model Performance"])

with tab1:
    col_inputs, col_results = st.columns([1, 2], gap="large")

    with col_inputs:
        st.subheader("📏 Measurements")
        
        with st.container():
            st.markdown("**Sepal Dimensions**")
            sepal_length = st.slider("Length (cm)", 4.0, 8.0, 5.8, key="sl")
            sepal_width = st.slider("Width (cm)", 2.0, 4.5, 3.0, key="sw")
            
        st.write("")
        
        with st.container():
            st.markdown("**Petal Dimensions**")
            petal_length = st.slider("Length (cm)", 1.0, 7.0, 4.3, key="pl")
            petal_width = st.slider("Width (cm)", 0.1, 2.5, 1.3, key="pw")

    input_data = pd.DataFrame({
        'SepalLength': [sepal_length],
        'SepalWidth': [sepal_width],
        'PetalLength': [petal_length],
        'PetalWidth': [petal_width]
    })

    prediction_encoded = pipeline.predict(input_data)
    prediction_species = label_encoder.inverse_transform(prediction_encoded)[0]
    prediction_proba = pipeline.predict_proba(input_data)[0]

    with col_results:
        st.subheader("Results")
        
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Sepal Length", f"{sepal_length} cm")
        m2.metric("Sepal Width", f"{sepal_width} cm")
        m3.metric("Petal Length", f"{petal_length} cm")
        m4.metric("Petal Width", f"{petal_width} cm")
        
        st.write("")
        
        species_display = {'setosa': '🌸 Setosa', 'versicolor': '🌼 Versicolor', 'virginica': '🌺 Virginica'}
        st.success(f"### The model predicts: **{species_display.get(prediction_species, prediction_species.capitalize())}**")
        
        st.markdown("#### Confidence Breakdown")
        proba_df = pd.DataFrame({
            'Species': [name.capitalize() for name in label_encoder.classes_],
            'Confidence (%)': prediction_proba * 100 
        })
        st.bar_chart(proba_df.set_index('Species'), color="#4CAF50", height=200)

with tab2:
    st.subheader("Testing Set Evaluation (20% of Data)")
    
    st.metric(label="Overall Test Accuracy", value=f"{accuracy * 100:.1f}%")
    st.divider()
    
    col_metrics1, col_metrics2 = st.columns(2)
    
    with col_metrics1:
        st.markdown("#### Confusion Matrix")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=[name.capitalize() for name in label_encoder.classes_],
                    yticklabels=[name.capitalize() for name in label_encoder.classes_], ax=ax)
        plt.ylabel('Actual Species')
        plt.xlabel('Predicted Species')
        st.pyplot(fig)
        
    with col_metrics2:
        st.markdown("#### Classification Report")
        report_df = pd.DataFrame(report_dict).transpose()
        
        styled_report = report_df.style.format(subset=['precision', 'recall', 'f1-score'], formatter="{:.2f}")
        st.dataframe(styled_report, use_container_width=True)
        
        st.info("**Tip:** Look at the 'recall' metric. If it is 1.00, it means the model successfully identified 100% of the flowers belonging to that specific species in the test set.")