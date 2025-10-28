"""
Iris Flower Classification App
This Streamlit app predicts the species of iris flowers based on their measurements.
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import plotly.graph_objects as go
import plotly.express as px
import joblib
import os

# Page configuration
st.set_page_config(
    page_title="Iris Flower Classifier",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #4B0082;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #6A5ACD;
        margin-top: 2rem;
    }
    .info-box {
        background-color: rgba(75, 0, 130, 0.1);
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #4B0082;
    }
    </style>
""", unsafe_allow_html=True)

# Title and description
st.markdown('<h1 class="main-header">🌸 Iris Flower Classification App by ARKA</h1>', unsafe_allow_html=True)
st.markdown("""
    <div class="info-box">
    This application uses Machine Learning to classify iris flowers into three species:
    <b>Setosa</b>, <b>Versicolor</b>, and <b>Virginica</b> based on their sepal and petal measurements.
    </div>
""", unsafe_allow_html=True)

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["🏠 Home", "📊 Train Model", "🔮 Make Predictions", "📈 Model Analytics"])

# Initialize session state for model persistence
if 'model' not in st.session_state:
    st.session_state.model = None
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'accuracy' not in st.session_state:
    st.session_state.accuracy = None
if 'X_train' not in st.session_state:
    st.session_state.X_train = None
if 'feature_names' not in st.session_state:
    st.session_state.feature_names = None

# Function to load data
@st.cache_data
def load_data(file):
    """Load data from uploaded file"""
    if file.name.endswith('.csv'):
        return pd.read_csv(file)
    elif file.name.endswith('.xlsx'):
        return pd.read_excel(file)
    return None

# Function to train model
def train_model(X_train, y_train, n_estimators, max_depth):
    """Train Random Forest Classifier"""
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model

# HOME PAGE
if page == "🏠 Home":
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<h2 class="sub-header">About the Dataset</h2>', unsafe_allow_html=True)
        st.write("""
        The **Iris dataset** is one of the most famous datasets in machine learning. It contains measurements 
        of 150 iris flowers from three different species:
        
        - **Iris Setosa** 🌼
        - **Iris Versicolor** 🌺
        - **Iris Virginica** 🌷
        
        Each flower is described by four features:
        1. **Sepal Length** (cm)
        2. **Sepal Width** (cm)
        3. **Petal Length** (cm)
        4. **Petal Width** (cm)
        """)
        
        st.markdown('<h2 class="sub-header">How to Use This App</h2>', unsafe_allow_html=True)
        st.write("""
        1. **Train Model**: Upload your iris dataset and train a Random Forest classifier
        2. **Make Predictions**: Enter flower measurements to predict the species
        3. **Model Analytics**: View detailed performance metrics and visualizations
        """)
    
    with col2:
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/4/41/Iris_versicolor_3.jpg/440px-Iris_versicolor_3.jpg", 
                 caption="Iris Versicolor", use_container_width=True)
        st.info("💡 **Tip**: Start by training the model with your dataset in the 'Train Model' section!")

# TRAIN MODEL PAGE
elif page == "📊 Train Model":
    st.markdown('<h2 class="sub-header">Train Your Model</h2>', unsafe_allow_html=True)
    
    # File uploader
    uploaded_file = st.file_uploader("Upload your Iris dataset (CSV or XLSX)", type=['csv', 'xlsx'])
    
    if uploaded_file is not None:
        data = load_data(uploaded_file)
        
        if data is not None:
            st.success(f"✅ Dataset loaded successfully! Shape: {data.shape}")
            
            # Display data preview
            with st.expander("📋 View Dataset Preview"):
                st.dataframe(data.head(10))
                
            # Display statistics
            with st.expander("📊 Dataset Statistics"):
                st.write(data.describe())
                
            # Display class distribution
            with st.expander("📈 Class Distribution"):
                class_col = data.columns[-1]
                class_counts = data[class_col].value_counts()
                fig = px.bar(x=class_counts.index, y=class_counts.values, 
                            labels={'x': 'Species', 'y': 'Count'},
                            title='Distribution of Iris Species')
                st.plotly_chart(fig, use_container_width=True)
            
            # Model configuration
            st.markdown("### ⚙️ Model Configuration")
            col1, col2 = st.columns(2)
            
            with col1:
                n_estimators = st.slider("Number of Trees", 10, 200, 100, 10)
            with col2:
                max_depth = st.slider("Maximum Depth", 2, 20, 10, 1)
            
            test_size = st.slider("Test Set Size (%)", 10, 40, 20, 5) / 100
            
            # Train button
            if st.button("🚀 Train Model", type="primary"):
                with st.spinner("Training model..."):
                    # Prepare data
                    X = data.iloc[:, :-1]
                    y = data.iloc[:, -1]
                    
                    # Split data
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=test_size, random_state=42, stratify=y
                    )
                    
                    # Train model
                    model = train_model(X_train, y_train, n_estimators, max_depth)
                    
                    # Make predictions
                    y_pred = model.predict(X_test)
                    accuracy = accuracy_score(y_test, y_pred)
                    
                    # Save to session state
                    st.session_state.model = model
                    st.session_state.model_trained = True
                    st.session_state.accuracy = accuracy
                    st.session_state.X_train = X_train
                    st.session_state.y_test = y_test
                    st.session_state.y_pred = y_pred
                    st.session_state.feature_names = X.columns.tolist()
                    
                    st.success(f"✅ Model trained successfully! Accuracy: {accuracy*100:.2f}%")
                    
                    # Display metrics
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Accuracy", f"{accuracy*100:.2f}%")
                    col2.metric("Training Samples", len(X_train))
                    col3.metric("Test Samples", len(X_test))
                    
                    # Confusion matrix
                    st.markdown("### Confusion Matrix")
                    cm = confusion_matrix(y_test, y_pred)
                    fig = px.imshow(cm, text_auto=True, 
                                   labels=dict(x="Predicted", y="Actual"),
                                   x=model.classes_, y=model.classes_)
                    st.plotly_chart(fig, use_container_width=True)

# MAKE PREDICTIONS PAGE
elif page == "🔮 Make Predictions":
    st.markdown('<h2 class="sub-header">Make Predictions</h2>', unsafe_allow_html=True)
    
    if not st.session_state.model_trained:
        st.warning("⚠️ Please train a model first in the 'Train Model' section!")
    else:
        st.success(f"✅ Model loaded! Current accuracy: {st.session_state.accuracy*100:.2f}%")
        
        st.markdown("### 🌸 Enter Flower Measurements")
        
        col1, col2 = st.columns(2)
        
        with col1:
            sepal_length = st.number_input("Sepal Length (cm)", 0.0, 10.0, 5.0, 0.1)
            petal_length = st.number_input("Petal Length (cm)", 0.0, 10.0, 3.0, 0.1)
        
        with col2:
            sepal_width = st.number_input("Sepal Width (cm)", 0.0, 10.0, 3.0, 0.1)
            petal_width = st.number_input("Petal Width (cm)", 0.0, 10.0, 1.0, 0.1)
        
        # Predict button
        if st.button("🔍 Predict Species", type="primary"):
            input_data = [[sepal_length, sepal_width, petal_length, petal_width]]
            prediction = st.session_state.model.predict(input_data)[0]
            probabilities = st.session_state.model.predict_proba(input_data)[0]
            
            # Display results
            st.markdown("### 🎯 Prediction Results")
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.metric("Predicted Species", prediction)
                max_prob = max(probabilities)
                st.metric("Confidence", f"{max_prob*100:.2f}%")
            
            with col2:
                # Probability chart
                prob_df = pd.DataFrame({
                    'Species': st.session_state.model.classes_,
                    'Probability': probabilities
                })
                fig = px.bar(prob_df, x='Species', y='Probability', 
                            title='Prediction Probabilities',
                            color='Probability',
                            color_continuous_scale='viridis')
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
            
            # Display input summary
            with st.expander("📊 Input Summary"):
                input_df = pd.DataFrame({
                    'Feature': ['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width'],
                    'Value (cm)': [sepal_length, sepal_width, petal_length, petal_width]
                })
                st.dataframe(input_df, use_container_width=True)

# MODEL ANALYTICS PAGE
elif page == "📈 Model Analytics":
    st.markdown('<h2 class="sub-header">Model Analytics</h2>', unsafe_allow_html=True)
    
    if not st.session_state.model_trained:
        st.warning("⚠️ Please train a model first in the 'Train Model' section!")
    else:
        # Performance metrics
        st.markdown("### 📊 Performance Metrics")
        col1, col2, col3 = st.columns(3)
        col1.metric("Model Accuracy", f"{st.session_state.accuracy*100:.2f}%")
        col2.metric("Number of Features", 4)
        col3.metric("Number of Classes", 3)
        
        # Feature importance
        st.markdown("### 🎯 Feature Importance")
        if st.session_state.feature_names:
            feature_importance = pd.DataFrame({
                'Feature': st.session_state.feature_names,
                'Importance': st.session_state.model.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            fig = px.bar(feature_importance, x='Importance', y='Feature', 
                        orientation='h', title='Feature Importance Ranking',
                        color='Importance', color_continuous_scale='blues')
            st.plotly_chart(fig, use_container_width=True)
        
        # Classification report
        st.markdown("### 📋 Classification Report")
        report = classification_report(st.session_state.y_test, st.session_state.y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df.style.format("{:.3f}"), use_container_width=True)
        
        # Confusion matrix
        st.markdown("### 🔢 Confusion Matrix")
        cm = confusion_matrix(st.session_state.y_test, st.session_state.y_pred)
        fig = px.imshow(cm, text_auto=True, 
                       labels=dict(x="Predicted Class", y="Actual Class"),
                       x=st.session_state.model.classes_, 
                       y=st.session_state.model.classes_,
                       color_continuous_scale='purples')
        st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #666;'>
    Made with ❤️ using Streamlit | Data Science Project
    </div>
""", unsafe_allow_html=True)