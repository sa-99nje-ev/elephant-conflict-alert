# File: app/explainability.py

import shap
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st


class ModelExplainabilityEngine:
    def __init__(self, model, X_train):
        self.model = model
        self.explainer = shap.TreeExplainer(model)
        self.X_train = X_train
        
    def generate_shap_explanations(self, X_instance):
        """Explain a single prediction"""
        
        shap_values = self.explainer.shap_values(X_instance)
        
        # For multi-class, take the predicted class
        if len(shap_values.shape) == 3:
            prediction = self.model.predict(X_instance)[0]
            shap_values = shap_values[:, :, prediction]
        
        return shap_values
    
    def create_waterfall_plot(self, X_instance, feature_names):
        """Create SHAP waterfall plot"""
        
        shap_values = self.explainer(X_instance)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        shap.plots.waterfall(shap_values[0], max_display=10, show=False)
        
        return fig
    
    def get_feature_importance(self):
        """Get overall feature importance"""
        
        shap_values = self.explainer.shap_values(self.X_train)
        
        if len(shap_values.shape) == 3:
            shap_values = np.abs(shap_values).mean(axis=2)
        
        importance = np.abs(shap_values).mean(axis=0)
        
        feature_importance = pd.DataFrame({
            'feature': self.X_train.columns,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return feature_importance

# In dashboard.py

def show_explainability_dashboard():
    st.header("🔍 Model Explainability")
    
    # Load model and create explainer
    model = load_trained_model()
    X_train = load_training_data()
    
    explainer = ModelExplainabilityEngine(model, X_train)
    
    # 1. Feature Importance
    st.subheader("Feature Importance")
    
    importance_df = explainer.get_feature_importance()
    
    fig = px.bar(importance_df.head(10), 
                 x='importance', 
                 y='feature',
                 orientation='h',
                 title='Top 10 Features Driving Predictions')
    st.plotly_chart(fig)
    
    # 2. Explain Individual Prediction
    st.subheader("Explain a Prediction")
    
    location = st.selectbox("Select Location", get_locations())
    
    if st.button("Explain Prediction"):
        X_instance = get_current_features(location)
        
        fig = explainer.create_waterfall_plot(X_instance, X_train.columns)
        st.pyplot(fig)
        
        st.info("""
        **How to read this:**
        - Red bars increase risk
        - Blue bars decrease risk
        - Bar size = impact magnitude
        """)