
import streamlit as st
import joblib
import pandas as pd
import numpy as np
import json

st.set_page_config(layout="wide", page_title="Wine Quality Explainability Dashboard")

@st.cache_resource
def load_assets():
    svm = joblib.load('svm_model.pkl')
    mlp = joblib.load('mlp_model.pkl')
    nb = joblib.load('nb_model.pkl')
    scaler = joblib.load('scaler.pkl')
    le = joblib.load('label_encoder.pkl')
    df = pd.read_csv('wine_cleaned.csv')
    with open('feature_names.json') as f:
        feature_names = json.load(f)
    with open('class_names.json') as f:
        class_names = json.load(f)
    with open('model_metrics.json') as f:
        metrics = json.load(f)
    return svm, mlp, nb, scaler, le, df, feature_names, class_names, metrics

svm, mlp, nb, scaler, le, df, feature_names, class_names, metrics = load_assets()

st.sidebar.title("Wine Quality Explainability")

tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Model Comparison", "Explainability", "What-If Simulator"])

with tab1:
    st.header("Dataset Overview")
    col1, col2 = st.columns(2)
    col1.metric("Total Samples", len(df))
    col2.metric("Features", len(feature_names))
    st.dataframe(df.head(10))
    st.subheader("Exploratory Visuals")
    c1, c2 = st.columns(2)
    c1.image('chart_correlation.png', caption="Feature Correlation Heatmap")
    c2.image('chart_correlation_network.png', caption="Correlation Network")
    c3, c4 = st.columns(2)
    c3.image('chart_ridge_alcohol.png', caption="Alcohol Distribution by Quality Tier")
    c4.image('chart_violin_grid.png', caption="Feature Spread by Quality Tier")
    c5, c6 = st.columns(2)
    c5.image('chart_parallel_coords.png', caption="Parallel Coordinates")
    c6.image('chart_type_donut.png', caption="Red vs White Distribution")
    st.image('chart_streamgraph.png', caption="Feature Trends Across Quality Tiers")

with tab2:
    st.header("Model Performance Comparison")
    metric_cols = st.columns(3)
    for i, (name, m) in enumerate(metrics.items()):
        metric_cols[i].metric(name, str(round(m['accuracy']*100,1)) + "%")
    st.image('chart_accuracy.png', caption="Accuracy Comparison")
    st.image('chart_roc_overlay.png', caption="ROC Curve Comparison")
    st.subheader("Confusion Matrices")
    cc1, cc2, cc3 = st.columns(3)
    cc1.image('chart_confmatrix_SVM.png', caption="SVM")
    cc2.image('chart_confmatrix_MLP.png', caption="MLP")
    cc3.image('chart_confmatrix_NaiveBayes.png', caption="Naive Bayes")

with tab3:
    st.header("SHAP Explainability")
    st.subheader("Global Feature Importance")
    e1, e2, e3 = st.columns(3)
    e1.image('chart_shap_beeswarm_mlp.png', caption="MLP")
    e2.image('chart_shap_beeswarm_svm.png', caption="SVM")
    e3.image('chart_shap_beeswarm_nb.png', caption="Naive Bayes")
    st.subheader("Feature Importance Radar")
    st.image('chart_flavor_wheel_radar.png', caption="SHAP Comparison Across Models")
    st.subheader("Feature Dependence")
    d1, d2 = st.columns(2)
    d1.image('chart_shap_dependence_alcohol.png', caption="Alcohol")
    d2.image('chart_shap_dependence_volatile_acidity.png', caption="Volatile Acidity")
    st.subheader("Single Prediction Explanation")
    st.image('chart_shap_waterfall.png', caption="Waterfall Plot")

with tab4:
    st.header("What-If Simulator")
    col1, col2 = st.columns(2)
    with col1:
        fixed_acidity = st.slider("Fixed Acidity", 3.8, 15.9, 7.0)
        volatile_acidity = st.slider("Volatile Acidity", 0.08, 1.6, 0.4)
        citric_acid = st.slider("Citric Acid", 0.0, 1.7, 0.3)
        residual_sugar = st.slider("Residual Sugar", 0.6, 66.0, 5.0)
        chlorides = st.slider("Chlorides", 0.009, 0.61, 0.05)
        free_so2 = st.slider("Free Sulfur Dioxide", 1.0, 290.0, 30.0)
    with col2:
        total_so2 = st.slider("Total Sulfur Dioxide", 6.0, 440.0, 115.0)
        density = st.slider("Density", 0.987, 1.004, 0.996)
        ph = st.slider("pH", 2.7, 4.0, 3.2)
        sulphates = st.slider("Sulphates", 0.2, 2.0, 0.5)
        alcohol = st.slider("Alcohol", 8.0, 14.9, 10.0)
        wine_type = st.selectbox("Type", ["red", "white"])

    model_choice = st.selectbox("Model to use", ["MLP", "SVM", "NaiveBayes"])

    if st.button("Predict"):
        type_white = 1 if wine_type == "white" else 0
        input_row = pd.DataFrame([[fixed_acidity, volatile_acidity, citric_acid, residual_sugar,
            chlorides, free_so2, total_so2, density, ph, sulphates, alcohol, type_white]], columns=feature_names)
        input_scaled = scaler.transform(input_row)
        model_map = {"MLP": mlp, "SVM": svm, "NaiveBayes": nb}
        chosen_model = model_map[model_choice]
        pred = chosen_model.predict(input_scaled)[0]
        pred_proba = chosen_model.predict_proba(input_scaled)[0]
        pred_label = le.inverse_transform([pred])[0]
        st.success("Predicted Quality Tier: " + str(pred_label))
        proba_df = pd.DataFrame({'Tier': class_names, 'Probability': pred_proba})
        st.bar_chart(proba_df.set_index('Tier'))
