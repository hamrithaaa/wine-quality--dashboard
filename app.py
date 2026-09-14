import streamlit as st
import joblib
import pandas as pd
import numpy as np
import json
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(layout="wide", page_title="Wine Quality Explainability Dashboard", page_icon="🍷")

PRIMARY = "#6D2E46"
SECONDARY = "#A26769"
GOLD = "#C9A24B"
CREAM = "#F3ECE2"
TIER_COLORS = {"Low": "#A26769", "Medium": "#6D2E46", "High": "#C9A24B"}
MODEL_COLORS = {"SVM": "#A26769", "MLP": "#6D2E46", "NaiveBayes": "#8C5B4A"}

st.markdown("""
<style>
    .stApp { background-color: #FAF6F0; }
    div[data-testid="stMetric"] {
        background-color: white; border-radius: 10px; padding: 15px;
        border: 1px solid #EAD9CC;
    }
    h1, h2, h3 { color: #2A0E1A; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_assets():
    svm = joblib.load('svm_model.pkl')
    mlp = joblib.load('mlp_model.pkl')
    nb = joblib.load('nb_model.pkl')
    scaler = joblib.load('scaler.pkl')
    le = joblib.load('label_encoder.pkl')
    df = pd.read_csv('wine_cleaned.csv')

    shap_mlp = np.load('shap_values_mlp.npy')
    shap_svm = np.load('shap_values_svm.npy')
    shap_nb = np.load('shap_values_nb.npy')
    X_sample_scaled = np.load('X_test_scaled_sample.npy')

    exp_mlp = np.load('expected_value_mlp.npy')
    exp_svm = np.load('expected_value_svm.npy')
    exp_nb = np.load('expected_value_nb.npy')

    with open('feature_names.json') as f:
        feature_names = json.load(f)
    with open('class_names.json') as f:
        class_names = json.load(f)
    with open('model_metrics.json') as f:
        metrics = json.load(f)

    conf_matrices, roc_data = None, None
    try:
        with open('confusion_matrices.json') as f:
            conf_matrices = json.load(f)
    except FileNotFoundError:
        pass
    try:
        with open('roc_data.json') as f:
            roc_data = json.load(f)
    except FileNotFoundError:
        pass

    X_sample_real = scaler.inverse_transform(X_sample_scaled)

    return (svm, mlp, nb, scaler, le, df, feature_names, class_names, metrics,
            shap_mlp, shap_svm, shap_nb, X_sample_scaled, X_sample_real,
            exp_mlp, exp_svm, exp_nb, conf_matrices, roc_data)


(svm, mlp, nb, scaler, le, df, feature_names, class_names, metrics,
 shap_mlp, shap_svm, shap_nb, X_sample_scaled, X_sample_real,
 exp_mlp, exp_svm, exp_nb, conf_matrices, roc_data) = load_assets()

SHAP_MAP = {"MLP": shap_mlp, "SVM": shap_svm, "NaiveBayes": shap_nb}
EXP_MAP = {"MLP": exp_mlp, "SVM": exp_svm, "NaiveBayes": exp_nb}
high_idx = class_names.index("High") if "High" in class_names else 0

st.sidebar.title("🍷 Wine Quality Explainability")
st.sidebar.markdown("Interactive ML + SHAP dashboard")

tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🤖 Model Comparison", "🔍 Explainability", "🎛️ What-If Simulator"])

# ============================================================
# TAB 1 — OVERVIEW
# ============================================================
with tab1:
    st.header("Dataset Overview")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Samples", len(df))
    col2.metric("Features", len(feature_names))
    col3.metric("Quality Tiers", df['quality_tier'].nunique() if 'quality_tier' in df.columns else 3)

    st.dataframe(df.head(10), use_container_width=True)

    st.subheader("Quality Tier Distribution")
    if 'quality_tier' in df.columns:
        order = ["Low", "Medium", "High"]
        counts = df['quality_tier'].value_counts().reindex(order).reset_index()
        counts.columns = ["Tier", "Count"]
        fig = px.bar(counts, x="Tier", y="Count", color="Tier",
                     color_discrete_map=TIER_COLORS, text="Count")
        fig.update_traces(textposition="outside", hovertemplate="%{x}: %{y} samples<extra></extra>")
        fig.update_layout(showlegend=False, plot_bgcolor="white", paper_bgcolor="white",
                           yaxis_title="Number of samples", xaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Feature Correlation")
        num_df = df.select_dtypes(include=[np.number])
        corr = num_df.corr()
        fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r",
                         aspect="auto", zmin=-1, zmax=1)
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.subheader("Wine Type Split")
        if 'type' in df.columns:
            type_counts = df['type'].value_counts().reset_index()
            type_counts.columns = ["Type", "Count"]
            fig = px.pie(type_counts, names="Type", values="Count", hole=0.45,
                         color="Type", color_discrete_map={"red": "#7A2E3A", "white": GOLD})
            fig.update_traces(textinfo="percent+label", hovertemplate="%{label}: %{value} samples<extra></extra>")
            st.plotly_chart(fig, use_container_width=True)

    st.subheader("Feature Spread by Quality Tier")
    if 'quality_tier' in df.columns:
        feat_choice = st.selectbox("Choose a feature to explore", feature_names, key="violin_feat")
        if feat_choice in df.columns:
            fig = px.violin(df, x="quality_tier", y=feat_choice, color="quality_tier",
                             category_orders={"quality_tier": ["Low", "Medium", "High"]},
                             color_discrete_map=TIER_COLORS, box=True, points=False)
            fig.update_layout(showlegend=False, plot_bgcolor="white", paper_bgcolor="white")
            st.plotly_chart(fig, use_container_width=True)

# ============================================================
# TAB 2 — MODEL COMPARISON
# ============================================================
with tab2:
    st.header("Model Performance Comparison")

    mcols = st.columns(len(metrics))
    for i, (name, m) in enumerate(metrics.items()):
        mcols[i].metric(name, f"{m['accuracy']*100:.1f}%")

    acc_df = pd.DataFrame({
        "Model": list(metrics.keys()),
        "Accuracy": [m['accuracy'] * 100 for m in metrics.values()]
    }).sort_values("Accuracy", ascending=True)
    fig = px.bar(acc_df, x="Accuracy", y="Model", orientation="h", color="Model",
                 color_discrete_map=MODEL_COLORS, text="Accuracy",
                 range_x=[0, 100])
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside",
                       hovertemplate="%{y}: %{x:.1f}% accuracy<extra></extra>")
    fig.update_layout(showlegend=False, plot_bgcolor="white", paper_bgcolor="white",
                       xaxis_title="Accuracy (%)", yaxis_title="")
    st.plotly_chart(fig, use_container_width=True)

    if roc_data:
        st.subheader("ROC Curve Comparison")
        fig = go.Figure()
        for name, rd in roc_data.items():
            fig.add_trace(go.Scatter(x=rd["fpr"], y=rd["tpr"], mode="lines",
                                      name=f"{name} (AUC={rd['auc']:.2f})",
                                      line=dict(color=MODEL_COLORS.get(name, "#888"), width=3),
                                      hovertemplate="FPR: %{x:.2f}<br>TPR: %{y:.2f}<extra>" + name + "</extra>"))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                                  line=dict(dash="dash", color="lightgray"), showlegend=False,
                                  hoverinfo="skip"))
        fig.update_layout(xaxis_title="False Positive Rate", yaxis_title="True Positive Rate",
                           plot_bgcolor="white", paper_bgcolor="white", height=450)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("ROC data not found — run the extra export cell in Colab (confusion_matrices.json / roc_data.json) and add those files to your repo for this chart.")

    st.subheader("Confusion Matrices")
    if conf_matrices:
        model_pick = st.radio("Model", list(conf_matrices.keys()), horizontal=True, key="cm_pick")
        cm = np.array(conf_matrices[model_pick])
        fig = px.imshow(cm, text_auto=True, color_continuous_scale="Purples",
                         x=class_names, y=class_names, aspect="auto")
        fig.update_layout(xaxis_title="Predicted", yaxis_title="Actual", height=450)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Confusion matrix data not found — run the extra export cell in Colab and add confusion_matrices.json to your repo.")

# ============================================================
# TAB 3 — EXPLAINABILITY (SHAP)
# ============================================================
with tab3:
    st.header("SHAP Explainability")
    st.caption("All charts below explain each model's prediction of the 'High' quality tier.")

    model_for_shap = st.selectbox("Choose a model to explain", list(SHAP_MAP.keys()), key="shap_model")
    sv = SHAP_MAP[model_for_shap]
    ev = EXP_MAP[model_for_shap]

    st.subheader(f"Global Feature Importance — {model_for_shap} (Beeswarm)")
    mean_abs = np.abs(sv).mean(axis=0)
    order_idx = np.argsort(mean_abs)
    ordered_features = [feature_names[i] for i in order_idx]

    rows = []
    n_samples = sv.shape[0]
    rng = np.random.default_rng(42)
    for rank, fi in enumerate(order_idx):
        jitter = rng.uniform(-0.35, 0.35, size=n_samples)
        for s in range(n_samples):
            rows.append({
                "Feature": feature_names[fi],
                "y": rank + jitter[s],
                "SHAP value": sv[s, fi],
                "Feature value": X_sample_real[s, fi],
            })
    bee_df = pd.DataFrame(rows)
    fig = px.scatter(bee_df, x="SHAP value", y="y", color="Feature value",
                      color_continuous_scale="RdBu_r",
                      hover_data={"Feature": True, "Feature value": ":.2f", "y": False})
    fig.update_yaxes(tickmode="array", tickvals=list(range(len(ordered_features))),
                      ticktext=ordered_features, title="")
    fig.update_layout(height=500, plot_bgcolor="white", paper_bgcolor="white",
                       coloraxis_colorbar=dict(title="Feature<br>value"))
    fig.add_vline(x=0, line_dash="dash", line_color="gray")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Each dot is one wine. Red = high feature value, blue = low. Right of the line pushes toward High quality; left pushes away.")

    st.subheader("Feature Dependence")
    dep_feat = st.selectbox("Feature", feature_names, key="dep_feat",
                             index=int(order_idx[-1]))
    fi = feature_names.index(dep_feat)
    dep_df = pd.DataFrame({
        "Feature value": X_sample_real[:, fi],
        "SHAP value": sv[:, fi],
    })
    fig = px.scatter(dep_df, x="Feature value", y="SHAP value", color="SHAP value",
                      color_continuous_scale="RdBu_r")
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.update_layout(plot_bgcolor="white", paper_bgcolor="white",
                       xaxis_title=f"{dep_feat} (actual value)", yaxis_title="SHAP value")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Single Prediction Explanation (Waterfall)")
    sample_idx = st.slider("Choose a test wine sample", 0, n_samples - 1, 0, key="wf_sample")

    base_value = float(ev[high_idx]) if hasattr(ev, "__len__") else float(ev)
    contribs = sv[sample_idx]
    top_n = 8
    order = np.argsort(-np.abs(contribs))[:top_n]
    labels = [feature_names[i] for i in order]
    values = [contribs[i] for i in order]
    other = contribs[[i for i in range(len(contribs)) if i not in order]].sum()
    if abs(other) > 1e-6:
        labels.append("Other features")
        values.append(other)

    fig = go.Figure(go.Waterfall(
        orientation="v",
        measure=["relative"] * len(labels),
        x=labels,
        y=values,
        base=base_value,
        text=[f"{v:+.3f}" for v in values],
        textposition="outside",
        increasing={"marker": {"color": PRIMARY}},
        decreasing={"marker": {"color": SECONDARY}},
        connector={"line": {"color": "lightgray"}},
    ))
    fig.update_layout(
        title=f"From base rate ({base_value:.3f}) to this wine's prediction",
        plot_bgcolor="white", paper_bgcolor="white", height=500,
        yaxis_title="Model output (High-quality probability)",
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Bars show how much each feature pushed this specific wine's prediction up or down from the model's average output.")

# ============================================================
# TAB 4 — WHAT-IF SIMULATOR
# ============================================================
with tab4:
    st.header("What-If Simulator")
    st.caption("Adjust the chemistry and see the prediction update live.")

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

    model_choice = st.selectbox("Model to use", ["MLP", "SVM", "NaiveBayes"], key="whatif_model")

    if st.button("Predict", type="primary"):
        type_white = 1 if wine_type == "white" else 0
        input_row = pd.DataFrame([[fixed_acidity, volatile_acidity, citric_acid, residual_sugar,
            chlorides, free_so2, total_so2, density, ph, sulphates, alcohol, type_white]],
            columns=feature_names)
        input_scaled = scaler.transform(input_row)
        model_map = {"MLP": mlp, "SVM": svm, "NaiveBayes": nb}
        chosen_model = model_map[model_choice]
        pred = chosen_model.predict(input_scaled)[0]
        pred_proba = chosen_model.predict_proba(input_scaled)[0]
        pred_label = le.inverse_transform([pred])[0]

        st.success(f"Predicted Quality Tier: **{pred_label}**")

        proba_df = pd.DataFrame({"Tier": class_names, "Probability": pred_proba * 100})
        fig = px.bar(proba_df, x="Tier", y="Probability", color="Tier",
                     color_discrete_map=TIER_COLORS, text="Probability",
                     category_orders={"Tier": ["Low", "Medium", "High"]})
        fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig.update_layout(showlegend=False, plot_bgcolor="white", paper_bgcolor="white",
                           yaxis_title="Probability (%)", yaxis_range=[0, 100])
        st.plotly_chart(fig, use_container_width=True)
