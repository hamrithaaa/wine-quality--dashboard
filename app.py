import json
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="WineScope | Quality Intelligence",
    page_icon="🍷",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Visual system ────────────────────────────────────────────────────────────
BG = "#080D1B"
PANEL = "#111A2E"
PANEL_2 = "#17233B"
TEXT = "#EDF3FF"
MUTED = "#91A2C6"
CYAN = "#38D9F5"
VIOLET = "#A78BFA"
PINK = "#FF5DB1"
LIME = "#B7F566"
AMBER = "#FFC857"
RED = "#FF6B7A"

TIER_COLORS = {"Low": RED, "Medium": AMBER, "High": LIME}
MODEL_COLORS = {"SVM": CYAN, "MLP": VIOLET, "NaiveBayes": PINK}
CHART_TEMPLATE = "plotly_dark"

st.markdown(
    f"""
    <style>
    :root {{
        color-scheme: dark;
        --ink: {TEXT};
        --muted: {MUTED};
        --panel: {PANEL};
        --cyan: {CYAN};
    }}
    .stApp {{
        background:
            radial-gradient(ellipse at 12% 0%, rgba(56,217,245,.11), transparent 34%),
            radial-gradient(ellipse at 90% 8%, rgba(167,139,250,.13), transparent 30%),
            {BG};
        color: {TEXT};
    }}
    [data-testid="stHeader"] {{ background: rgba(8,13,27,.82); }}
    [data-testid="stSidebar"] {{
        background: linear-gradient(180deg, #10182B 0%, #0A1020 100%);
        border-right: 1px solid rgba(145,162,198,.16);
    }}
    [data-testid="stSidebar"] > div {{ padding-top: 1.3rem; }}
    h1, h2, h3, h4, p, label, .stMarkdown {{ color: {TEXT}; }}
    h1 {{ letter-spacing: -0.04em; }}
    h2, h3 {{ letter-spacing: -0.025em; }}
    [data-testid="stMetric"] {{
        background: linear-gradient(145deg, rgba(23,35,59,.96), rgba(17,26,46,.96));
        border: 1px solid rgba(145,162,198,.19);
        border-radius: 18px;
        padding: 18px 20px;
        box-shadow: 0 10px 32px rgba(0,0,0,.12);
    }}
    [data-testid="stMetricLabel"] {{ color: {MUTED}; }}
    [data-testid="stMetricValue"] {{ color: {TEXT}; }}
    div[data-testid="stPlotlyChart"] {{
        background: rgba(17,26,46,.72);
        border: 1px solid rgba(145,162,198,.14);
        border-radius: 18px;
        padding: 8px;
    }}
    div[data-testid="stDataFrame"], div[data-testid="stTable"] {{
        border: 1px solid rgba(145,162,198,.17);
        border-radius: 14px;
        overflow: hidden;
    }}
    .hero {{
        padding: 1.55rem 1.7rem;
        border-radius: 24px;
        background: linear-gradient(120deg, rgba(56,217,245,.12), rgba(167,139,250,.13) 54%, rgba(255,93,177,.08));
        border: 1px solid rgba(145,162,198,.22);
        margin-bottom: 1.15rem;
    }}
    .eyebrow {{
        color: {CYAN};
        text-transform: uppercase;
        letter-spacing: .18em;
        font-size: .72rem;
        font-weight: 800;
    }}
    .hero h1 {{ margin: .3rem 0 .4rem 0; }}
    .hero p {{ color: {MUTED}; margin: 0; font-size: 1rem; }}
    .pill {{
        display: inline-block;
        padding: .25rem .65rem;
        margin: .2rem .25rem .2rem 0;
        border-radius: 999px;
        border: 1px solid rgba(56,217,245,.35);
        color: {CYAN};
        background: rgba(56,217,245,.08);
        font-size: .78rem;
    }}
    .section-note {{ color: {MUTED}; font-size: .9rem; }}
    .stButton > button[kind="primary"] {{
        background: linear-gradient(100deg, {CYAN}, {VIOLET});
        color: #07101E;
        border: 0;
        border-radius: 12px;
        font-weight: 800;
        min-height: 2.8rem;
    }}
    .stButton > button {{
        border-radius: 12px;
    }}
    div[data-testid="stRadio"] label {{ color: {TEXT}; }}
    hr {{ border-color: rgba(145,162,198,.17); }}
    </style>
    """,
    unsafe_allow_html=True,
)


# ── Assets ───────────────────────────────────────────────────────────────────
@st.cache_resource
def load_assets():
    svm = joblib.load("svm_model.pkl")
    mlp = joblib.load("mlp_model.pkl")
    nb = joblib.load("nb_model.pkl")
    scaler = joblib.load("scaler.pkl")
    label_encoder = joblib.load("label_encoder.pkl")
    df = pd.read_csv("wine_cleaned.csv")

    shap_mlp = np.load("shap_values_mlp.npy")
    shap_svm = np.load("shap_values_svm.npy")
    shap_nb = np.load("shap_values_nb.npy")
    X_sample_scaled = np.load("X_test_scaled_sample.npy")

    exp_mlp = np.load("expected_value_mlp.npy")
    exp_svm = np.load("expected_value_svm.npy")
    exp_nb = np.load("expected_value_nb.npy")

    with open("feature_names.json", encoding="utf-8") as f:
        feature_names = json.load(f)
    with open("class_names.json", encoding="utf-8") as f:
        class_names = json.load(f)
    with open("model_metrics.json", encoding="utf-8") as f:
        metrics = json.load(f)

    conf_matrices, roc_data = None, None
    try:
        with open("confusion_matrices.json", encoding="utf-8") as f:
            conf_matrices = json.load(f)
    except FileNotFoundError:
        pass
    try:
        with open("roc_data.json", encoding="utf-8") as f:
            roc_data = json.load(f)
    except FileNotFoundError:
        pass

    X_sample_real = scaler.inverse_transform(X_sample_scaled)
    return (
        svm, mlp, nb, scaler, label_encoder, df, feature_names, class_names,
        metrics, shap_mlp, shap_svm, shap_nb, X_sample_scaled, X_sample_real,
        exp_mlp, exp_svm, exp_nb, conf_matrices, roc_data,
    )


try:
    (
        svm, mlp, nb, scaler, le, df, feature_names, class_names, metrics,
        shap_mlp, shap_svm, shap_nb, X_sample_scaled, X_sample_real,
        exp_mlp, exp_svm, exp_nb, conf_matrices, roc_data,
    ) = load_assets()
except FileNotFoundError as exc:
    st.error(
        f"Required project asset not found: `{exc.filename}`. "
        "Keep app.py and the model/data files together in the repository."
    )
    st.stop()
except Exception as exc:
    st.error(f"The dashboard could not load its model assets: {exc}")
    st.stop()

MODEL_MAP = {"MLP": mlp, "SVM": svm, "NaiveBayes": nb}
SHAP_MAP = {"MLP": shap_mlp, "SVM": shap_svm, "NaiveBayes": shap_nb}
EXP_MAP = {"MLP": exp_mlp, "SVM": exp_svm, "NaiveBayes": exp_nb}
high_idx = class_names.index("High") if "High" in class_names else len(class_names) - 1


def normalize_name(name):
    return str(name).strip().lower().replace("_", " ").replace("-", " ")


def feature_index(*names):
    normalized = {normalize_name(n) for n in names}
    for i, name in enumerate(feature_names):
        if normalize_name(name) in normalized:
            return i
    return None


def prepare_row(values):
    """Build a model input row in the exact feature order used during training."""
    return pd.DataFrame([values], columns=feature_names)


def shap_2d(values):
    """Normalize saved SHAP arrays to sample × feature shape."""
    arr = np.asarray(values)
    if arr.ndim == 3:
        # Saved multi-class explanations are expected as samples × features × classes.
        arr = arr[:, :, high_idx] if arr.shape[-1] > high_idx else arr[:, :, -1]
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D SHAP values after normalization; got shape {arr.shape}.")
    return arr


def expected_for_class(values):
    arr = np.asarray(values)
    if arr.ndim == 0:
        return float(arr)
    flat = arr.reshape(-1)
    return float(flat[high_idx]) if flat.size > high_idx else float(flat[-1])


def apply_dark_layout(fig, height=None, **kwargs):
    fig.update_layout(
        template=CHART_TEMPLATE,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=TEXT, family="Inter, sans-serif"),
        margin=dict(l=30, r=25, t=55, b=35),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        **kwargs,
    )
    if height:
        fig.update_layout(height=height)
    fig.update_xaxes(gridcolor="rgba(145,162,198,.13)", zerolinecolor="rgba(145,162,198,.22)")
    fig.update_yaxes(gridcolor="rgba(145,162,198,.13)", zerolinecolor="rgba(145,162,198,.22)")
    return fig


def hero(eyebrow, title, subtitle):
    st.markdown(
        f"""
        <div class="hero">
          <div class="eyebrow">{eyebrow}</div>
          <h1>{title}</h1>
          <p>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def get_prediction(values, model_name):
    row = prepare_row(values)
    scaled = scaler.transform(row)
    model = MODEL_MAP[model_name]
    prediction = model.predict(scaled)[0]
    probabilities = model.predict_proba(scaled)[0]
    label = le.inverse_transform([prediction])[0]
    return label, probabilities, scaled


def feature_value_from_df(row, feature_name, fallback=0.0):
    if feature_name in row.index:
        try:
            return float(row[feature_name])
        except (TypeError, ValueError):
            return fallback
    return fallback


# ── Sidebar navigation ───────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        f"""
        <div style="padding:.2rem .15rem 1rem .15rem">
          <div style="font-size:2rem">🍷</div>
          <div style="font-size:1.18rem;font-weight:850;letter-spacing:-.03em">WineScope</div>
          <div style="color:{MUTED};font-size:.78rem">QUALITY INTELLIGENCE LAB</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("---")
    page = st.radio(
        "NAVIGATION",
        [
            "🌌 Overview",
            "⚔️ Model Arena",
            "🧠 SHAP Explainability",
            "🧪 What-If Lab",
            "🧬 Wine Fingerprint",
        ],
        label_visibility="visible",
        key="page_navigation",
    )
    st.markdown("---")
    st.markdown(
        f'<div style="color:{MUTED};font-size:.78rem">Powered by three classifiers<br/>Interactive analysis · Model explanations</div>',
        unsafe_allow_html=True,
    )


# ── Page 1: Overview ─────────────────────────────────────────────────────────
if page == "🌌 Overview":
    hero(
        "WINE QUALITY / ANALYTICS CONSOLE",
        "Read the chemistry. Reveal the quality.",
        "Explore the dataset, compare chemical signals, and understand how machine-learning models classify wine.",
    )
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Wine samples", f"{len(df):,}")
    col2.metric("Model features", len(feature_names))
    col3.metric("Quality tiers", df["quality_tier"].nunique() if "quality_tier" in df.columns else len(class_names))
    col4.metric("Classifiers", len(MODEL_MAP))

    st.markdown("#### Dataset snapshot")
    left, right = st.columns([1.45, 1])
    with left:
        st.dataframe(df.head(8), use_container_width=True, hide_index=True)
    with right:
        if "quality_tier" in df.columns:
            counts = df["quality_tier"].value_counts().reindex(["Low", "Medium", "High"]).dropna()
            fig = go.Figure(
                go.Pie(
                    labels=counts.index,
                    values=counts.values,
                    hole=.68,
                    sort=False,
                    marker=dict(colors=[TIER_COLORS.get(x, VIOLET) for x in counts.index],
                                line=dict(color=BG, width=4)),
                    textinfo="label+percent",
                    hovertemplate="%{label}: %{value} wines<extra></extra>",
                )
            )
            fig.add_annotation(text=f"<b>{len(df):,}</b><br><sup>samples</sup>", showarrow=False,
                               font=dict(size=22, color=TEXT))
            apply_dark_layout(fig, height=350, showlegend=False, title="Quality tier mix")
            st.plotly_chart(fig, use_container_width=True, theme=None)

    st.markdown("#### Chemical relationships")
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.shape[1] >= 2:
        corr = numeric_df.corr(numeric_only=True)
        fig = px.imshow(
            corr, text_auto=".2f", aspect="auto",
            color_continuous_scale=[[0, "#FF6B7A"], [.5, "#111A2E"], [1, "#38D9F5"]],
            zmin=-1, zmax=1,
        )
        apply_dark_layout(fig, height=max(470, 28 * len(corr.columns)), title="Correlation constellation")
        st.plotly_chart(fig, use_container_width=True, theme=None)

    if "quality_tier" in df.columns and feature_names:
        st.markdown("#### Explore a chemical signal")
        available_features = [f for f in feature_names if f in df.columns and pd.api.types.is_numeric_dtype(df[f])]
        if available_features:
            selected_feature = st.selectbox("Choose a feature", available_features, key="overview_feature")
            fig = px.violin(
                df, x="quality_tier", y=selected_feature, color="quality_tier",
                category_orders={"quality_tier": ["Low", "Medium", "High"]},
                color_discrete_map=TIER_COLORS, box=True, points=False,
            )
            apply_dark_layout(fig, height=420, title=f"{selected_feature}: distribution by tier", showlegend=False)
            st.plotly_chart(fig, use_container_width=True, theme=None)


# ── Page 2: Model comparison ─────────────────────────────────────────────────
elif page == "⚔️ Model Arena":
    hero(
        "BENCHMARK / CLASSIFICATION",
        "The Model Arena",
        "Compare the classifiers using the metrics exported with your trained models.",
    )
    metric_cols = st.columns(max(1, len(metrics)))
    for i, (name, result) in enumerate(metrics.items()):
        metric_cols[i].metric(name, f"{result.get('accuracy', 0) * 100:.1f}% accuracy")

    acc_df = pd.DataFrame(
        [{"Model": name, "Accuracy": result.get("accuracy", 0) * 100} for name, result in metrics.items()]
    ).sort_values("Accuracy", ascending=True)
    fig = px.bar(
        acc_df, x="Accuracy", y="Model", orientation="h", color="Model",
        color_discrete_map=MODEL_COLORS, text="Accuracy",
    )
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside", cliponaxis=False)
    fig.update_xaxes(range=[0, max(100, float(acc_df["Accuracy"].max()) * 1.12)], title="Accuracy (%)")
    apply_dark_layout(fig, height=330, title="Accuracy leaderboard", showlegend=False)
    st.plotly_chart(fig, use_container_width=True, theme=None)

    if roc_data:
        st.markdown("#### ROC orbit")
        fig = go.Figure()
        for name, rd in roc_data.items():
            fig.add_trace(go.Scatter(
                x=rd["fpr"], y=rd["tpr"], mode="lines",
                name=f"{name} · AUC {rd['auc']:.3f}",
                line=dict(color=MODEL_COLORS.get(name, CYAN), width=3),
                hovertemplate="False positive rate: %{x:.3f}<br>True positive rate: %{y:.3f}<extra></extra>",
            ))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                                 line=dict(color=MUTED, dash="dash"), name="Random baseline"))
        apply_dark_layout(fig, height=430, title="Receiver operating characteristic")
        fig.update_xaxes(title="False positive rate", range=[0, 1])
        fig.update_yaxes(title="True positive rate", range=[0, 1])
        st.plotly_chart(fig, use_container_width=True, theme=None)
    else:
        st.info("Optional ROC data was not found. Add `roc_data.json` to enable this chart.")

    st.markdown("#### Confusion matrix")
    if conf_matrices:
        selected_cm = st.selectbox("Choose classifier", list(conf_matrices.keys()), key="cm_model")
        cm = np.asarray(conf_matrices[selected_cm])
        fig = px.imshow(
            cm, text_auto=True, x=class_names, y=class_names, aspect="auto",
            color_continuous_scale=[[0, PANEL], [.45, "#5446A6"], [1, CYAN]],
        )
        apply_dark_layout(fig, height=420, title=f"{selected_cm} · actual vs predicted")
        fig.update_xaxes(title="Predicted class")
        fig.update_yaxes(title="Actual class")
        st.plotly_chart(fig, use_container_width=True, theme=None)
    else:
        st.info("Optional confusion-matrix data was not found. Add `confusion_matrices.json` to enable this chart.")


# ── Page 3: SHAP explainability ──────────────────────────────────────────────
elif page == "🧠 SHAP Explainability":
    hero(
        "MODEL TRANSPARENCY / SHAP",
        "Inside the prediction",
        "Global feature influence and one-wine explanations based on the SHAP arrays exported for your models.",
    )
    model_for_shap = st.selectbox("Select model", list(SHAP_MAP.keys()), key="shap_model")
    try:
        sv = shap_2d(SHAP_MAP[model_for_shap])
    except ValueError as exc:
        st.error(str(exc))
        st.stop()
    ev = EXP_MAP[model_for_shap]
    n_samples = min(sv.shape[0], X_sample_real.shape[0])
    n_features = min(sv.shape[1], len(feature_names), X_sample_real.shape[1])
    sv = sv[:n_samples, :n_features]
    feature_labels = feature_names[:n_features]
    sample_values = X_sample_real[:n_samples, :n_features]

    mean_abs = np.abs(sv).mean(axis=0)
    order_idx = np.argsort(mean_abs)
    importance_df = pd.DataFrame({
        "Feature": [feature_labels[i] for i in order_idx],
        "Mean |SHAP|": mean_abs[order_idx],
    })
    fig = px.bar(
        importance_df, x="Mean |SHAP|", y="Feature", orientation="h",
        color="Mean |SHAP|", color_continuous_scale=[[0, "#293653"], [.55, VIOLET], [1, CYAN]],
    )
    apply_dark_layout(fig, height=max(430, n_features * 28), title=f"{model_for_shap} · global importance")
    fig.update_layout(coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True, theme=None)

    st.markdown("#### SHAP impact cloud")
    rows = []
    rng = np.random.default_rng(42)
    for rank, fi in enumerate(order_idx):
        jitter = rng.uniform(-.32, .32, size=n_samples)
        for sample_idx in range(n_samples):
            rows.append({
                "Feature": feature_labels[fi],
                "y": rank + jitter[sample_idx],
                "SHAP value": float(sv[sample_idx, fi]),
                "Feature value": float(sample_values[sample_idx, fi]),
            })
    bee_df = pd.DataFrame(rows)
    fig = px.scatter(
        bee_df, x="SHAP value", y="y", color="Feature value",
        color_continuous_scale=[[0, "#38D9F5"], [.5, "#A78BFA"], [1, "#FF5DB1"]],
        hover_data={"Feature": True, "Feature value": ":.3f", "y": False},
    )
    fig.update_yaxes(tickmode="array", tickvals=list(range(len(order_idx))),
                     ticktext=[feature_labels[i] for i in order_idx], title="")
    fig.add_vline(x=0, line_dash="dash", line_color=MUTED)
    apply_dark_layout(fig, height=max(450, n_features * 30), title="Each point is one wine sample")
    st.plotly_chart(fig, use_container_width=True, theme=None)
    st.caption("SHAP values to the right of zero push the model output upward; values to the left push it downward. Colour encodes the feature's actual value.")

    st.markdown("#### Individual wine · contribution waterfall")
    sample_idx = st.slider("Test sample", 0, max(0, n_samples - 1), 0, key="wf_sample")
    base_value = expected_for_class(ev)
    contributions = sv[sample_idx]
    top_n = min(8, len(contributions))
    selected = np.argsort(-np.abs(contributions))[:top_n]
    labels = [feature_labels[i] for i in selected]
    values = [float(contributions[i]) for i in selected]
    other = float(np.sum([contributions[i] for i in range(len(contributions)) if i not in selected]))
    if abs(other) > 1e-6:
        labels.append("Remaining features")
        values.append(other)
    colors = [CYAN if value >= 0 else PINK for value in values]
    running = [base_value]
    for value in values:
        running.append(running[-1] + value)
    fig = go.Figure()
    for i, (label, value) in enumerate(zip(labels, values)):
        start = running[i]
        end = running[i + 1]
        fig.add_trace(go.Bar(
            x=[label], y=[abs(value)], base=[min(start, end)],
            marker_color=CYAN if value >= 0 else PINK,
            name="Pushes output up" if value >= 0 else "Pushes output down",
            showlegend=(i == 0 or i == next((j for j, v in enumerate(values) if v < 0), -1)),
            hovertemplate=f"{label}<br>Contribution: {value:+.4f}<extra></extra>",
        ))
    fig.add_hline(y=base_value, line_dash="dot", line_color=AMBER,
                  annotation_text=f"Base {base_value:.3f}", annotation_font_color=AMBER)
    apply_dark_layout(fig, height=430, title="Contribution steps (relative to the exported SHAP base value)")
    fig.update_layout(barmode="overlay", yaxis_title="Model output scale")
    st.plotly_chart(fig, use_container_width=True, theme=None)
    st.caption("This visual shows relative SHAP contributions starting from the exported expected value. The output scale depends on how the SHAP values were generated.")


# ── Page 4: What-if lab ──────────────────────────────────────────────────────
elif page == "🧪 What-If Lab":
    hero(
        "INTERACTIVE EXPERIMENT / PREDICTION SURFACE",
        "Change the chemistry. Explore the outcome.",
        "Adjust a wine profile, then inspect a model-generated response surface showing how alcohol and volatile acidity interact.",
    )
    st.markdown(
        f'<span class="pill">LIVE MODEL INFERENCE</span><span class="pill">2D RESPONSE SURFACE</span><span class="pill">PROBABILITY CONTOURS</span>',
        unsafe_allow_html=True,
    )

    defaults = {
        "fixed acidity": (7.0, 3.8, 15.9, .1),
        "volatile acidity": (.4, .08, 1.6, .01),
        "citric acid": (.3, 0.0, 1.7, .01),
        "residual sugar": (5.0, .6, 66.0, .1),
        "chlorides": (.05, .009, .61, .001),
        "free sulfur dioxide": (30.0, 1.0, 290.0, 1.0),
        "total sulfur dioxide": (115.0, 6.0, 440.0, 1.0),
        "density": (.996, .987, 1.004, .0001),
        "ph": (3.2, 2.7, 4.0, .01),
        "sulphates": (.5, .2, 2.0, .01),
        "alcohol": (10.0, 8.0, 14.9, .1),
    }
    current_values = {}
    input_col1, input_col2 = st.columns(2)
    for i, feature in enumerate(feature_names):
        normalized = normalize_name(feature)
        container = input_col1 if i < (len(feature_names) + 1) // 2 else input_col2
        if normalized in {"type white", "white wine", "type"}:
            continue
        spec = defaults.get(normalized)
        if spec is None:
            # Use observed dataset range when a feature name differs from the usual wine schema.
            if feature in df.columns and pd.api.types.is_numeric_dtype(df[feature]):
                low = float(df[feature].quantile(.01))
                high = float(df[feature].quantile(.99))
                default = float(df[feature].median())
                if not np.isfinite(low + high + default) or low >= high:
                    low, high, default = 0.0, 1.0, .5
                step = max((high - low) / 100, 0.001)
            else:
                low, high, default, step = 0.0, 1.0, 0.5, .01
        else:
            default, low, high, step = spec
        # Avoid sliders whose bounds don't cover the saved dataset's actual domain.
        with container:
            current_values[feature] = st.slider(
                feature.replace("_", " ").title(),
                min_value=float(low), max_value=float(high),
                value=float(np.clip(default, low, high)), step=float(step),
                key=f"whatif_{feature}",
            )

    # The training feature order is authoritative. The original project encoded white wine as 1.
    values = []
    for feature in feature_names:
        normalized = normalize_name(feature)
        if normalized in {"type white", "white wine", "type"}:
            values.append(1.0 if st.session_state.get("whatif_wine_type", "White") == "White" else 0.0)
        elif feature in current_values:
            values.append(current_values[feature])
        else:
            values.append(float(df[feature].median()) if feature in df.columns and pd.api.types.is_numeric_dtype(df[feature]) else 0.0)

    type_idx = feature_index("type_white", "white wine", "type")
    if type_idx is not None:
        wine_type = st.selectbox("Wine type", ["White", "Red"], key="whatif_wine_type")
        values[type_idx] = 1.0 if wine_type == "White" else 0.0
    else:
        wine_type = "White"

    model_choice = st.selectbox("Inference model", list(MODEL_MAP.keys()), key="whatif_model")
    try:
        predicted_label, probabilities, _ = get_prediction(values, model_choice)
    except Exception as exc:
        st.error(f"Could not generate a prediction for this feature vector: {exc}")
        st.stop()

    pred_color = TIER_COLORS.get(str(predicted_label), CYAN)
    p1, p2, p3 = st.columns([1.1, 1, 1])
    p1.metric("Predicted tier", str(predicted_label))
    p2.metric("Top-class confidence", f"{float(np.max(probabilities)) * 100:.1f}%")
    p3.metric("Selected model", model_choice)

    probability_df = pd.DataFrame({"Tier": class_names, "Probability": probabilities * 100})
    fig = go.Figure(go.Bar(
        x=probability_df["Tier"], y=probability_df["Probability"],
        marker_color=[TIER_COLORS.get(str(tier), VIOLET) for tier in probability_df["Tier"]],
        text=[f"{v:.1f}%" for v in probability_df["Probability"]],
        textposition="outside",
        hovertemplate="%{x}: %{y:.2f}%<extra></extra>",
    ))
    apply_dark_layout(fig, height=300, title="Current profile · class probabilities", showlegend=False)
    fig.update_yaxes(title="Probability (%)", range=[0, 112])
    st.plotly_chart(fig, use_container_width=True, theme=None)

    alcohol_idx = feature_index("alcohol")
    volatile_idx = feature_index("volatile acidity")
    if alcohol_idx is None or volatile_idx is None:
        st.warning("The response surface needs feature names for both `alcohol` and `volatile acidity`.")
    else:
        st.markdown("#### Response surface · explore two variables at once")
        st.caption("Every grid cell is a fresh prediction from your selected trained model. All other features stay fixed at the slider values above.")
        grid_n = st.slider("Surface resolution", min_value=15, max_value=45, value=25, step=5, key="surface_resolution")
        alcohol_domain = defaults.get("alcohol", (10.0, 8.0, 14.9, .1))
        volatile_domain = defaults.get("volatile acidity", (.4, .08, 1.6, .01))
        alcohol_grid = np.linspace(alcohol_domain[1], alcohol_domain[2], grid_n)
        volatile_grid = np.linspace(volatile_domain[1], volatile_domain[2], grid_n)
        grid_rows = []
        for va in volatile_grid:
            for alc in alcohol_grid:
                sample = list(values)
                sample[alcohol_idx] = float(alc)
                sample[volatile_idx] = float(va)
                grid_rows.append(sample)
        grid_df = pd.DataFrame(grid_rows, columns=feature_names)
        scaled_grid = scaler.transform(grid_df)
        chosen_model = MODEL_MAP[model_choice]
        grid_probabilities = chosen_model.predict_proba(scaled_grid)
        encoded_high = int(le.transform(["High"])[0]) if "High" in le.classes_ else None
        model_classes = list(getattr(chosen_model, "classes_", range(grid_probabilities.shape[1])))
        high_class_pos = model_classes.index(encoded_high) if encoded_high in model_classes else int(np.argmax(np.mean(grid_probabilities, axis=0)))
        z = grid_probabilities[:, high_class_pos].reshape(grid_n, grid_n) * 100
        fig = go.Figure(go.Contour(
            x=alcohol_grid,
            y=volatile_grid,
            z=z,
            colorscale=[[0, "#19233E"], [.25, "#5446A6"], [.55, "#A78BFA"], [.78, "#FF5DB1"], [1, "#38D9F5"]],
            contours=dict(showlabels=True, labelfont=dict(color=TEXT, size=10), coloring="heatmap"),
            colorbar=dict(title="High tier<br>prob. (%)", ticksuffix="%"),
            hovertemplate="Alcohol: %{x:.2f}<br>Volatile acidity: %{y:.3f}<br>High-tier probability: %{z:.1f}%<extra></extra>",
        ))
        current_alcohol = values[alcohol_idx]
        current_va = values[volatile_idx]
        fig.add_trace(go.Scatter(
            x=[current_alcohol], y=[current_va], mode="markers+text",
            text=["Current profile"], textposition="top center",
            marker=dict(size=14, color=LIME, line=dict(color=BG, width=2), symbol="star"),
            name="Current profile",
            hovertemplate="Current profile<br>Alcohol: %{x:.2f}<br>Volatile acidity: %{y:.3f}<extra></extra>",
        ))
        apply_dark_layout(fig, height=570, title="Predicted probability landscape · High quality")
        fig.update_xaxes(title="Alcohol")
        fig.update_yaxes(title="Volatile acidity")
        st.plotly_chart(fig, use_container_width=True, theme=None)
        st.caption("Bright regions indicate higher model-estimated probability for the High tier. This is a model response surface, not a claim of causal chemistry.")

    with st.expander("Inspect the exact feature vector sent to the model"):
        st.dataframe(pd.DataFrame([values], columns=feature_names), use_container_width=True)


# ── Page 5: Chemical fingerprint ─────────────────────────────────────────────
elif page == "🧬 Wine Fingerprint":
    hero(
        "CHEMICAL SIGNATURE / COMPARATIVE PROFILE",
        "Every wine has a fingerprint.",
        "Compare one dataset sample against the average High-tier profile across the chemical features available in your dataset.",
    )
    if "quality_tier" not in df.columns:
        st.warning("The dataset needs a `quality_tier` column to build a High-tier reference profile.")
        st.stop()

    numeric_candidates = [
        name for name in feature_names
        if name in df.columns and pd.api.types.is_numeric_dtype(df[name])
    ]
    if len(numeric_candidates) < 3:
        numeric_candidates = [
            name for name in df.select_dtypes(include=[np.number]).columns
            if name != "quality" and name != "quality_tier"
        ]
    if len(numeric_candidates) < 3:
        st.warning("At least three numeric features are needed to create a fingerprint.")
        st.stop()

    sample_choice = st.selectbox(
        "Choose a wine sample",
        options=list(range(len(df))),
        format_func=lambda i: f"Sample #{i + 1} · {df.iloc[i].get('quality_tier', 'tier unknown')}",
        key="fingerprint_sample",
    )
    chosen_row = df.iloc[sample_choice]
    high_df = df[df["quality_tier"].astype(str).str.lower() == "high"]
    if high_df.empty:
        st.warning("No rows labeled `High` were found in `quality_tier`.")
        st.stop()

    # Normalize each selected feature to the dataset's observed range so unlike units can share a radar.
    profile_features = numeric_candidates[:12]
    minima = df[profile_features].min()
    maxima = df[profile_features].max()
    spans = (maxima - minima).replace(0, 1)
    sample_norm = ((chosen_row[profile_features].astype(float) - minima) / spans).clip(0, 1)
    high_norm = ((high_df[profile_features].mean() - minima) / spans).clip(0, 1)

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=sample_norm.tolist() + [sample_norm.iloc[0]],
        theta=profile_features + [profile_features[0]],
        fill="toself",
        name=f"Sample #{sample_choice + 1}",
        line=dict(color=CYAN, width=3),
        fillcolor="rgba(56,217,245,.16)",
        hovertemplate="%{theta}<br>Normalized level: %{r:.2f}<extra>Selected wine</extra>",
    ))
    fig.add_trace(go.Scatterpolar(
        r=high_norm.tolist() + [high_norm.iloc[0]],
        theta=profile_features + [profile_features[0]],
        fill="toself",
        name="High-tier average",
        line=dict(color=PINK, width=3, dash="dot"),
        fillcolor="rgba(255,93,177,.10)",
        hovertemplate="%{theta}<br>Normalized level: %{r:.2f}<extra>High-tier average</extra>",
    ))
    fig.update_layout(
        template=CHART_TEMPLATE,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=TEXT),
        height=620,
        title="Normalized chemical fingerprint",
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(visible=True, range=[0, 1], gridcolor="rgba(145,162,198,.2)", tickfont=dict(color=MUTED)),
            angularaxis=dict(gridcolor="rgba(145,162,198,.2)", tickfont=dict(color=TEXT)),
        ),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=45, r=45, t=70, b=35),
    )
    st.plotly_chart(fig, use_container_width=True, theme=None)

    high_count = len(high_df)
    st.markdown("#### Sample context")
    c1, c2, c3 = st.columns(3)
    c1.metric("Selected sample tier", str(chosen_row["quality_tier"]))
    c2.metric("High-tier reference size", f"{high_count:,} wines")
    c3.metric("Features on fingerprint", len(profile_features))
    st.caption("Radar values are min–max normalized against the full dataset so features with different units can be compared. This chart compares profiles; it does not itself predict quality.")
