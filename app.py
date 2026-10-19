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
        fillcolor="rgba(0,229,255,.20)",
        hovertemplate="%{theta}<br>Normalized level: %{r:.2f}<extra>Selected wine</extra>",
    ))
    fig.add_trace(go.Scatterpolar(
        r=high_norm.tolist() + [high_norm.iloc[0]],
        theta=profile_features + [profile_features[0]],
        fill="toself",
        name="High-tier average",
        line=dict(color=ORANGE, width=3, dash="dot"),
        fillcolor="rgba(255,138,0,.18)",
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
            radialaxis=dict(visible=True, range=[0, 1], gridcolor="rgba(0,229,255,.28)", tickfont=dict(color=MUTED)),
            angularaxis=dict(gridcolor="rgba(0,229,255,.28)", tickfont=dict(color=TEXT)),
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
