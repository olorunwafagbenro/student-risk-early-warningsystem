# ============================================================
# dashboard.py
# Student Learning Risk Early Warning System
# Teacher-facing Streamlit dashboard (deployment-safe version)
#
# Design rules for Streamlit Community Cloud:
#   1. st.set_page_config is the first Streamlit call
#   2. Paths are built from this file's location (BASE_DIR)
#   3. Only config + SHAP output are loaded (no model/scaler)
#   4. If artifacts are missing or fail to load, the app falls
#      back to demo mode instead of crashing
#   5. Light on CPU/memory: capped tables, no heavy styling
#   6. Runtime needs only: streamlit, pandas, numpy,
#      matplotlib, plotly (no shap/lightgbm/xgboost)
# ============================================================

import os
import json
import gzip
import pickle
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# 1. PAGE CONFIG (must be the first Streamlit command)
# ------------------------------------------------------------
st.set_page_config(
    page_title="Student Risk Dashboard",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)


# 2. CONSTANTS
# ------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

MAX_TABLE_ROWS = 200      # keeps the app light on the free tier
MAX_SELECT_OPTIONS = 300  # same reason for the student selector

RISK_ORDER = ["High", "Medium", "Low"]
RISK_COLORS = {"High": "#e74c3c", "Medium": "#e67e22", "Low": "#2ecc71"}

EXPLANATIONS = {
    "engagement_dropoff": " Platform engagement has dropped significantly",
    "total_clicks": " Very low overall platform activity",
    "avg_early_score": " Assessment scores were low from the start",
    "score_trend": " Assessment scores have been declining",
    "submission_rate": " Assignment completion rate is below average",
    "late_submission_count": " Multiple assignments submitted late",
    "days_since_last_active": " Has not engaged with the platform recently",
    "num_of_prev_attempts": " Has attempted this course before",
    "early_clicks": " Low engagement in the first weeks of term",
    "late_registration": " Registered after the course started",
    "active_days": " Few days of active engagement",
    "studied_credits": " Heavy course load",
}

PROTECTIVE = {
    "engagement_dropoff": "Engagement has stayed steady",
    "total_clicks": "Strong overall platform activity",
    "avg_early_score": "Solid early assessment scores",
    "score_trend": "Scores are stable or improving",
    "submission_rate": "Submits most assignments",
    "late_submission_count": "Submits work on time",
    "days_since_last_active": "Active on the platform recently",
    "num_of_prev_attempts": "First attempt at the course",
    "early_clicks": "Strong engagement early in term",
    "late_registration": "Registered on time",
    "active_days": "Engages on many different days",
}



# 3. SMALL HELPERS
# ------------------------------------------------------------
def stretch(fn, *args, **kwargs):
    """Call a Streamlit element at full width on old and new versions."""
    try:
        return fn(*args, width="stretch", **kwargs)
    except TypeError:
        return fn(*args, use_container_width=True, **kwargs)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))



# 4. DATA LOADING (real artifacts, with demo fallback)
# ------------------------------------------------------------
DEFAULT_CONFIG = {
    "model_name": "LightGBM",
    "threshold": 0.5,
    "training_date": "N/A",
}


def _normalise_payload(payload):
    """Make sure the SHAP payload has predictable shapes and types."""
    X = payload["X_test"]
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X, columns=payload["feature_names"])
    X = X.reset_index(drop=True)

    shap_vals = payload["shap_values"]
    if isinstance(shap_vals, list):          # old shap: [class0, class1]
        shap_vals = shap_vals[1]
    shap_vals = np.asarray(shap_vals)
    if shap_vals.ndim == 3:                  # new shap: (n, features, classes)
        shap_vals = shap_vals[:, :, 1]

    probas = np.asarray(payload["risk_probabilities"], dtype=float).ravel()
    base = payload.get("base_value", 0.0)
    base = float(np.ravel(base)[-1]) if np.ndim(base) else float(base)

    return {
        "X": X,
        "shap_values": shap_vals,
        "probas": probas,
        "feature_names": list(X.columns),
        "base_value": base,
    }


def _load_real_artifacts():
    """Return (config, payload). Raises if files are missing/broken."""
    config = dict(DEFAULT_CONFIG)
    config_path = os.path.join(MODELS_DIR, "model_config.json")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config.update(json.load(f))

    gz_path = os.path.join(MODELS_DIR, "shap_output.pkl.gz")
    pkl_path = os.path.join(MODELS_DIR, "shap_output.pkl")

    if os.path.exists(gz_path):
        with gzip.open(gz_path, "rb") as f:
            raw = pickle.load(f)
    elif os.path.exists(pkl_path):
        with open(pkl_path, "rb") as f:
            raw = pickle.load(f)
    else:
        raise FileNotFoundError(
            "models/shap_output.pkl.gz (or .pkl) was not found in the repo."
        )

    return config, _normalise_payload(raw)


def _generate_demo(n=600, seed=42):
    """Synthetic but internally consistent data for demo mode."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({
        "avg_early_score": rng.normal(58, 18, n).clip(0, 100),
        "submission_rate": rng.beta(5, 2, n),
        "total_clicks": rng.exponential(1500, n),
        "active_days": rng.integers(2, 120, n).astype(float),
        "days_since_last_active": rng.exponential(10, n),
        "score_trend": rng.normal(-2, 9, n),
        "engagement_dropoff": rng.normal(-120, 350, n),
        "late_submission_count": rng.poisson(1.2, n).astype(float),
        "num_of_prev_attempts": rng.choice(
            [0, 1, 2, 3], n, p=[0.65, 0.2, 0.1, 0.05]
        ).astype(float),
    })

    weights = {  # positive weight = a higher value raises risk
        "avg_early_score": -0.6, "submission_rate": -0.8,
        "total_clicks": -0.7, "active_days": -0.4,
        "days_since_last_active": 0.6, "score_trend": -0.4,
        "engagement_dropoff": -0.9, "late_submission_count": 0.3,
        "num_of_prev_attempts": 0.5,
    }
    z = (X - X.mean()) / X.std()
    shap_vals = np.column_stack([weights[c] * z[c] * 0.5 for c in X.columns])
    base = -0.5
    probas = sigmoid(base + shap_vals.sum(axis=1))

    return {
        "X": X,
        "shap_values": shap_vals,
        "probas": probas,
        "feature_names": list(X.columns),
        "base_value": base,
    }


@st.cache_resource(show_spinner="Loading student risk data...")
def _load_live_cached():
    # Exceptions are NOT cached by Streamlit, so a failed load is
    # retried on the next rerun (e.g. after you add the file).
    return _load_real_artifacts()


@st.cache_resource(show_spinner=False)
def _demo_cached():
    return _generate_demo()


def load_data():
    """Returns (config, payload, mode, message)."""
    try:
        config, payload = _load_live_cached()
        return config, payload, "live", ""
    except Exception as e:  # missing file, bad pickle, version mismatch
        return (
            dict(DEFAULT_CONFIG),
            _demo_cached(),
            "demo",
            f"{type(e).__name__}: {e}",
        )


def list_repo_files():
    """Small diagnostic: what the deployed app can actually see."""
    lines = [f"App folder: {BASE_DIR}", "", "Files next to dashboard.py:"]
    try:
        lines += [f"  {n}" for n in sorted(os.listdir(BASE_DIR))]
    except Exception as e:
        lines.append(f"  (could not list: {e})")
    lines += ["", "Files in models/:"]
    if os.path.isdir(MODELS_DIR):
        for n in sorted(os.listdir(MODELS_DIR)):
            size = os.path.getsize(os.path.join(MODELS_DIR, n)) / 1024 / 1024
            lines.append(f"  {n}  ({size:.2f} MB)")
    else:
        lines.append("  (no models/ folder found)")
    return "\n".join(lines)


config, payload, mode, load_message = load_data()

X = payload["X"]
shap_values = payload["shap_values"]
feature_names = payload["feature_names"]
base_value = payload["base_value"]
threshold = float(config.get("threshold", 0.5))



# 5. BUILD THE STUDENT TABLE
# ------------------------------------------------------------
@st.cache_data(show_spinner=False)
def build_student_df(_X, probas, threshold):
    df = _X.copy().reset_index(drop=True)
    df["student_id"] = [f"S{1000 + i}" for i in range(len(df))]
    df["risk_proba"] = probas
    df["risk_pct"] = (probas * 100).round(1)
    df["risk_level"] = pd.cut(
        df["risk_proba"],
        bins=[0, 0.33, 0.66, 1.0],
        labels=["Low", "Medium", "High"],
        include_lowest=True,
    ).astype(str)
    df["flagged"] = (df["risk_proba"] >= threshold).astype(int)
    return df


student_df = build_student_df(X, payload["probas"], threshold)
total = len(student_df)



# 6. SIDEBAR
# ------------------------------------------------------------
with st.sidebar:
    st.title(" Risk Dashboard")
    st.markdown("---")
    st.subheader(" Filters")

    risk_filter = st.multiselect(
        "Risk level", RISK_ORDER, default=["High", "Medium"]
    )
    flagged_only = st.toggle("Flagged students only", value=False)
    score_range = st.slider("Risk score range (%)", 0, 100, (0, 100))
    search_id = st.text_input(" Search student ID", placeholder="e.g. S1042")

    st.markdown("---")
    st.subheader(" Model info")
    st.caption(f"Model: {config.get('model_name', 'N/A')}")
    st.caption(f"Flag threshold: {threshold:.0%}")
    st.caption(f"Trained: {config.get('training_date', 'N/A')}")
    st.caption(f"Features: {len(feature_names)}")
    st.caption("Data: " + ("live model output" if mode == "live" else "demo data"))

    st.markdown("---")
    st.subheader(" Export")
    export_cols = [
        c for c in ["student_id", "risk_pct", "risk_level",
                    "avg_early_score", "submission_rate", "total_clicks"]
        if c in student_df.columns
    ]
    export_df = student_df.loc[student_df["flagged"] == 1, export_cols]
    st.download_button(
        " Download flagged students",
        data=export_df.to_csv(index=False),
        file_name="flagged_students.csv",
        mime="text/csv",
    )


# ------------------------------------------------------------
# 7. HEADER
# ------------------------------------------------------------
st.title("🎓 Student Learning Risk Dashboard")
st.caption(
    "Early warning system. Predictions support teacher judgment; "
    "they do not replace it."
)

if mode == "demo":
    st.info(
        "Demo mode: showing simulated data because the real model output "
        "could not be loaded."
    )
    if load_message:
        with st.expander("Why demo mode? (technical detail)"):
            st.code(load_message)
            st.markdown("**What the deployed app can see:**")
            st.code(list_repo_files())



# 8. OVERVIEW CARDS
# ------------------------------------------------------------
high_n = int((student_df["risk_level"] == "High").sum())
med_n = int((student_df["risk_level"] == "Medium").sum())
low_n = int((student_df["risk_level"] == "Low").sum())
flagged_n = int(student_df["flagged"].sum())

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total students", f"{total:,}")
c2.metric(" High risk", f"{high_n:,}", f"{high_n / total:.1%} of cohort",
          delta_color="off")
c3.metric(" Medium risk", f"{med_n:,}", f"{med_n / total:.1%} of cohort",
          delta_color="off")
c4.metric(" Low risk", f"{low_n:,}", f"{low_n / total:.1%} of cohort",
          delta_color="off")
c5.metric(" Flagged for review", f"{flagged_n:,}",
          f"avg risk {student_df['risk_pct'].mean():.1f}%", delta_color="off")

st.markdown("---")



# 9. DISTRIBUTION CHARTS
# ------------------------------------------------------------
left, right = st.columns(2)

with left:
    st.subheader(" Risk level distribution")
    counts = (
        student_df["risk_level"].value_counts()
        .reindex(RISK_ORDER).fillna(0).reset_index()
    )
    counts.columns = ["Risk level", "Students"]
    fig_pie = px.pie(
        counts, names="Risk level", values="Students", hole=0.45,
        color="Risk level", color_discrete_map=RISK_COLORS,
    )
    fig_pie.update_layout(height=300, margin=dict(t=10, b=10, l=10, r=10),
                          showlegend=False)
    stretch(st.plotly_chart, fig_pie)

with right:
    st.subheader(" Risk score distribution")
    fig_hist = px.histogram(
        student_df, x="risk_pct", color="risk_level", nbins=40,
        color_discrete_map=RISK_COLORS,
        category_orders={"risk_level": RISK_ORDER},
        labels={"risk_pct": "Risk score (%)", "risk_level": "Risk level"},
    )
    fig_hist.update_layout(height=300, margin=dict(t=10, b=10, l=10, r=10))
    stretch(st.plotly_chart, fig_hist)

st.markdown("---")


# ------------------------------------------------------------
# 10. STUDENT TABLE (filtered, capped)
# ------------------------------------------------------------
st.subheader(" Student risk table")

filtered = student_df.copy()
if risk_filter:
    filtered = filtered[filtered["risk_level"].isin(risk_filter)]
if flagged_only:
    filtered = filtered[filtered["flagged"] == 1]
filtered = filtered[
    (filtered["risk_pct"] >= score_range[0])
    & (filtered["risk_pct"] <= score_range[1])
]
if search_id:
    filtered = filtered[
        filtered["student_id"].str.contains(search_id, case=False, na=False)
    ]
filtered = filtered.sort_values("risk_proba", ascending=False)

table_cols = {
    "student_id": "Student ID",
    "risk_pct": "Risk %",
    "risk_level": "Risk level",
    "avg_early_score": "Early score",
    "submission_rate": "Submission rate",
    "total_clicks": "Platform clicks",
    "days_since_last_active": "Days inactive",
}
present = [c for c in table_cols if c in filtered.columns]
view = filtered[present].head(MAX_TABLE_ROWS).rename(columns=table_cols)

for col in ["Early score", "Platform clicks", "Days inactive"]:
    if col in view.columns:
        view[col] = view[col].round(1)
if "Submission rate" in view.columns:
    view["Submission rate"] = (view["Submission rate"] * 100).round(0)

col_config = {
    "Risk %": st.column_config.ProgressColumn(
        "Risk %", min_value=0, max_value=100, format="%.1f"
    ),
}
if "Submission rate" in view.columns:
    col_config["Submission rate"] = st.column_config.NumberColumn(
        "Submission rate", format="%d%%"
    )

stretch(st.dataframe, view, hide_index=True, column_config=col_config,
        height=350)
st.caption(
    f"Showing the top {min(len(filtered), MAX_TABLE_ROWS):,} of "
    f"{len(filtered):,} matching students ({total:,} in cohort)."
)

st.markdown("---")



# 11. INDIVIDUAL STUDENT ANALYSIS
# ------------------------------------------------------------
st.subheader(" Individual student analysis")

options = filtered["student_id"].head(MAX_SELECT_OPTIONS).tolist()

if not options:
    st.warning("No students match the current filters.")
else:
    selected_id = st.selectbox("Select a student", options)
    row = student_df.loc[student_df["student_id"] == selected_id].iloc[0]
    idx = int(row.name)  # position in X / shap_values
    level = str(row["risk_level"])
    color = RISK_COLORS.get(level, "#95a5a6")

    st.markdown(
        f"""
        <div style="background:rgba(128,128,128,0.08);
                    border-left:5px solid {color};
                    border-radius:8px; padding:1rem 1.2rem;
                    margin-bottom:1rem;">
            <h3 style="margin:0;">Student {selected_id}</h3>
            <p style="margin:0.3rem 0; font-size:1.4rem;
                      color:{color}; font-weight:700;">
                Risk score: {row['risk_pct']:.1f}% &nbsp;|&nbsp; {level} risk
            </p>
            <p style="margin:0; opacity:0.7;">
                Suggested action: teacher review recommended
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --- key metrics vs cohort median ---
    metric_defs = [
        ("avg_early_score", "Early score", "{:.1f}"),
        ("submission_rate", "Submission rate", "{:.0%}"),
        ("total_clicks", "Platform clicks", "{:,.0f}"),
        ("days_since_last_active", "Days inactive", "{:.0f}"),
    ]
    metric_defs = [m for m in metric_defs if m[0] in student_df.columns]
    if metric_defs:
        cols = st.columns(len(metric_defs))
        for col, (key, label, fmt) in zip(cols, metric_defs):
            median = student_df[key].median()
            col.metric(label, fmt.format(row[key]),
                       f"cohort median {fmt.format(median)}",
                       delta_color="off")

    # --- plain-English reasons from SHAP ---
    st.markdown("####  Why was this student flagged?")
    sv = pd.Series(shap_values[idx], index=feature_names)
    top_risk = sv[sv > 0].sort_values(ascending=False).head(3)
    top_protect = sv[sv < 0].sort_values().head(2)

    a, b = st.columns(2)
    with a:
        st.markdown("** Risk-increasing signals**")
        if len(top_risk):
            for feat in top_risk.index:
                st.markdown(f"- {EXPLANATIONS.get(feat, feat)}")
        else:
            st.markdown("- No strong risk signals detected")
    with b:
        st.markdown("** Protective factors**")
        if len(top_protect):
            for feat in top_protect.index:
                st.markdown(f"- {PROTECTIVE.get(feat, feat)}")
        else:
            st.markdown("- No strong protective factors detected")

    # --- lightweight SHAP bar chart (matplotlib only) ---
    st.markdown("####  Signal breakdown")
    top = sv.reindex(sv.abs().sort_values(ascending=False).head(10).index)
    top = top.iloc[::-1]  # biggest at the top of the chart

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(
        top.index, top.values,
        color=["#e74c3c" if v > 0 else "#3498db" for v in top.values],
    )
    ax.axvline(0, color="grey", linewidth=0.8)
    ax.set_xlabel("Impact on risk  (red raises risk, blue lowers it)")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # --- intervention log (session only) ---
    st.markdown("####  Teacher action panel")
    st.session_state.setdefault("action_log", [])

    x1, x2 = st.columns(2)
    with x1:
        intervention = st.selectbox(
            "Log intervention taken",
            [
                "-- Select --",
                "One-to-one conversation",
                "Parent/guardian contacted",
                "Referred to student support",
                "Additional tutoring arranged",
                "Assignment extension granted",
                "Welfare check requested",
                "No action needed",
            ],
        )
    with x2:
        notes = st.text_area("Teacher notes", height=100,
                             placeholder="Observations or context...")

    if st.button("💾Save action", type="primary"):
        if intervention == "-- Select --":
            st.warning("Please choose an intervention first.")
        else:
            st.session_state["action_log"].append({
                "time": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "student": selected_id,
                "risk_%": row["risk_pct"],
                "intervention": intervention,
                "notes": notes,
            })
            st.success(f"Logged for {selected_id}: {intervention}")

    if st.session_state["action_log"]:
        log_df = pd.DataFrame(st.session_state["action_log"])
        with st.expander(f"Actions logged this session ({len(log_df)})"):
            stretch(st.dataframe, log_df, hide_index=True)
            st.download_button(
                " Download action log", log_df.to_csv(index=False),
                "action_log.csv", "text/csv",
            )
            st.caption(
                "The log lives in this browser session only. Download it "
                "to keep it."
            )

st.markdown("---")
st.caption(
    " Student Learning Risk Early Warning System · predictions support "
    "teacher judgment, not replace it · built on OULAD"
)
