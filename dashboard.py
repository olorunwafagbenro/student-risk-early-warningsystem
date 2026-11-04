# ============================================================
# dashboard.py
# Student Learning Risk Early Warning System
# Teacher-Facing Dashboard
# ============================================================
import gzip
import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import shap
import joblib
import pickle
import json
import plotly.express as px
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')

# ---- Page configuration ------------------------------------
st.set_page_config(
    page_title = "Student Risk Dashboard",
    page_icon  = "",
    layout     = "wide",
    initial_sidebar_state = "expanded"
)

# ---- Custom CSS --------------------------------------------
st.markdown("""
<style>
    .main-header {
        font-size: 2rem;
        font-weight: 700;
        color: #2c3e50;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 0.95rem;
        color: #7f8c8d;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        border-left: 4px solid #3498db;
    }
    .risk-high   { color: #e74c3c; font-weight: 700; }
    .risk-medium { color: #e67e22; font-weight: 700; }
    .risk-low    { color: #2ecc71; font-weight: 700; }
    .stDataFrame { border-radius: 8px; }
</style>
""", unsafe_allow_html=True)
# ============================================================
# LOAD MODEL ARTIFACTS
# ============================================================

import os, json, gzip, pickle

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_resource
def load_artifacts():
    """Load only what the dashboard displays: config + SHAP output."""
    config_path = os.path.join(BASE_DIR, 'models', 'model_config.json')
    shap_gz     = os.path.join(BASE_DIR, 'models', 'shap_output.pkl.gz')
    shap_pkl    = os.path.join(BASE_DIR, 'models', 'shap_output.pkl')

    with open(config_path, 'r') as f:
        config = json.load(f)

    if os.path.exists(shap_gz):
        with gzip.open(shap_gz, 'rb') as f:
            shap_output = pickle.load(f)
    elif os.path.exists(shap_pkl):
        with open(shap_pkl, 'rb') as f:
            shap_output = pickle.load(f)
    else:
        st.error("SHAP output file not found in models/.")
        st.stop()

    return config, shap_output

config, shap_output = load_artifacts()

# Unpack SHAP output
shap_values   = shap_output['shap_values']
X_test        = shap_output['X_test']
y_test        = shap_output['y_test']
risk_probas   = shap_output['risk_probabilities']
feature_names = shap_output['feature_names']
base_value    = shap_output['base_value']

# ============================================================
# BUILD STUDENT DATAFRAME FOR DISPLAY
# ============================================================

@st.cache_data
def build_student_df(_X_test, _y_test, _risk_probas, _config):
    """
    Build the main student display dataframe.
    Uses underscore prefix to avoid Streamlit hash issues.
    """
    df = _X_test.copy()
    df['actual']      = _y_test.values
    df['risk_proba']  = _risk_probas
    df['risk_pct']    = (_risk_probas * 100).round(1)

    # Assign risk level
    df['risk_level'] = pd.cut(
        df['risk_proba'],
        bins   = [0, 0.33, 0.66, 1.0],
        labels = ['Low', 'Medium', 'High']
    )

    # Apply threshold from config
    threshold         = _config.get('threshold', 0.5)
    df['flagged']     = (df['risk_proba'] >= threshold).astype(int)

    # Generate student IDs
    df['student_id']  = [
        f"S{1000 + i}" for i in range(len(df))
    ]

    return df.reset_index(drop=True)

student_df = build_student_df(
    X_test, y_test, risk_probas, config
)

# ============================================================
# SIDEBAR — FILTERS AND NAVIGATION
# ============================================================

with st.sidebar:
    st.image(
        "https://img.icons8.com/color/96/graduation-cap.png",
        width=60
    )
    st.title(" Risk Dashboard")
    st.markdown("---")

    st.subheader("Filters")

    # Risk level filter
    risk_filter = st.multiselect(
        "Risk Level",
        options = ['High', 'Medium', 'Low'],
        default = ['High', 'Medium']
    )

    # Flagged only toggle
    flagged_only = st.toggle(
        "Show flagged students only",
        value = True
    )

    # Risk score range
    score_range = st.slider(
        "Risk Score Range (%)",
        min_value = 0,
        max_value = 100,
        value     = (0, 100)
    )

    # Search by student ID
    search_id = st.text_input(
        "Search Student ID",
        placeholder = "e.g. S1042"
    )

    st.markdown("---")
    st.subheader("Model Info")
    st.caption(f"Model: {config.get('model_name', 'Unknown')}")
    st.caption(f"Threshold: {config.get('threshold', 0.5):.0%}")
    st.caption(f"Trained: {config.get('training_date', 'N/A')}")
    st.caption(f"Features: {len(feature_names)}")

    st.markdown("---")

    # Export button
    st.subheader("Export")
    flagged_export = student_df[student_df['flagged'] == 1][[
        'student_id', 'risk_pct', 'risk_level',
        'avg_early_score', 'submission_rate', 'total_clicks'
    ]].copy()
    flagged_export.columns = [
        'Student ID', 'Risk %', 'Risk Level',
        'Avg Score', 'Submission Rate', 'Total Clicks'
    ]
    csv = flagged_export.to_csv(index=False)
    st.download_button(
        label     = "⬇Download Flagged Students",
        data      = csv,
        file_name = "flagged_students.csv",
        mime      = "text/csv"
    )

    # ============================================================
# MAIN CONTENT — HEADER
# ============================================================

st.markdown(
    '<p class="main-header">🎓 Student Learning Risk Dashboard</p>',
    unsafe_allow_html=True
)
st.markdown(
    '<p class="sub-header">Early warning system — '
    'predictions support teacher judgment, not replace it.</p>',
    unsafe_allow_html=True
)

# ============================================================
# OVERVIEW METRIC CARDS
# ============================================================

total       = len(student_df)
high_risk   = (student_df['risk_level'] == 'High').sum()
medium_risk = (student_df['risk_level'] == 'Medium').sum()
low_risk    = (student_df['risk_level'] == 'Low').sum()
flagged     = student_df['flagged'].sum()
avg_score   = student_df['risk_pct'].mean()

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric("Total Students", f"{total:,}")

with col2:
    st.metric(
        "High Risk",
        f"{high_risk:,}",
        delta=f"{high_risk/total:.1%} of cohort",
        delta_color="inverse"
    )

with col3:
    st.metric(
        "Medium Risk",
        f"{medium_risk:,}",
        delta=f"{medium_risk/total:.1%} of cohort",
        delta_color="off"
    )

with col4:
    st.metric(
        "Low Risk",
        f"{low_risk:,}",
        delta=f"{low_risk/total:.1%} of cohort"
    )

with col5:
    st.metric(
        "Flagged for Review",
        f"{flagged:,}",
        delta=f"Avg risk: {avg_score:.1f}%",
        delta_color="inverse"
    )

st.markdown("---")

# ============================================================
# RISK DISTRIBUTION CHARTS
# ============================================================

col_left, col_right = st.columns([1, 1])

with col_left:
    st.subheader("Risk Level Distribution")
    risk_counts = student_df['risk_level'].value_counts()

    fig_pie = go.Figure(data=[go.Pie(
        labels = risk_counts.index,
        values = risk_counts.values,
        hole   = 0.45,
        marker_colors = ['#e74c3c', '#2ecc71', '#e67e22'],
        textinfo = 'label+percent'
    )])
    fig_pie.update_layout(
        showlegend    = False,
        height        = 300,
        margin        = dict(t=20, b=20, l=20, r=20),
        paper_bgcolor = 'rgba(0,0,0,0)',
        plot_bgcolor  = 'rgba(0,0,0,0)'
    )
    st.plotly_chart(fig_pie, use_container_width=True)

with col_right:
    st.subheader("Risk Score Distribution")
    fig_hist = px.histogram(
        student_df,
        x        = 'risk_pct',
        color    = 'risk_level',
        nbins    = 40,
        color_discrete_map = {
            'High':   '#e74c3c',
            'Medium': '#e67e22',
            'Low':    '#2ecc71'
        },
        labels = {'risk_pct': 'Risk Score (%)'}
    )
    fig_hist.update_layout(
        height        = 300,
        showlegend    = True,
        margin        = dict(t=20, b=20, l=20, r=20),
        paper_bgcolor = 'rgba(0,0,0,0)',
        plot_bgcolor  = 'rgba(0,0,0,0)'
    )
    st.plotly_chart(fig_hist, use_container_width=True)

st.markdown("---")

# ============================================================
# STUDENT TABLE — FILTERED AND SORTABLE
# ============================================================

st.subheader("Student Risk Table")

# Apply filters
filtered_df = student_df.copy()

if risk_filter:
    filtered_df = filtered_df[
        filtered_df['risk_level'].isin(risk_filter)
    ]

if flagged_only:
    filtered_df = filtered_df[filtered_df['flagged'] == 1]

filtered_df = filtered_df[
    (filtered_df['risk_pct'] >= score_range[0]) &
    (filtered_df['risk_pct'] <= score_range[1])
]

if search_id:
    filtered_df = filtered_df[
        filtered_df['student_id'].str.contains(
            search_id, case=False
        )
    ]

# Sort by risk
filtered_df = filtered_df.sort_values(
    'risk_proba', ascending=False
)

# Display columns
display_cols = {
    'student_id':          'Student ID',
    'risk_pct':            'Risk Score (%)',
    'risk_level':          'Risk Level',
    'avg_early_score':     'Early Score',
    'submission_rate':     'Submission Rate',
    'total_clicks':        'Platform Clicks',
    'days_since_last_active': 'Days Inactive'
}

display_df = filtered_df[
    list(display_cols.keys())
].rename(columns=display_cols)

# Colour-code risk level
def colour_risk(val):
    if val == 'High':
        return 'background-color: #fde8e8; color: #e74c3c; font-weight: bold'
    elif val == 'Medium':
        return 'background-color: #fef3e2; color: #e67e22; font-weight: bold'
    else:
        return 'background-color: #e8f8f0; color: #2ecc71; font-weight: bold'

styled_df = display_df.style.map(
    colour_risk, subset=['Risk Level']
).format({
    'Risk Score (%)':   '{:.1f}%',
    'Early Score':      '{:.1f}',
    'Submission Rate':  '{:.1%}',
    'Platform Clicks':  '{:,.0f}',
    'Days Inactive':    '{:.0f}'
})

st.dataframe(styled_df, use_container_width=True, height=350)
st.caption(
    f"Showing {len(filtered_df):,} of {total:,} students"
)

st.markdown("---")

# ============================================================
# INDIVIDUAL STUDENT DEEP DIVE
# ============================================================

st.subheader("Individual Student Analysis")

# Student selector
student_options = filtered_df['student_id'].tolist()

if not student_options:
    st.warning("No students match the current filters.")
else:
    selected_id = st.selectbox(
        "Select a student to analyse:",
        options     = student_options,
        index       = 0
    )

    selected_row = filtered_df[
        filtered_df['student_id'] == selected_id
    ].iloc[0]
    selected_idx = filtered_df[
        filtered_df['student_id'] == selected_id
    ].index[0]

    # ---- Student header card ---
    risk_color = {
        'High':   '#e74c3c',
        'Medium': '#e67e22',
        'Low':    '#2ecc71'
    }.get(str(selected_row['risk_level']), '#95a5a6')

    st.markdown(f"""
    <div style="
        background: #f8f9fa;
        border-left: 5px solid {risk_color};
        border-radius: 8px;
        padding: 1.2rem;
        margin-bottom: 1rem;
    ">
        <h3 style="margin:0; color:#2c3e50;">
            Student {selected_id}
        </h3>
        <p style="margin:0.3rem 0; font-size:1.4rem;
                  color:{risk_color}; font-weight:700;">
            Risk Score: {selected_row['risk_pct']:.1f}%
            &nbsp;|&nbsp; {selected_row['risk_level']} Risk
        </p>
        <p style="margin:0; color:#7f8c8d;">
            Suggested action: Teacher review recommended
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ---- Key metrics ---
    m1, m2, m3, m4 = st.columns(4)

    with m1:
        st.metric(
            "Early Assessment Score",
            f"{selected_row.get('avg_early_score', 0):.1f}",
            delta="Below average" if
            selected_row.get('avg_early_score', 0) < 50
            else "Above average"
        )
    with m2:
        st.metric(
            "Submission Rate",
            f"{selected_row.get('submission_rate', 0):.0%}",
            delta="Low" if
            selected_row.get('submission_rate', 0) < 0.6
            else "OK"
        )
    with m3:
        st.metric(
            "Platform Clicks",
            f"{selected_row.get('total_clicks', 0):,.0f}",
            delta="Low activity" if
            selected_row.get('total_clicks', 0) < 500
            else "Active"
        )
    with m4:
        st.metric(
            "Days Inactive",
            f"{selected_row.get('days_since_last_active', 0):.0f}",
            delta="Long absence" if
            selected_row.get('days_since_last_active', 0) > 14
            else "Recent activity",
            delta_color="inverse"
        )

    st.markdown("####  Why Was This Student Flagged?")

    # ---- SHAP waterfall explanation ---
    explanation_map = {
        'engagement_dropoff':
            'Platform engagement has dropped significantly',
        'total_clicks':
            'Very low overall platform activity',
        'avg_early_score':
            'Assessment scores were low from the start',
        'score_trend':
            'Assessment scores have been declining',
        'submission_rate':
            'Assignment completion rate is below average',
        'late_submission_count':
            'Multiple assignments submitted late',
        'days_since_last_active':
            'Has not engaged with the platform recently',
        'num_of_prev_attempts':
            'Has attempted this course before without success',
        'early_clicks':
            'Low engagement in the first weeks of term',
        'late_registration':
            'Registered for the course after it started'
    }

    shap_series = pd.Series(
        shap_values[selected_idx],
        index=feature_names
    ).sort_values(ascending=False)

    top_risk    = shap_series[shap_series > 0].head(3)
    top_protect = shap_series[shap_series < 0].head(2)

    col_risk, col_protect = st.columns(2)

    with col_risk:
        st.markdown("**Risk-increasing signals:**")
        for feat, val in top_risk.items():
            msg = explanation_map.get(feat, feat)
            st.markdown(f"• {msg}")

    with col_protect:
        st.markdown("**Protective factors:**")
        if len(top_protect) > 0:
            for feat, val in top_protect.items():
                msg = explanation_map.get(feat, feat)
                st.markdown(f"• {msg}")
        else:
            st.markdown("• No strong protective factors detected")

    # ---- SHAP waterfall chart ---
    st.markdown("#### Full Signal Breakdown")

    explanation = shap.Explanation(
        values        = shap_values[selected_idx],
        base_values   = base_value,
        data          = X_test.iloc[selected_idx].values,
        feature_names = feature_names
    )

    fig_shap, ax = plt.subplots(figsize=(10, 5))
    shap.plots.waterfall(
        explanation,
        max_display = 10,
        show        = False
    )
    plt.tight_layout()
    st.pyplot(fig_shap)
    plt.close()

    # ---- Teacher action panel ---
    st.markdown("#### Teacher Action Panel")

    action_col1, action_col2 = st.columns(2)

    with action_col1:
        intervention = st.selectbox(
            "Log intervention taken:",
            options = [
                "-- Select --",
                "One-to-one conversation",
                "Parent/guardian contacted",
                "Referred to student support",
                "Additional tutoring arranged",
                "Assignment extension granted",
                "Welfare check requested",
                "No action needed"
            ]
        )

    with action_col2:
        notes = st.text_area(
            "Teacher notes:",
            placeholder = "Add any observations or context...",
            height      = 100
        )

    if st.button("Save Action", type="primary"):
        if intervention != "-- Select --":
            st.success(
                f"Action logged for {selected_id}: "
                f"{intervention}"
            )
        else:
            st.warning("Please select an intervention first.")

st.markdown("---")

# ============================================================
# FOOTER
# ============================================================

st.markdown("""
<div style="
    text-align: center;
    color: #95a5a6;
    font-size: 0.8rem;
    padding: 1rem;
    border-top: 1px solid #ecf0f1;
    margin-top: 2rem;
">
    Student Learning Risk Early Warning System &nbsp;|&nbsp;
    Predictions support teacher judgment — not replace it &nbsp;|&nbsp;
    Built with OULAD Dataset + XGBoost + SHAP
</div>
""", unsafe_allow_html=True)
