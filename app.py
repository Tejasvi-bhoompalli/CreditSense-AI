"""CreditSense AI - application entry point.

Run with::

    streamlit run app.py

Navigation is defined explicitly with ``st.navigation`` so page order, labels and
grouping match the product specification and are not derived from the file system.
"""

from __future__ import annotations

import streamlit as st

from src.ui import configure_page, footer, load_css

# set_page_config must be the first Streamlit call, and app.py is the entry point,
# so the pages themselves must never call it again.
configure_page()
load_css()

NAV_SYSTEM = "Credit risk decision support"

PAGES = [
    st.Page("pages/dashboard.py", title="Overview", default=True,
            icon=":material/dashboard:"),
    st.Page("pages/risk_assessment.py", title="Risk Assessment",
            icon=":material/assignment:"),
    st.Page("pages/what_if_analysis.py", title="What-if Simulator",
            icon=":material/tune:"),
    st.Page("pages/explainable_ai.py", title="Explainable AI",
            icon=":material/hub:"),
    st.Page("pages/model_insights.py", title="Model Intelligence",
            icon=":material/monitoring:"),
    st.Page("pages/dataset_explorer.py", title="Dataset Explorer",
            icon=":material/table_view:"),
    st.Page("pages/assessment_history.py", title="Assessment History",
            icon=":material/history:"),
    st.Page("pages/reports.py", title="Reports", icon=":material/description:"),
    st.Page("pages/about.py", title="About & Methodology", icon=":material/info:"),
]

navigation = st.navigation({NAV_SYSTEM: PAGES}, position="sidebar")

with st.sidebar:
    st.markdown(
        '<div style="font-size:0.72rem;color:#93a3c4;line-height:1.5;">'
        "Academic prototype &middot; not a lender, credit bureau or regulated "
        "underwriting process.</div>",
        unsafe_allow_html=True,
    )

navigation.run()

footer()
