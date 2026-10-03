"""
Cyberattack Pattern Mining Across Networks
Data Warehousing & Mining (DWM) mini project - Streamlit app.
"""
import streamlit as st

st.set_page_config(page_title="Cyberattack Pattern Mining", page_icon="🛡️", layout="wide",
                   initial_sidebar_state="collapsed")

from core import nav, ui  # noqa: E402  (ui sets the Plotly theme on import)
from core.data import DS1, DS2, load_raw  # noqa: E402
from views import (association, clustering, comparison, dashboard, dataset, j48, naive_bayes,  # noqa: E402
                   preprocessing, regression, warehouse)

ui.inject_css()


def page(fn, title, path, icon, default=False):
    return st.Page(fn, title=title, url_path=path, icon=icon, default=default)


PAGES = {
    "Overview": [page(dashboard.render, "Overview", "overview", ":material/radar:", default=True),
                 page(dataset.render, "Datasets", "datasets", ":material/database:")],
    "Warehouse": [page(warehouse.render, "Data Warehouse & OLAP", "warehouse", ":material/view_in_ar:"),
                  page(preprocessing.render, "Preprocessing", "preprocessing", ":material/cleaning_services:")],
    "Mining": [page(association.render, "Association Rules", "association-rules", ":material/hub:"),
               page(j48.render, "J48 Decision Tree", "j48", ":material/account_tree:"),
               page(naive_bayes.render, "Naive Bayes", "naive-bayes", ":material/casino:"),
               page(regression.render, "Regression", "regression", ":material/show_chart:"),
               page(clustering.render, "Clustering", "clustering", ":material/bubble_chart:")],
    "Evaluation": [page(comparison.render, "Model Comparison", "model-comparison", ":material/leaderboard:")],
}
nav.PAGES = {p.url_path: p for group in PAGES.values() for p in group}

st.navigation(PAGES, position="top").run()

up = st.session_state.get("upload_df")
ui.footer(["Cyberattack Pattern Mining Across Networks · DWM mini project",
           f"DS1 network traffic · {len(load_raw(DS1)):,} rows",
           f"DS2 cyber incidents · {len(load_raw(DS2)):,} rows"]
          + ([f"Uploaded · {len(up):,} rows"] if up is not None else []))
