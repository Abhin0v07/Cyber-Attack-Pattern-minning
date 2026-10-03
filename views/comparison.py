import time

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from core import ui
from core.data import encode_features, numeric_cols, sample
from core.models import MixedNB, evaluate, per_class, summary
from views.common import class_setup

MODELS = ["ZeroR (baseline)", "J48 (C4.5)", "Naive Bayes", "k-NN (IBk, k=5)", "Logistic regression", "Random forest"]


def build(name, df, feats):
    num = [f for f in feats if f in numeric_cols(df)]
    if name == "Naive Bayes":
        X = df[feats].copy()
        for f in num:
            if X[f].min() >= 0 and X[f].max() > 100:
                X[f] = np.log1p(X[f])
        return MixedNB(numeric=tuple(num), nominal=tuple(f for f in feats if f not in num)), X
    X = encode_features(df, feats)
    if name.startswith("ZeroR"):
        return DummyClassifier(strategy="most_frequent"), X
    if name.startswith("J48"):
        return DecisionTreeClassifier(criterion="entropy", min_samples_leaf=2, ccp_alpha=0.0005, random_state=42), X
    X = X.copy()
    for f in num:
        if X[f].min() >= 0 and X[f].max() > 100:
            X[f] = np.log1p(X[f])
    if name.startswith("k-NN"):
        return make_pipeline(StandardScaler(), KNeighborsClassifier(5)), X
    if name.startswith("Logistic"):
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=600)), X
    return RandomForestClassifier(150, n_jobs=-1, random_state=42), X


@st.cache_data(show_spinner=False)
def run_all(df, target, feats, mode, split, models):
    out, per = [], []
    y = df[target].astype(str)
    for m in models:
        model, X = build(m, df, list(feats))
        t0 = time.perf_counter()
        res = evaluate(model, X, y, mode=mode, split=split, folds=5)
        s = summary(res)
        s.update(model=m, seconds=time.perf_counter() - t0)
        out.append(s)
        pc, _ = per_class(res)
        per.append(pc.assign(model=m))
    return pd.DataFrame(out), pd.concat(per)


def render():
    ui.hero("Evaluation", "Model Comparison & Best Model",
            "J48 and Naive Bayes compared against a ZeroR baseline and three other classifiers on the same data "
            "and the same test protocol.")
    left, right = st.columns([1, 2.6])
    with left:
        name, df, target, feats, mode, split = class_setup("cmp")
        models = st.multiselect("Classifiers", MODELS, default=MODELS)
        rows = st.select_slider("Rows used", [2000, 5000, 10000, 20000, 40000], value=10000,
                                help="Sampling keeps the comparison quick on the free cloud tier.")
        go = st.button("▶ Run comparison", type="primary", width="stretch")
    if not feats or not models:
        st.warning("Choose attributes and classifiers.")
        return
    key = (name, target, tuple(feats), mode, split, tuple(models), rows)
    if go:
        st.session_state["cmp_key"] = key
    if st.session_state.get("cmp_key") != key:
        with right:
            st.info("Press **Run comparison** to train every classifier "
                    f"({'5-fold cross-validation' if mode == 'cv' else f'{split:.0%} split'}).")
        return
    data = sample(df, rows, seed=7).reset_index(drop=True)
    with st.spinner("Training classifiers…"):
        res, per = run_all(data, target, tuple(feats), mode, split, tuple(models))
    res = res.set_index("model")
    best = res["f1"].idxmax()
    with right:
        ui.kpis([("Best model", best, "highest weighted F1", "cyan"),
                 ("Accuracy", f"{res.loc[best, 'accuracy']:.2%}", f"ZeroR: {res['accuracy'].min():.2%}", "violet"),
                 ("Weighted F1", f"{res.loc[best, 'f1']:.3f}", "precision/recall balance", "amber"),
                 ("Train+test time", f"{res.loc[best, 'seconds']:.2f}s", f"{len(data):,} rows", "red")])
        st.write("")
        m = res[["accuracy", "precision", "recall", "f1", "roc_auc"]].reset_index().melt(id_vars="model")
        fig = px.bar(m, x="variable", y="value", color="model", barmode="group", title="Metric comparison")
        fig.update_layout(yaxis_range=[0, 1.05], xaxis_title=None, legend_title=None)
        ui.show(fig, 380)
    c1, c2 = st.columns([1.2, 1])
    with c1:
        st.markdown("**Summary**")
        st.dataframe(res.round(4).style.highlight_max(subset=["accuracy", "precision", "recall", "f1", "kappa", "roc_auc"],
                                                      color="#ffd8a8").highlight_min(subset=["seconds"], color="#ffd8a8"),
                     width="stretch")
    with c2:
        h = per.pivot_table(index="model", columns="class", values="F-measure")
        fig = px.imshow(h.round(2), text_auto=True, color_continuous_scale=ui.SEQ, aspect="auto",
                        title="F-measure per class")
        ui.show(fig, 320)
    j, n = res.loc["J48 (C4.5)"] if "J48 (C4.5)" in res.index else None, \
        res.loc["Naive Bayes"] if "Naive Bayes" in res.index else None
    if j is not None and n is not None:
        winner = "J48" if j["f1"] >= n["f1"] else "Naive Bayes"
        st.markdown(f"#### J48 vs Naive Bayes → **{winner}**")
        st.markdown(
            f"- J48 accuracy **{j['accuracy']:.2%}** vs Naive Bayes **{n['accuracy']:.2%}** "
            f"(F1 {j['f1']:.3f} vs {n['f1']:.3f}).\n"
            "- J48 captures attribute *interactions* (e.g. `flag=S0` **and** high `count` ⇒ DoS) and yields readable "
            "rules; Naive Bayes assumes independence, which is violated by correlated features like "
            "`count`/`srv_count`/`serror_rate`.\n"
            "- Naive Bayes trains in a single pass, gives calibrated-looking probabilities and copes well with rare "
            "classes (e.g. U2R) when data is scarce.")
