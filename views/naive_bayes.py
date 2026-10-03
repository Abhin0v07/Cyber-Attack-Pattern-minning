import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

from core import ui
from core.data import discretize, numeric_cols
from core.models import MixedNB, evaluate
from views.common import class_setup, input_form, show_evaluation


def prepare(df, feats, log_numeric, discretise, bins):
    X = df[feats].copy()
    num = [f for f in feats if f in numeric_cols(df)]
    if log_numeric:
        for f in num:
            if X[f].min() >= 0 and X[f].max() > 100:
                X[f] = np.log1p(X[f])
    if discretise:
        for f in num:
            X[f] = discretize(X[f], bins, "width", [f"bin{i + 1}" for i in range(bins)]) if X[f].nunique() > bins \
                else X[f].astype(str)
        num = []
    nom = [f for f in feats if f not in num]
    return X, num, nom


@st.cache_data(show_spinner=False)
def train(df, target, feats, mode, split, log_numeric, discretise, bins, alpha):
    X, num, nom = prepare(df, list(feats), log_numeric, discretise, bins)
    model = MixedNB(numeric=tuple(num), nominal=tuple(nom), alpha=alpha)
    res = evaluate(model, X, df[target].astype(str), mode=mode, split=split)
    return res, X, num, nom


def render():
    ui.hero("Classification · Naive Bayes", "Naive Bayes Classifier",
            "A probabilistic classifier applying Bayes' theorem with the 'naive' assumption that attributes are "
            "conditionally independent given the class — fast, robust and surprisingly strong on security logs.")
    ui.theory("<b>P(C|X)</b> = P(X|C)·P(C) / P(X) &nbsp;and with independence &nbsp;"
              "<b>P(X|C) = Π<sub>k</sub> P(x<sub>k</sub>|C)</b>. Nominal attributes use frequency tables with "
              "<b>Laplace correction</b> (count+1)/(n+|values|); numeric attributes use a <b>Gaussian</b> "
              "g(x, μ<sub>C</sub>, σ<sub>C</sub>) — exactly as Weka's <code>NaiveBayes</code>. Predict the class with "
              "the maximum posterior.")

    left, right = st.columns([1, 2.6])
    with left:
        name, df, target, feats, mode, split = class_setup("nb")
        st.markdown("**Naive Bayes options**")
        log_numeric = st.checkbox("log(1+x) on skewed numeric attributes", value=True,
                                  help="Byte counts are heavy-tailed; a log makes them closer to Gaussian.")
        discretise = st.checkbox("Discretise numeric attributes (-D)", value=False)
        bins = st.slider("Bins", 3, 10, 5, disabled=not discretise)
        alpha = st.select_slider("Laplace smoothing α", [0.0001, 0.5, 1.0, 2.0], value=1.0)
    if not feats:
        st.warning("Choose at least one attribute.")
        return
    with st.spinner("Estimating probabilities…"):
        res, X, num, nom = train(df, target, tuple(feats), mode, split, log_numeric, discretise, bins, alpha)
    nb = res["model"]
    classes = list(nb.classes_)
    with right:
        st.markdown(f"**Dataset:** {name} · {len(df):,} instances · {len(num)} Gaussian + {len(nom)} nominal attributes"
                    f" · test mode: {'10-fold CV' if mode == 'cv' else f'{split:.0%} split'}")
        show_evaluation(res, "Naive Bayes")

    tabs = st.tabs(["🅿️ Prior probabilities", "📑 Likelihood tables", "🧮 Posterior step-by-step",
                    "🔮 Classify a new record"])
    with tabs[0]:
        pri = pd.DataFrame({"class": classes, "count": nb.class_count_, "P(C)": nb.prior_})
        c1, c2 = st.columns([1, 1.3])
        c1.dataframe(pri.round(4), hide_index=True, width="stretch")
        fig = px.bar(pri, x="class", y="P(C)", color="class", color_discrete_map=ui.ATTACK_COLORS, title="Class priors")
        fig.update_layout(showlegend=False)
        with c2:
            ui.show(fig, 320)

    with tabs[1]:
        attr = st.selectbox("Attribute", list(num) + list(nom))
        if attr in nom:
            vals = nb.values_[attr]
            cpt = pd.DataFrame(nb.cpt_[attr], index=classes, columns=vals)
            st.markdown(f"**P({attr} = v | class)** with Laplace α = {alpha}")
            fig = px.imshow(cpt.round(3), text_auto=True, aspect="auto", color_continuous_scale=ui.SEQ,
                            title=f"Conditional probability table · {attr}")
            ui.show(fig, 360 if len(vals) < 15 else 520)
        else:
            tab = pd.DataFrame({"class": classes, "mean μ": nb.mean_[attr], "std dev σ": nb.std_[attr]})
            c1, c2 = st.columns([1, 1.6])
            c1.dataframe(tab.round(4), hide_index=True, width="stretch")
            c1.caption("log(1+x) scale" if log_numeric and df[attr].max() > 100 else "")
            xs = np.linspace(X[attr].min(), X[attr].max(), 400)
            fig = go.Figure()
            for i, c in enumerate(classes):
                m, s = nb.mean_[attr][i], nb.std_[attr][i]
                fig.add_scatter(x=xs, y=np.exp(-(xs - m) ** 2 / (2 * s ** 2)) / (s * np.sqrt(2 * np.pi)), name=c,
                                line=dict(color=ui.ATTACK_COLORS.get(c, ui.PALETTE[i % 10])))
            fig.update_layout(title=f"Gaussian likelihood g({attr} | class)", yaxis_type="log",
                              yaxis_range=[-4, None])
            with c2:
                ui.show(fig, 380)

    with tabs[2]:
        st.markdown("Pick a test record; the table multiplies the prior by every attribute likelihood "
                    "(computed in log-space to avoid underflow) and normalises.")
        idx = st.number_input("Record index", 0, len(df) - 1, int(df.index[df[target] != df[target].mode()[0]][0]))
        rec = X.iloc[[idx]]
        terms = nb._log_terms(rec)
        rows = [{"factor": "prior P(C)", **{c: nb.prior_[i] for i, c in enumerate(classes)}}]
        for f, t in terms.items():
            v = rec.iloc[0][f]
            rows.append({"factor": f"P({f}={v if isinstance(v, str) else round(float(v), 3)} | C)",
                         **{c: float(np.exp(t[0, i])) for i, c in enumerate(classes)}})
        tab = pd.DataFrame(rows).set_index("factor")
        joint = pd.Series(np.log(nb.prior_) + sum(t[0] for t in terms.values()), index=classes)
        post = np.exp(joint - joint.max())
        post = post / post.sum()
        tab.loc["log P(C)·ΠP(x|C)"] = joint
        tab.loc["posterior P(C|X)"] = post
        st.dataframe(tab.style.format("{:.4g}").highlight_max(axis=1, subset=pd.IndexSlice[["posterior P(C|X)"], :],
                                                            color="#ffd8a8"), width="stretch")
        pred = post.idxmax()
        actual = df[target].iloc[idx]
        (st.success if pred == actual else st.error)(
            f"Predicted **{pred}** (posterior {post.max():.2%}) · actual class **{actual}**")

    with tabs[3]:
        row = input_form(df, list(feats), "nbp")
        Xn, _, _ = prepare(pd.concat([df[list(feats)], row], ignore_index=True), list(feats), log_numeric, discretise, bins)
        p = nb.predict_proba(Xn.tail(1))[0]
        c1, c2 = st.columns([1, 1.5])
        c1.metric("Predicted class", classes[int(np.argmax(p))], f"posterior {p.max():.1%}", delta_color="off")
        fig = px.bar(x=classes, y=p, color=classes, color_discrete_map=ui.ATTACK_COLORS, title="Posterior P(C|X)",
                     labels={"x": "", "y": "probability"})
        fig.update_layout(showlegend=False)
        with c2:
            ui.show(fig, 300)
