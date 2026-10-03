import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from core import ui
from core.data import DS1, DS2, UPLOAD, available_datasets, cfg, get_df, numeric_cols, sample

STORY = {
    DS1: "Predict **traffic volume (packets)** of a connection from its byte counts, duration and connection rate — "
         "used for capacity planning and spotting floods whose packet counts don't match their payload.",
    DS2: "Predict the **financial loss (US$ M)** of a security incident from records breached, recovery time and "
         "users affected — the basis of cyber-insurance pricing and risk budgeting.",
}


def fit_block(name, key):
    df = get_df(name)
    c = cfg(name)
    num = [x for x in numeric_cols(df)]
    st.markdown(STORY.get(name, ""))
    left, right = st.columns([1, 2.6])
    with left:
        target = st.selectbox("Target (dependent y)", num, index=num.index(c["reg_target"]) if c["reg_target"] in num else 0,
                              key=f"{key}_t")
        kind = st.radio("Model", ["Simple linear", "Multiple linear", "Polynomial", "Ridge (regularised)"],
                        index=1, key=f"{key}_k")
        feat_opts = [x for x in num if x != target]
        if kind == "Simple linear":
            feats = [st.selectbox("Predictor (x)", feat_opts, index=feat_opts.index(c["reg_features"][0])
                                  if c["reg_features"] and c["reg_features"][0] in feat_opts else 0, key=f"{key}_x")]
        else:
            feats = st.multiselect("Predictors (x₁…xₙ)", feat_opts,
                                   default=[x for x in c["reg_features"] if x in feat_opts], key=f"{key}_f")
        degree = st.slider("Polynomial degree", 2, 4, 2, key=f"{key}_d") if kind == "Polynomial" else 1
        ridge_a = st.select_slider("Ridge α", [0.01, 0.1, 1.0, 10.0, 100.0], value=1.0, key=f"{key}_a") \
            if kind.startswith("Ridge") else None
        log_y = st.checkbox("log(1+y) target", value=False, key=f"{key}_ly")
        test = st.slider("Test %", 10, 50, 30, key=f"{key}_ts") / 100
    if not feats:
        st.warning("Choose at least one predictor.")
        return

    X = df[feats].astype(float)
    y = df[target].astype(float)
    y_fit = np.log1p(y.clip(lower=0)) if log_y else y
    Xtr, Xte, ytr, yte = train_test_split(X, y_fit, test_size=test, random_state=42)
    if kind == "Polynomial":
        model = make_pipeline(PolynomialFeatures(degree, include_bias=False), StandardScaler(), LinearRegression())
    elif kind.startswith("Ridge"):
        model = make_pipeline(StandardScaler(), Ridge(alpha=ridge_a))
    else:
        model = LinearRegression()
    model.fit(Xtr, ytr)
    ptr, pte = model.predict(Xtr), model.predict(Xte)
    if log_y:
        yte_o, pte_o = np.expm1(yte), np.expm1(pte)
    else:
        yte_o, pte_o = yte, pte
    r2 = r2_score(yte, pte)
    n, p = len(yte), Xte.shape[1] * (degree if kind == "Polynomial" else 1)
    adj = 1 - (1 - r2) * (n - 1) / max(n - p - 1, 1)
    rmse = np.sqrt(mean_squared_error(yte_o, pte_o))
    mae = mean_absolute_error(yte_o, pte_o)

    with right:
        ui.kpis([("R² (test)", f"{r2:.4f}", f"train R² {r2_score(ytr, ptr):.4f}", "cyan"),
                 ("Adjusted R²", f"{adj:.4f}", f"{n:,} test rows", "violet"),
                 ("RMSE", f"{rmse:,.3f}", f"in units of {target}", "amber"),
                 ("MAE", f"{mae:,.3f}", "mean absolute error", "red")])
        st.write("")
        if kind in ("Simple linear", "Multiple linear"):
            terms = " ".join(f"{'+' if b >= 0 else '−'} {abs(b):.4g}·{f}" for b, f in zip(model.coef_, feats))
            ylab = f"log(1+{target})" if log_y else target
            st.markdown(f"**Fitted equation**  \n`{ylab} = {model.intercept_:.4g} {terms}`")
        s = sample(pd.DataFrame({"actual": np.asarray(yte_o), "predicted": np.asarray(pte_o)}), 4000)
        c1, c2 = st.columns(2)
        with c1:
            fig = px.scatter(s, x="actual", y="predicted", opacity=.45, title="Actual vs predicted (test set)",
                             color_discrete_sequence=["#d9480f"])
            lo, hi = float(s.min().min()), float(s.max().max())
            fig.add_scatter(x=[lo, hi], y=[lo, hi], mode="lines", line=dict(color="#1d1c1a", dash="dash"),
                            name="ideal")
            fig.update_traces(marker_size=4, selector=dict(mode="markers"))
            fig.update_layout(showlegend=False)
            ui.show(fig, 360)
        with c2:
            res = np.asarray(yte) - np.asarray(pte)
            rs = sample(pd.DataFrame({"fitted": np.asarray(pte), "residual": res}), 4000)
            fig = px.scatter(rs, x="fitted", y="residual", opacity=.45, title="Residuals vs fitted",
                             color_discrete_sequence=["#1c7ed6"])
            fig.add_hline(y=0, line_color="#1d1c1a", line_dash="dash")
            fig.update_traces(marker_size=4)
            ui.show(fig, 360)

    tabs = st.tabs(["📐 Least-squares working", "📊 Coefficients", "🔮 Predict"])
    with tabs[0]:
        x = X[feats[0]]
        xm, ym = x.mean(), y_fit.mean()
        sxy = ((x - xm) * (y_fit - ym)).sum()
        sxx = ((x - xm) ** 2).sum()
        b1 = sxy / sxx if sxx else 0
        b0 = ym - b1 * xm
        r = np.corrcoef(x, y_fit)[0, 1]
        st.markdown(f"Simple linear regression of **{'log(1+' + target + ')' if log_y else target}** on "
                    f"**{feats[0]}** by the method of least squares:")
        st.latex(r"\beta_1=\frac{\sum (x_i-\bar x)(y_i-\bar y)}{\sum (x_i-\bar x)^2}"
                 rf"=\frac{{{sxy:,.4g}}}{{{sxx:,.4g}}}={b1:.6g}\qquad \beta_0=\bar y-\beta_1\bar x={b0:.6g}")
        st.latex(rf"\hat y = {b0:.4g} + {b1:.4g}\,x \qquad r = {r:.4f}\qquad r^2 = {r * r:.4f}")
        s = sample(pd.DataFrame({"x": x, "y": y_fit}), 3000)
        fig = px.scatter(s, x="x", y="y", opacity=.4, labels={"x": feats[0], "y": target},
                         title="Regression line", color_discrete_sequence=["#d9480f"])
        xs = np.linspace(x.min(), x.max(), 100)
        fig.add_scatter(x=xs, y=b0 + b1 * xs, mode="lines", line=dict(color="#c92a2a", width=3), name="ŷ")
        fig.update_traces(marker_size=4, selector=dict(mode="markers"))
        ui.show(fig, 380)

    with tabs[1]:
        if kind in ("Simple linear", "Multiple linear"):
            coef = pd.DataFrame({"term": ["intercept"] + feats, "coefficient": [model.intercept_] + list(model.coef_)})
            std = pd.DataFrame({"predictor": feats,
                                "standardised β": [b * X[f].std() / (y_fit.std() or 1) for b, f in zip(model.coef_, feats)]})
            c1, c2 = st.columns(2)
            c1.dataframe(coef, hide_index=True, width="stretch")
            fig = px.bar(std, x="standardised β", y="predictor", orientation="h",
                         title="Standardised coefficients (relative influence)", color_discrete_sequence=["#1d1c1a"])
            with c2:
                ui.show(fig, 300)
        else:
            inner = model[-1]
            names = model[0].get_feature_names_out(feats) if kind == "Polynomial" else feats
            coef = pd.DataFrame({"term": names, "coefficient (scaled)": inner.coef_}).sort_values(
                "coefficient (scaled)", key=abs, ascending=False)
            st.dataframe(coef, hide_index=True, width="stretch")
        corr = df[feats + [target]].corr()[target].drop(target).sort_values()
        fig = px.bar(corr.rename_axis("predictor").reset_index(name="r"), x="r", y="predictor", orientation="h",
                     title=f"Correlation of each predictor with {target}", color_discrete_sequence=["#d9480f"])
        ui.show(fig, 280)

    with tabs[2]:
        cols = st.columns(min(4, len(feats)))
        vals = {}
        for i, f in enumerate(feats):
            vals[f] = cols[i % len(cols)].number_input(f, value=float(X[f].median()), key=f"{key}_in_{f}")
        pv = model.predict(pd.DataFrame([vals]))[0]
        pv = np.expm1(pv) if log_y else pv
        st.metric(f"Predicted {target}", f"{pv:,.3f}")


def render():
    ui.hero("Regression", "Regression on Two Datasets",
            "Regression models a continuous dependent variable as a function of one or more predictors. "
            "Applied to connection traffic volume (DS1) and incident financial loss (DS2).")
    ui.theory("<b>Simple linear</b>: y = β₀ + β₁x &nbsp;·&nbsp; <b>Multiple linear</b>: y = β₀ + Σβᵢxᵢ &nbsp;·&nbsp; "
              "<b>Polynomial</b>: adds x², x₁x₂ … terms &nbsp;·&nbsp; <b>Ridge</b>: least squares + α‖β‖² penalty. "
              "Evaluated with <b>R²</b> = 1 − SS<sub>res</sub>/SS<sub>tot</sub>, adjusted R², RMSE and MAE on a "
              "held-out test set.")
    names = [n for n in available_datasets()]
    tabs = st.tabs([("🌐 " if n == DS1 else "🏢 " if n == DS2 else "⬆️ ") + n for n in names])
    for t, n in zip(tabs, names):
        with t:
            fit_block(n, {DS1: "r1", DS2: "r2", UPLOAD: "ru"}[n])
