import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import StandardScaler

from core import ui
from core.data import categorical_cols, cfg, dataset_picker, get_df, numeric_cols, sample


def render():
    ui.hero("Data Preprocessing", "Cleaning, Transformation & Reduction",
            "Real-world security logs are dirty: missing fields, duplicated events, extreme outliers and features on "
            "wildly different scales. These steps prepare the data for mining.")
    name = dataset_picker("pp_ds")
    raw = get_df(name, cleaned=False)
    num = [c for c in numeric_cols(raw)]

    tabs = st.tabs(["🩹 Missing & duplicates", "📦 Outliers", "📏 Normalisation", "🪣 Discretisation & smoothing",
                    "🧮 Feature selection", "🗜️ PCA reduction"])

    with tabs[0]:
        ui.theory("<b>Data cleaning</b> fills missing values (global constant, mean/median, mode) and removes "
                  "duplicate records that would bias support counts and class priors.")
        miss = raw.isna().sum()
        miss = miss[miss > 0].sort_values(ascending=False)
        c1, c2 = st.columns([1.2, 1])
        with c1:
            if len(miss):
                fig = px.bar(miss.rename_axis("column").reset_index(name="missing"), x="column", y="missing",
                             title="Missing values per column", color_discrete_sequence=["#c92a2a"])
                fig.update_layout(xaxis_title=None)
                ui.show(fig, 320)
            else:
                st.success("No missing values in this dataset.")
        with c2:
            num_strategy = st.radio("Numeric imputation", ["median", "mean", "drop rows"], horizontal=True)
            cat_strategy = st.radio("Categorical imputation", ["'Unknown'", "mode", "drop rows"], horizontal=True)
            drop_dups = st.checkbox("Remove duplicate rows", value=True)
        out = raw.drop_duplicates() if drop_dups else raw.copy()
        for c in miss.index:
            is_num = pd.api.types.is_numeric_dtype(out[c])
            strat = num_strategy if is_num else cat_strategy
            if strat == "drop rows":
                out = out[out[c].notna()]
            elif strat == "median":
                out[c] = out[c].fillna(out[c].median())
            elif strat == "mean":
                out[c] = out[c].fillna(out[c].mean())
            elif strat == "mode":
                out[c] = out[c].fillna(out[c].mode()[0])
            else:
                out[c] = out[c].fillna("Unknown")
        a, b, c, d = st.columns(4)
        a.metric("Rows before", f"{len(raw):,}")
        b.metric("Rows after", f"{len(out):,}", delta=f"{len(out) - len(raw):,}")
        c.metric("Missing before", f"{int(raw.isna().sum().sum()):,}")
        d.metric("Missing after", f"{int(out.isna().sum().sum()):,}")
        if len(miss):
            before_after = pd.DataFrame({"missing before": raw[miss.index].isna().sum(),
                                         "missing after": out[miss.index].isna().sum()})
            st.dataframe(before_after, width="stretch")

    df = get_df(name)
    num = numeric_cols(df)

    with tabs[1]:
        ui.theory("<b>Outliers</b> are values far from the rest. <b>IQR rule</b>: outside [Q1 − 1.5·IQR, Q3 + 1.5·IQR]. "
                  "<b>Z-score rule</b>: |z| &gt; 3. In intrusion data, outliers are often the attacks themselves "
                  "(floods with huge <code>count</code>, exfiltration with huge <code>src_bytes</code>).")
        c1, c2 = st.columns([1, 2])
        col = c1.selectbox("Attribute", num, index=num.index("src_bytes") if "src_bytes" in num else 0)
        log = c1.checkbox("Log scale", value=True)
        s = df[col]
        q1, q3 = s.quantile([.25, .75])
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        iqr_out = ((s < lo) | (s > hi)).sum()
        z = (s - s.mean()) / (s.std() or 1)
        z_out = (z.abs() > 3).sum()
        c1.metric("IQR outliers", f"{iqr_out:,}", f"{iqr_out / len(s):.1%} of rows", delta_color="off")
        c1.metric("Z-score outliers", f"{z_out:,}", f"{z_out / len(s):.1%} of rows", delta_color="off")
        c1.caption(f"Q1={q1:,.2f}  Q3={q3:,.2f}  IQR={iqr:,.2f}  fences=[{lo:,.2f}, {hi:,.2f}]")
        with c2:
            label = cfg(name).get("clu_label")
            fig = px.box(sample(df, 8000), x=label if label in df else None, y=col, log_y=log, points="outliers",
                         color=label if label in df else None, color_discrete_map=ui.ATTACK_COLORS,
                         title=f"Distribution of {col}" + (f" by {label}" if label in df else ""))
            fig.update_layout(showlegend=False)
            ui.show(fig, 420)

    with tabs[2]:
        ui.theory("<b>Min-max</b>: v' = (v − min)/(max − min) → [0,1] &nbsp;·&nbsp; <b>Z-score</b>: v' = (v − μ)/σ "
                  "&nbsp;·&nbsp; <b>Decimal scaling</b>: v' = v / 10<sup>j</sup> where j = smallest integer with "
                  "max|v'| &lt; 1. Needed for distance-based methods (k-means, kNN).")
        col = st.selectbox("Attribute to normalise", num, index=num.index("dst_bytes") if "dst_bytes" in num else 0)
        s = df[col].astype(float)
        mm = (s - s.min()) / ((s.max() - s.min()) or 1)
        zs = (s - s.mean()) / (s.std() or 1)
        j = int(np.ceil(np.log10(s.abs().max() + 1))) if s.abs().max() > 0 else 0
        ds = s / (10 ** j)
        fig = make_subplots(1, 4, subplot_titles=["original", "min-max", "z-score", f"decimal (j={j})"])
        for i, v in enumerate([s, mm, zs, ds], 1):
            fig.add_trace(go.Histogram(x=v, nbinsx=50, marker_color=ui.PALETTE[i - 1], showlegend=False), 1, i)
        ui.show(fig, 320)
        st.dataframe(pd.DataFrame({"original": s, "min-max": mm, "z-score": zs, "decimal scaling": ds}).head(10).round(4),
                     width="stretch")

    with tabs[3]:
        ui.theory("<b>Discretisation</b> converts continuous values into intervals. <b>Equal-width</b> bins split the "
                  "range evenly; <b>equal-frequency</b> bins hold the same number of values. <b>Smoothing by bin "
                  "means / boundaries</b> replaces each value by its bin's mean / nearest boundary to reduce noise. "
                  "Association-rule mining needs this step because Apriori works on categorical items.")
        c1, c2 = st.columns([1, 2])
        col = c1.selectbox("Attribute", num, index=num.index("count") if "count" in num else 0, key="disc_col")
        k = c1.slider("Number of bins", 2, 10, 4)
        s = df[col]
        ew = pd.cut(s, bins=k)
        ef = pd.qcut(s.rank(method="first"), q=k)
        with c2:
            t = pd.DataFrame({"Equal-width": ew.value_counts(sort=False).values,
                              "Equal-frequency": ef.value_counts(sort=False).values},
                             index=[f"bin {i + 1}" for i in range(k)])
            fig = px.bar(t, barmode="group", title="Records per bin")
            fig.update_layout(xaxis_title=None, yaxis_title="records", legend_title=None)
            ui.show(fig, 320)
        st.markdown("**Smoothing demo** on the first 12 sorted values (equal-frequency, 3 bins)")
        v = np.sort(s.dropna().sample(12, random_state=3).values)
        bins = np.array_split(v, 3)
        rows = []
        for i, b in enumerate(bins):
            for x in b:
                rows.append({"bin": i + 1, "value": x, "by bin mean": round(b.mean(), 2),
                             "by bin boundary": b.min() if abs(x - b.min()) <= abs(x - b.max()) else b.max()})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

    with tabs[4]:
        ui.theory("<b>Correlation analysis</b> finds redundant numeric attributes (|r| close to 1). "
                  "<b>Information gain / mutual information</b> ranks how much each attribute tells us about the "
                  "class — the same criterion J48 uses to choose splits.")
        c1, c2 = st.columns(2)
        with c1:
            corr = df[num].corr().round(2)
            fig = px.imshow(corr, text_auto=".1f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                            title="Pearson correlation", aspect="auto")
            ui.show(fig, 520)
        with c2:
            cats = categorical_cols(df)
            target = st.selectbox("Class attribute", cats, index=cats.index(cfg(name)["class_target"])
                                  if cfg(name)["class_target"] in cats else 0)
            leak = set(cfg(name).get("leak_cols", []))
            feats = [c for c in df.columns if c != target and c not in leak and c not in cfg(name)["id_cols"]
                     and not pd.api.types.is_datetime64_any_dtype(df[c])]
            s = sample(df, 8000)
            X = pd.DataFrame({c: (s[c] if c in num else s[c].astype("category").cat.codes) for c in feats})
            mi = mutual_info_classif(X, s[target], discrete_features=[c not in num for c in feats], random_state=0)
            mi = pd.Series(mi, index=feats).sort_values()
            fig = px.bar(mi.rename_axis("attribute").reset_index(name="mutual information (nats)"),
                         x="mutual information (nats)", y="attribute", orientation="h",
                         title=f"Information gain w.r.t. '{target}'", color_discrete_sequence=["#d9480f"])
            fig.update_layout(yaxis_title=None)
            ui.show(fig, 470)

    with tabs[5]:
        ui.theory("<b>Principal Component Analysis</b> projects standardised attributes onto orthogonal components "
                  "that capture maximum variance — a data-reduction technique that keeps most information in few "
                  "dimensions.")
        s = sample(df, 6000)
        Xs = StandardScaler().fit_transform(s[num])
        pca = PCA().fit(Xs)
        ev = pd.DataFrame({"component": [f"PC{i + 1}" for i in range(len(num))],
                           "explained": pca.explained_variance_ratio_,
                           "cumulative": np.cumsum(pca.explained_variance_ratio_)})
        c1, c2 = st.columns(2)
        with c1:
            fig = go.Figure()
            fig.add_bar(x=ev["component"], y=ev["explained"], name="explained", marker_color="#d9480f")
            fig.add_scatter(x=ev["component"], y=ev["cumulative"], name="cumulative", line=dict(color="#1d1c1a"))
            fig.update_layout(title="Scree plot", yaxis_tickformat=".0%")
            ui.show(fig, 380)
        with c2:
            pcs = pca.transform(Xs)[:, :2]
            label = cfg(name).get("clu_label")
            fig = px.scatter(x=pcs[:, 0], y=pcs[:, 1], color=s[label].values if label in s else None, opacity=.6,
                             color_discrete_map=ui.ATTACK_COLORS, labels={"x": "PC1", "y": "PC2", "color": label},
                             title="Data projected on first two components")
            fig.update_traces(marker_size=4)
            ui.show(fig, 380)
        k95 = int((ev["cumulative"] < .95).sum() + 1)
        st.info(f"**{k95}** of {len(num)} components retain 95% of the variance.")
