import numpy as np
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import streamlit as st
from scipy.cluster.hierarchy import linkage
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (adjusted_rand_score, calinski_harabasz_score, davies_bouldin_score,
                             normalized_mutual_info_score, silhouette_score)
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from core import ui
from core.data import DS1, DS2, UPLOAD, available_datasets, cfg, get_df, numeric_cols, sample

STORY = {
    DS1: "Group connections by **behaviour** (rates, error ratios, byte volumes) without using labels, then check "
         "whether the discovered groups line up with real attack categories — unsupervised attack discovery.",
    DS2: "Segment incidents by **impact profile** (users, records, detection & resolution time, loss) to find "
         "natural tiers of incidents for triage and response planning.",
}
MAX_ROWS = {"K-Means": 20000, "Hierarchical (agglomerative)": 4000, "DBSCAN": 8000}


@st.cache_data(show_spinner=False)
def elbow(Xs, kmax=10):
    rows = []
    fit = Xs[np.random.default_rng(0).choice(len(Xs), min(6000, len(Xs)), replace=False)]
    sub = fit[:2500]
    for k in range(2, kmax + 1):
        km = KMeans(k, n_init=4, random_state=42).fit(fit)
        rows.append({"k": k, "SSE (inertia)": km.inertia_,
                     "silhouette": silhouette_score(sub, km.predict(sub))})
    return pd.DataFrame(rows)


def block(name, key):
    df = get_df(name)
    c = cfg(name)
    num = numeric_cols(df)
    st.markdown(STORY.get(name, ""))
    left, right = st.columns([1, 2.6])
    with left:
        algo = st.radio("Algorithm", list(MAX_ROWS), key=f"{key}_alg")
        feats = st.multiselect("Attributes", num, default=[x for x in c["clu_features"] if x in num], key=f"{key}_f")
        log = st.checkbox("log(1+x) skewed attributes", value=True, key=f"{key}_log")
        scaler = st.radio("Scaling", ["z-score", "min-max"], horizontal=True, key=f"{key}_sc")
        if algo == "K-Means":
            k = st.slider("k (clusters)", 2, 10, 5 if name == DS1 else 4, key=f"{key}_k")
        elif algo.startswith("Hier"):
            k = st.slider("Number of clusters (cut height)", 2, 10, 4, key=f"{key}_hk")
            link = st.selectbox("Linkage", ["ward", "complete", "average", "single"], key=f"{key}_lk")
        else:
            eps = st.slider("ε (eps)", 0.05, 3.0, 0.5, 0.05, key=f"{key}_eps")
            mins = st.slider("MinPts", 3, 50, 10, key=f"{key}_mp")
    if len(feats) < 2:
        st.warning("Choose at least two attributes.")
        return

    data = sample(df, MAX_ROWS[algo], seed=1).reset_index(drop=True)
    X = data[feats].astype(float).copy()
    if log:
        for f in feats:
            if X[f].min() >= 0 and X[f].max() > 50:
                X[f] = np.log1p(X[f])
    Xs = (StandardScaler() if scaler == "z-score" else MinMaxScaler()).fit_transform(X)

    if algo == "K-Means":
        model = KMeans(k, n_init=10, random_state=42).fit(Xs)
        labels = model.labels_
    elif algo.startswith("Hier"):
        labels = AgglomerativeClustering(k, linkage=link).fit_predict(Xs)
    else:
        labels = DBSCAN(eps=eps, min_samples=mins).fit_predict(Xs)
    data["cluster"] = np.where(labels == -1, "noise", "C" + pd.Series(labels).astype(str))
    valid = labels != -1
    n_clusters = len(set(labels[valid]))

    truth = c.get("clu_label") if c.get("clu_label") in data else None
    sil = silhouette_score(Xs[valid][:4000], labels[valid][:4000]) if n_clusters > 1 else np.nan
    with right:
        ui.kpis([("Clusters found", n_clusters, f"{algo} · {len(data):,} rows", "cyan"),
                 ("Silhouette", f"{sil:.3f}" if n_clusters > 1 else "—", "−1 … 1, higher = better separated", "violet"),
                 ("Davies-Bouldin", f"{davies_bouldin_score(Xs[valid], labels[valid]):.3f}" if n_clusters > 1 else "—",
                  "lower = better", "amber"),
                 (("Noise points" if algo == "DBSCAN" else "Adjusted Rand vs labels"),
                  (f"{(~valid).sum():,}" if algo == "DBSCAN" else
                   f"{adjusted_rand_score(data[truth], labels):.3f}" if truth else "—"),
                  ("outliers / anomalies" if algo == "DBSCAN" else f"agreement with '{truth}'"), "red")])
        st.write("")
        pcs = PCA(2, random_state=0).fit_transform(Xs)
        view = pd.DataFrame({"PC1": pcs[:, 0], "PC2": pcs[:, 1], "cluster": data["cluster"]})
        if truth:
            view[truth] = data[truth]
        vs = sample(view, 5000)
        c1, c2 = st.columns(2)
        with c1:
            fig = px.scatter(vs, x="PC1", y="PC2", color="cluster", opacity=.6, title="Clusters (PCA projection)",
                             color_discrete_sequence=ui.PALETTE, category_orders={"cluster": sorted(view["cluster"].unique())})
            fig.update_traces(marker_size=4)
            ui.show(fig, 380)
        with c2:
            if truth:
                fig = px.scatter(vs, x="PC1", y="PC2", color=truth, opacity=.6, title=f"Actual '{truth}' (for comparison)",
                                 color_discrete_map=ui.ATTACK_COLORS)
                fig.update_traces(marker_size=4)
                ui.show(fig, 380)

    tabs = st.tabs(["📋 Cluster profiles", "🏷️ Clusters vs actual labels"] +
                   (["📉 Elbow & silhouette"] if algo == "K-Means" else []) +
                   (["🌲 Dendrogram"] if algo.startswith("Hier") else []) +
                   (["📏 k-distance (choose ε)"] if algo == "DBSCAN" else []))
    with tabs[0]:
        prof = data.groupby("cluster")[feats].mean()
        prof.insert(0, "size", data["cluster"].value_counts())
        st.dataframe(prof.round(2), width="stretch")
        z = (X.groupby(data["cluster"]).mean() - X.mean()) / X.std().replace(0, 1)
        fig = px.imshow(z.round(2), text_auto=True, color_continuous_scale="RdBu_r", zmin=-2, zmax=2, aspect="auto",
                        title="Cluster centroid deviation from overall mean (σ units)")
        ui.show(fig, 340)
        notes = []
        for cl, r in z.iterrows():
            hi = r.sort_values(ascending=False).index[:2]
            lo = r.sort_values().index[:1]
            notes.append(f"**{cl}** ({int(prof.loc[cl, 'size']):,}) — high *{hi[0]}*, *{hi[1]}*; low *{lo[0]}*"
                         + (f" → mostly **{data.loc[data['cluster'] == cl, truth].mode()[0]}**" if truth else ""))
        st.markdown("  \n".join(notes))

    with tabs[1]:
        if truth:
            ct = pd.crosstab(data["cluster"], data[truth])
            c1, c2 = st.columns([1.5, 1])
            with c1:
                fig = px.imshow(ct, text_auto=True, color_continuous_scale=ui.SEQ, aspect="auto",
                                title=f"Cluster × {truth}")
                ui.show(fig, 380)
            with c2:
                purity = ct.max(axis=1).sum() / ct.values.sum()
                st.metric("Purity", f"{purity:.2%}")
                st.metric("Normalised mutual information", f"{normalized_mutual_info_score(data[truth], labels):.3f}")
                st.metric("Calinski-Harabasz", f"{calinski_harabasz_score(Xs[valid], labels[valid]):,.0f}"
                          if n_clusters > 1 else "—")
                st.caption("Purity = fraction of points belonging to the majority label of their cluster.")
        else:
            st.info("No label column to compare with.")

    if algo == "K-Means":
        with tabs[2]:
            e = elbow(Xs)
            c1, c2 = st.columns(2)
            with c1:
                fig = px.line(e, x="k", y="SSE (inertia)", markers=True, title="Elbow method",
                              color_discrete_sequence=["#d9480f"])
                fig.add_vline(x=k, line_dash="dash", line_color="#1d1c1a")
                ui.show(fig, 320)
            with c2:
                fig = px.line(e, x="k", y="silhouette", markers=True, title="Silhouette coefficient",
                              color_discrete_sequence=["#1c7ed6"])
                fig.add_vline(x=k, line_dash="dash", line_color="#1d1c1a")
                ui.show(fig, 320)
            st.caption(f"Best silhouette at k = {int(e.loc[e['silhouette'].idxmax(), 'k'])}.")
    elif algo.startswith("Hier"):
        with tabs[2]:
            n = st.slider("Points in dendrogram", 30, 200, 80, key=f"{key}_dn")
            idx = np.random.default_rng(0).choice(len(Xs), n, replace=False)
            lab = (data.loc[idx, truth].astype(str) if truth else pd.Series(idx).astype(str)).tolist()
            fig = ff.create_dendrogram(Xs[idx], labels=lab, linkagefun=lambda x: linkage(x, method=link))
            fig.update_layout(title=f"Dendrogram ({link} linkage, {n} sampled points)", height=460,
                              paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            ui.show(fig)
    else:
        with tabs[2]:
            nn = NearestNeighbors(n_neighbors=mins).fit(Xs)
            d = np.sort(nn.kneighbors(Xs)[0][:, -1])
            fig = px.line(x=np.arange(len(d)), y=d, labels={"x": "points sorted by distance", "y": f"{mins}-NN distance"},
                          title="k-distance graph — pick ε at the knee", color_discrete_sequence=["#d9480f"])
            fig.add_hline(y=eps, line_dash="dash", line_color="#1d1c1a")
            ui.show(fig, 340)
            if (~valid).sum():
                st.markdown("**Noise points = anomalies** (first 50)")
                st.dataframe(data.loc[~valid].head(50), width="stretch", height=260)


def render():
    ui.hero("Clustering", "Clustering on Two Datasets",
            "Unsupervised grouping of attacks by behaviour. K-Means partitions, hierarchical clustering builds a "
            "dendrogram, and DBSCAN finds dense regions while flagging outliers as potential zero-day anomalies.")
    ui.theory("<b>K-Means</b> minimises SSE = ΣΣ‖x − μ<sub>j</sub>‖² by alternately assigning points to the nearest "
              "centroid and recomputing centroids. <b>Agglomerative</b> repeatedly merges the closest clusters "
              "(single / complete / average / Ward linkage). <b>DBSCAN</b> grows clusters from core points having "
              "≥ MinPts neighbours within ε; unreachable points are noise.")
    names = available_datasets()
    tabs = st.tabs([("🌐 " if n == DS1 else "🏢 " if n == DS2 else "⬆️ ") + n for n in names])
    for t, n in zip(tabs, names):
        with t:
            block(n, {DS1: "c1", DS2: "c2", UPLOAD: "cu"}[n])
