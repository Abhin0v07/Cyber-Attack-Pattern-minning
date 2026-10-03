import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.tree import DecisionTreeClassifier

from core import ui
from core.data import encode_features
from core.models import _cond, evaluate, gain_table, tree_dot, tree_rules
from views.common import class_setup, input_form, show_evaluation


@st.cache_data(show_spinner=False)
def train(df, target, feats, mode, split, min_leaf, max_depth, alpha):
    X = encode_features(df, feats)
    y = df[target].astype(str)
    model = DecisionTreeClassifier(criterion="entropy", min_samples_leaf=min_leaf,
                                   max_depth=max_depth or None, ccp_alpha=alpha, random_state=42)
    res = evaluate(model, X, y, mode=mode, split=split)
    return res, list(X.columns)


def render():
    ui.hero("Classification · J48", "J48 (C4.5) Decision Tree",
            "J48 is Weka's implementation of Quinlan's C4.5: a tree grown top-down by choosing the attribute with the "
            "highest information-gain ratio at each node, then pruned to avoid over-fitting.")
    ui.theory("<b>Info(D)</b> = −Σ p<sub>i</sub> log<sub>2</sub> p<sub>i</sub> &nbsp;·&nbsp; "
              "<b>Gain(A)</b> = Info(D) − Σ |D<sub>j</sub>|/|D| · Info(D<sub>j</sub>) &nbsp;·&nbsp; "
              "<b>SplitInfo(A)</b> = −Σ |D<sub>j</sub>|/|D| log<sub>2</sub>(|D<sub>j</sub>|/|D|) &nbsp;·&nbsp; "
              "<b>GainRatio</b> = Gain / SplitInfo. Weka parameters: <code>-C</code> confidence factor (pruning), "
              "<code>-M</code> minimum instances per leaf.")

    left, right = st.columns([1, 2.6])
    with left:
        name, df, target, feats, mode, split = class_setup("j48")
        st.markdown("**J48 parameters**")
        min_leaf = st.slider("-M  min instances per leaf", 1, 100, 2)
        max_depth = st.slider("Max depth (0 = unlimited)", 0, 20, 0)
        alpha = st.select_slider("Pruning strength (cost-complexity α)", [0.0, 0.0001, 0.0005, 0.001, 0.002, 0.005, 0.01],
                                 value=0.0005, help="Higher α ≈ lower Weka confidence factor → smaller tree")
    if not feats:
        st.warning("Choose at least one attribute.")
        return
    with st.spinner("Growing the tree…"):
        res, cols = train(df, target, tuple(feats), mode, split, min_leaf, max_depth, alpha)
    tree = res["model"]
    classes = list(res["classes"])

    with right:
        leaves = tree.get_n_leaves()
        st.markdown(f"**=== Run information ===**  ·  dataset **{name}**  ·  {len(df):,} instances · "
                    f"{len(feats)} attributes · test mode: "
                    f"{'10-fold cross-validation' if mode == 'cv' else f'{split:.0%} train / {1 - split:.0%} test split'}  \n"
                    f"Number of leaves: **{leaves}** · Size of the tree: **{tree.tree_.node_count}** · "
                    f"Depth: **{tree.get_depth()}**")
        show_evaluation(res, "J48")

    tabs = st.tabs(["🌳 Tree", "📜 Classification rules", "⚖️ Gain ratio (split selection)", "📊 Attribute importance",
                    "🔮 Classify a new connection"])
    with tabs[0]:
        depth = st.slider("Display depth", 2, 8, 4)
        st.graphviz_chart(tree_dot(tree, cols, classes, depth), width="stretch")
        st.caption("Leaves show the predicted class, number of training instances and purity. "
                   "'…' marks a subtree collapsed for display.")

    with tabs[1]:
        rules = tree_rules(tree, cols, classes)
        st.markdown(f"The tree converts into **{len(rules)} IF-THEN rules** (one per leaf). Largest rules first:")
        f = st.multiselect("Show rules predicting", classes, default=classes)
        shown = [r for r in rules if r["cls"] in f][:30]
        for r in shown:
            ui.rule_card(" AND ".join(r["conditions"]) or "TRUE", f"class = {r['cls']}",
                         f"· covers {r['samples']:,} · conf {r['confidence']:.0%}")
        txt = "\n".join(f"IF {' AND '.join(r['conditions'])} THEN {target} = {r['cls']}  "
                        f"(covers {r['samples']}, conf {r['confidence']:.2f})" for r in rules)
        st.download_button("⬇️ Download all rules", txt.encode(), "j48_rules.txt")

    with tabs[2]:
        st.markdown("C4.5 evaluates every attribute at the root and picks the one with the highest **gain ratio** "
                    "(numeric attributes are shown discretised into quartiles here).")
        gt = gain_table(df, target, feats)
        c1, c2 = st.columns([1.4, 1])
        with c1:
            st.dataframe(gt.round(4), hide_index=True, width="stretch", height=420)
        with c2:
            fig = px.bar(gt.sort_values("GainRatio"), x=["Gain", "GainRatio"], y="attribute", orientation="h",
                         barmode="group", title="Information gain vs gain ratio")
            fig.update_layout(yaxis_title=None, legend_title=None, xaxis_title=None)
            ui.show(fig, 420)
        best = gt.iloc[0]
        st.success(f"Root split candidate: **{best['attribute']}** — gain ratio {best['GainRatio']:.3f} "
                   f"(gain {best['Gain']:.3f} bits of Info(D) = {best['Info(D)']:.3f}).")

    with tabs[3]:
        imp = pd.Series(tree.feature_importances_, index=cols)
        grouped = imp.groupby([c.split("=")[0] for c in cols]).sum().sort_values()
        fig = px.bar(grouped.rename_axis("attribute").reset_index(name="importance"), x="importance", y="attribute",
                     orientation="h", title="Total entropy reduction contributed by each attribute",
                     color_discrete_sequence=["#d9480f"])
        fig.update_layout(yaxis_title=None)
        ui.show(fig, 440)

    with tabs[4]:
        st.markdown("Enter the attributes of a connection / incident and the trained tree classifies it.")
        row = input_form(df, feats, "j48p")
        X1 = encode_features(pd.concat([df[feats], row], ignore_index=True), feats).tail(1).reindex(columns=cols, fill_value=0)
        p = tree.predict_proba(X1)[0]
        k = int(np.argmax(p))
        path = tree.decision_path(X1).indices
        c1, c2 = st.columns([1, 1.4])
        c1.metric("Predicted class", classes[k], f"{p[k]:.0%} of training instances at this leaf", delta_color="off")
        conds = []
        t = tree.tree_
        for node in path[:-1]:
            f = cols[t.feature[node]]
            left = X1.iloc[0][f] <= t.threshold[node]
            conds.append(_cond(f, t.threshold[node], left))
        c1.markdown("**Path followed:**  \n" + "  \n".join(f"→ {c}" for c in conds))
        fig = px.bar(x=classes, y=p, color=classes, color_discrete_map=ui.ATTACK_COLORS, title="Class distribution at leaf",
                     labels={"x": "", "y": "probability"})
        fig.update_layout(showlegend=False)
        with c2:
            ui.show(fig, 300)
