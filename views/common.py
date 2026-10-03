"""UI pieces shared by the classification pages."""
import pandas as pd
import plotly.express as px
import streamlit as st

from core import ui
from core.data import categorical_cols, cfg, dataset_picker, get_df, numeric_cols
from core.models import per_class, summary


def class_setup(key):
    name = dataset_picker(f"{key}_ds")
    df = get_df(name)
    c = cfg(name)
    cats = categorical_cols(df)
    tgt_default = c["class_target"] if c["class_target"] in cats else cats[-1]
    target = st.selectbox("Class attribute", cats, index=cats.index(tgt_default), key=f"{key}_tgt")
    opts = [x for x in df.columns if x != target and x not in c["id_cols"]
            and not pd.api.types.is_datetime64_any_dtype(df[x])]
    default = [x for x in c["class_features"] if x in opts] if target == c["class_target"] else \
        [x for x in opts if x not in c.get("leak_cols", [])][:12]
    feats = st.multiselect("Attributes", opts, default=default, key=f"{key}_feats",
                           help="Leakage columns (e.g. attack_type) are excluded by default.")
    mode = st.radio("Test options", ["Percentage split", "10-fold cross-validation"], key=f"{key}_mode",
                    horizontal=True)
    split = st.slider("Training %", 50, 90, 66, key=f"{key}_split") / 100 if mode == "Percentage split" else 0.66
    return name, df, target, feats, ("split" if mode == "Percentage split" else "cv"), split


def input_form(df, feats, key):
    """Renders widgets for a single new instance; returns a one-row DataFrame."""
    num = numeric_cols(df)
    vals = {}
    cols = st.columns(4)
    for i, f in enumerate(feats):
        with cols[i % 4]:
            if f in num:
                vals[f] = st.number_input(f, value=float(df[f].median()), key=f"{key}_{f}")
            else:
                opts = sorted(df[f].astype(str).unique())
                vals[f] = st.selectbox(f, opts, key=f"{key}_{f}")
    return pd.DataFrame([vals])


def show_evaluation(res, title):
    s = summary(res)
    ui.kpis([("Accuracy", f"{s['accuracy']:.2%}", f"{int((res['y_true'] == res['y_pred']).sum()):,} correctly classified", "cyan"),
             ("Kappa", f"{s['kappa']:.3f}", "agreement beyond chance", "violet"),
             ("Weighted F1", f"{s['f1']:.3f}", f"P={s['precision']:.3f} · R={s['recall']:.3f}", "amber"),
             ("ROC area", f"{s['roc_auc']:.3f}", "one-vs-rest, weighted", "red")])
    st.write("")
    pc, cm = per_class(res)
    c1, c2 = st.columns([1, 1.15])
    with c1:
        fig = px.imshow(cm, x=list(res["classes"]), y=list(res["classes"]), text_auto=True,
                        color_continuous_scale=ui.SEQ, labels=dict(x="predicted", y="actual", color="count"),
                        title=f"Confusion matrix · {title}")
        fig.update_layout(coloraxis_showscale=False)
        ui.show(fig, 420)
    with c2:
        st.markdown("**Detailed accuracy by class**")
        st.dataframe(pc.round(3), hide_index=True, width="stretch")
        fig = px.bar(pc, x="class", y="F-measure", color="class", color_discrete_map=ui.ATTACK_COLORS,
                     title="F-measure per class")
        fig.update_layout(showlegend=False, yaxis_range=[0, 1.05])
        ui.show(fig, 240)
    return s
