import time
from itertools import combinations

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from mlxtend.frequent_patterns import apriori, association_rules, fpgrowth

from core import ui
from core.data import cfg, dataset_picker, discretize, get_df


@st.cache_data(show_spinner=False)
def to_transactions(df: pd.DataFrame, cols: tuple, bins: int):
    """Each record → a transaction of 'attribute=value' items (numerics discretised first)."""
    t = pd.DataFrame(index=df.index)
    labels = ["low", "med", "high", "v.high", "extreme"]
    for c in cols:
        s = df[c]
        if pd.api.types.is_numeric_dtype(s) and s.nunique() > bins:
            t[c] = discretize(s, bins, "quantile", labels)
        else:
            t[c] = s.astype(str)
    onehot = pd.get_dummies(t, prefix_sep="=").astype(bool)
    return t, onehot


def _fmt(items):
    return ", ".join(sorted(items))


@st.cache_data(show_spinner=False)
def mine(onehot: pd.DataFrame, algo: str, min_sup: float, max_len: int):
    t0 = time.perf_counter()
    fn = apriori if algo == "Apriori" else fpgrowth
    fi = fn(onehot, min_support=min_sup, use_colnames=True, max_len=max_len)
    return fi, time.perf_counter() - t0


def rules_from(fi, n, min_conf):
    if fi.empty:
        return pd.DataFrame()
    r = association_rules(fi, num_itemsets=n, metric="confidence", min_threshold=min_conf)
    if r.empty:
        return r
    r["antecedent"] = r["antecedents"].apply(_fmt)
    r["consequent"] = r["consequents"].apply(_fmt)
    r["length"] = r["antecedents"].apply(len) + r["consequents"].apply(len)
    return r


@st.cache_data(show_spinner=False)
def apriori_levels(onehot: pd.DataFrame, min_sup: float, max_k: int = 4):
    """From-scratch Apriori that records |C_k| (candidates) and |L_k| (frequent) at every level."""
    X = onehot.to_numpy()
    n = len(X)
    items = list(onehot.columns)
    col_of = {it: i for i, it in enumerate(items)}
    attr = {it: it.split("=")[0] for it in items}
    sup1 = X.mean(axis=0)
    L = [frozenset([it]) for it, s in zip(items, sup1) if s >= min_sup]
    levels = [dict(k=1, candidates=len(items), frequent=len(L), pruned=0)]
    examples = {1: sorted([(_fmt(i), float(sup1[col_of[next(iter(i))]])) for i in L], key=lambda x: -x[1])[:8]}
    k = 2
    while L and k <= max_k:
        prev = set(L)
        cands = set()
        Ls = sorted(L, key=lambda s: sorted(s))
        for a, b in combinations(Ls, 2):
            u = a | b
            if len(u) != k:
                continue
            if len({attr[i] for i in u}) < k:  # two values of the same attribute can never co-occur
                continue
            cands.add(u)
        before = len(cands)
        cands = {c for c in cands if all(frozenset(s) in prev for s in combinations(c, k - 1))}  # Apriori prune
        pruned = before - len(cands)
        freq = []
        for c in cands:
            idx = [col_of[i] for i in c]
            s = X[:, idx].all(axis=1).sum() / n
            if s >= min_sup:
                freq.append((c, s))
        levels.append(dict(k=k, candidates=before, frequent=len(freq), pruned=pruned))
        examples[k] = sorted([(_fmt(c), float(s)) for c, s in freq], key=lambda x: -x[1])[:8]
        L = [c for c, _ in freq]
        k += 1
    return pd.DataFrame(levels), examples


def render():
    ui.hero("Association Rule Mining", "Attack Signature Rules · Apriori & FP-Growth",
            "Every connection becomes a market-basket transaction of items such as <code>protocol=tcp</code>, "
            "<code>flag=S0</code>, <code>count=high</code>. Frequent co-occurrences expose the fingerprints of each "
            "attack and how they differ across networks.")
    ui.theory("<b>Support</b>(A⇒B) = P(A∪B) &nbsp;·&nbsp; <b>Confidence</b>(A⇒B) = P(B|A) = sup(A∪B)/sup(A) "
              "&nbsp;·&nbsp; <b>Lift</b> = conf / sup(B) (&gt;1 ⇒ positive correlation) &nbsp;·&nbsp; "
              "<b>Apriori property</b>: every subset of a frequent itemset is frequent, so candidates with an "
              "infrequent subset are pruned. <b>FP-Growth</b> avoids candidate generation by compressing the "
              "database into an FP-tree.")

    c1, c2 = st.columns([1, 2.4])
    with c1:
        name = dataset_picker("as_ds")
        df = get_df(name)
        conf = cfg(name)
        opts = [c for c in df.columns if c not in conf["id_cols"] and c not in conf.get("leak_cols", [])
                and not pd.api.types.is_datetime64_any_dtype(df[c])]
        cols = st.multiselect("Attributes (items)", opts, default=[c for c in conf["assoc_cols"] if c in opts])
        bins = st.slider("Bins for numeric attributes", 2, 5, 3, help="Equal-frequency discretisation")
        algo = st.radio("Algorithm", ["FP-Growth", "Apriori"], horizontal=True)
        min_sup = st.slider("Minimum support", 0.01, 0.5, 0.02, 0.01)
        min_conf = st.slider("Minimum confidence", 0.1, 1.0, 0.6, 0.05)
        min_lift = st.slider("Minimum lift", 1.0, 10.0, 1.2, 0.1)
        max_len = st.slider("Max itemset length", 2, 6, 4)
        net_col = "network" if "network" in df.columns else None
        scope = st.selectbox("Mine on", ["All networks"] + (sorted(df[net_col].unique()) if net_col else []))

    if not cols:
        st.warning("Select at least two attributes.")
        return
    work = df if scope == "All networks" else df[df[net_col] == scope]
    trans, onehot = to_transactions(work, tuple(cols), bins)
    with st.spinner(f"Mining {len(onehot):,} transactions × {onehot.shape[1]} items with {algo}…"):
        fi, secs = mine(onehot, algo, min_sup, max_len)
    rules = rules_from(fi, len(onehot), min_conf)
    if not rules.empty:
        rules = rules[rules["lift"] >= min_lift].sort_values(["lift", "confidence"], ascending=False)

    with c2:
        ui.kpis([("Transactions", f"{len(onehot):,}", scope, "cyan"),
                 ("Distinct items", onehot.shape[1], f"{len(cols)} attributes", "violet"),
                 ("Frequent itemsets", f"{len(fi):,}", f"{algo} · {secs:.2f}s", "amber"),
                 ("Strong rules", f"{len(rules):,}", f"conf ≥ {min_conf}, lift ≥ {min_lift}", "red")])
        st.write("")
        t_trans, t_items = st.columns(2)
        with t_trans:
            st.markdown("**Sample transactions**")
            st.dataframe(pd.DataFrame({"TID": trans.index[:8],
                                       "items": trans.head(8).apply(lambda r: "{" + ", ".join(f"{k}={v}" for k, v in r.items()) + "}", axis=1)}),
                         hide_index=True, width="stretch", height=250)
        with t_items:
            top1 = onehot.mean().sort_values(ascending=False).head(12).rename_axis("item").reset_index(name="support")
            fig = px.bar(top1, x="support", y="item", orientation="h", title="Most frequent items",
                         color_discrete_sequence=["#d9480f"])
            fig.update_layout(yaxis=dict(autorange="reversed", title=None))
            ui.show(fig, 260)

    tabs = st.tabs(["🎯 Attack signature rules", "📋 All rules", "📊 Rule visualisation", "🪜 Apriori step-by-step",
                    "🌐 Compare across networks", "⚡ Apriori vs FP-Growth"])

    target_prefix = conf.get("assoc_target_prefix") or ""
    with tabs[0]:
        if rules.empty:
            st.info("No rules at these thresholds — lower support or confidence.")
        else:
            prefix = st.text_input("Consequent must start with", target_prefix,
                                   help="e.g. attack_category=  or  severity=  (leave empty for all)")
            sig = rules[(rules["consequents"].apply(len) == 1) & rules["consequent"].str.startswith(prefix)]
            sig = sig[~sig["antecedent"].str.contains(prefix, regex=False)] if prefix else sig
            st.caption(f"{len(sig)} rules predict **{prefix or 'any item'}** — the top ones by lift:")
            best = sig.sort_values(["consequent", "lift"], ascending=[True, False]).groupby("consequent").head(4)
            for _, r in best.sort_values("lift", ascending=False).head(18).iterrows():
                ui.rule_card(r["antecedent"], r["consequent"],
                             f"sup={r['support']:.3f} · conf={r['confidence']:.2f} · lift={r['lift']:.2f}")

    with tabs[1]:
        if not rules.empty:
            show = rules[["antecedent", "consequent", "support", "confidence", "lift", "leverage", "conviction",
                          "length"]].round(4)
            st.dataframe(show, hide_index=True, width="stretch", height=460)
            st.download_button("⬇️ Download rules CSV", show.to_csv(index=False).encode(), "association_rules.csv")
        st.markdown("**Frequent itemsets**")
        fi2 = fi.assign(itemset=fi["itemsets"].apply(_fmt), length=fi["itemsets"].apply(len)) \
            .sort_values("support", ascending=False)[["itemset", "length", "support"]]
        st.dataframe(fi2.round(4), hide_index=True, width="stretch", height=300)

    with tabs[2]:
        if not rules.empty:
            c1, c2 = st.columns(2)
            with c1:
                fig = px.scatter(rules, x="support", y="confidence", color="lift", size="length",
                                 hover_data=["antecedent", "consequent"], color_continuous_scale=ui.SEQ,
                                 title="Support vs confidence (colour = lift)")
                ui.show(fig, 440)
            with c2:
                top = rules.head(25)
                m = top.pivot_table(index="antecedent", columns="consequent", values="lift", aggfunc="max")
                fig = px.imshow(m, color_continuous_scale=ui.SEQ, aspect="auto", title="Lift matrix · top 25 rules")
                fig.update_xaxes(tickangle=30)
                ui.show(fig, 440)
            lens = fi["itemsets"].apply(len).value_counts().sort_index()
            fig = px.bar(x=lens.index.astype(str), y=lens.values, labels={"x": "itemset size k", "y": "frequent itemsets"},
                         title="Frequent itemsets by size", color_discrete_sequence=["#1c7ed6"])
            ui.show(fig, 300)

    with tabs[3]:
        ui.theory("Level-wise search: <b>C<sub>k</sub></b> = candidates generated by joining L<sub>k−1</sub> with "
                  "itself, then <b>pruned</b> if any (k−1)-subset is infrequent; <b>L<sub>k</sub></b> = candidates "
                  "whose support ≥ min_support after one database scan. Implemented from scratch below.")
        lv, ex = apriori_levels(onehot, min_sup, min(max_len, 4))
        c1, c2 = st.columns([1, 1.3])
        with c1:
            st.dataframe(lv.rename(columns={"k": "level k", "candidates": "|C_k| generated",
                                            "pruned": "pruned by Apriori property", "frequent": "|L_k| frequent"}),
                         hide_index=True, width="stretch")
            fig = px.bar(lv.melt(id_vars="k", value_vars=["candidates", "frequent"]), x="k", y="value",
                         color="variable", barmode="group", title="Candidates vs frequent itemsets per level",
                         log_y=True)
            fig.update_layout(legend_title=None)
            ui.show(fig, 320)
        with c2:
            for k, items in ex.items():
                if items:
                    st.markdown(f"**L{k}** – top frequent {k}-itemsets")
                    st.dataframe(pd.DataFrame(items, columns=["itemset", "support"]).round(4),
                                 hide_index=True, width="stretch")

    with tabs[4]:
        if not net_col:
            st.info("This dataset has no `network` column.")
        else:
            st.markdown("Rules are mined **separately inside each network** and their confidence compared — the same "
                        "attack signature can be strong on one network and absent on another.")
            per = []
            for net in sorted(df[net_col].unique()):
                sub = df[df[net_col] == net]
                _, oh = to_transactions(sub, tuple(c for c in cols if c != net_col), bins)
                f, _ = mine(oh, "FP-Growth", min_sup, 3)
                r = rules_from(f, len(oh), min_conf)
                if r.empty:
                    continue
                r = r[(r["consequents"].apply(len) == 1) & r["consequent"].str.startswith(target_prefix)]
                if target_prefix:
                    r = r[~r["antecedent"].str.contains(target_prefix, regex=False)]
                r = r.assign(network=net, rule=r["antecedent"] + " ⇒ " + r["consequent"])
                per.append(r[["network", "rule", "support", "confidence", "lift"]])
            if per:
                allr = pd.concat(per)
                counts = allr.groupby("network").size().reset_index(name="rules")
                c1, c2 = st.columns([1, 2.2])
                with c1:
                    ui.show(px.bar(counts, x="network", y="rules", title="Strong rules per network",
                                   color_discrete_sequence=["#d9480f"]), 420)
                with c2:
                    top_rules = allr.sort_values("lift", ascending=False).drop_duplicates("rule").head(15)["rule"]
                    m = allr[allr["rule"].isin(top_rules)].pivot_table(index="rule", columns="network",
                                                                      values="confidence")
                    fig = px.imshow(m.round(2), text_auto=True, color_continuous_scale=ui.SEQ, aspect="auto",
                                    title="Confidence of top rules in each network (blank = rule not found)")
                    fig.update_layout(yaxis_title=None, xaxis_title=None)
                    ui.show(fig, 520)
                uniq = allr.groupby("rule")["network"].nunique()
                only = allr[allr["rule"].isin(uniq[uniq == 1].index)].sort_values("lift", ascending=False) \
                    .drop_duplicates("network")
                st.markdown("**Network-specific signatures** (rule found in only one network)")
                for _, r in only.iterrows():
                    a, b = r["rule"].split(" ⇒ ")
                    ui.rule_card(a, b, f"· only in {r['network']} · conf={r['confidence']:.2f} · lift={r['lift']:.2f}")
            else:
                st.info("No rules found per network at these thresholds.")

    with tabs[5]:
        st.markdown("Both algorithms return identical frequent itemsets; they differ in how they find them.")
        if st.button("Run benchmark", type="primary"):
            res = []
            for sup in [0.2, 0.1, 0.05, 0.03]:
                for a in ["Apriori", "FP-Growth"]:
                    t0 = time.perf_counter()
                    f = (apriori if a == "Apriori" else fpgrowth)(onehot, min_support=sup, use_colnames=True, max_len=max_len)
                    res.append({"min_support": sup, "algorithm": a, "seconds": time.perf_counter() - t0,
                                "itemsets": len(f)})
            res = pd.DataFrame(res)
            fig = px.line(res, x="min_support", y="seconds", color="algorithm", markers=True,
                          title="Run-time vs minimum support", color_discrete_sequence=["#c92a2a", "#d9480f"])
            fig.update_xaxes(autorange="reversed")
            ui.show(fig, 360)
            st.dataframe(res.round(4), hide_index=True, width="stretch")
        st.markdown("| | Apriori | FP-Growth |\n|---|---|---|\n"
                    "| Strategy | Breadth-first, generate & test candidates | Divide & conquer on an FP-tree |\n"
                    "| DB scans | One per level k | Two |\n| Candidate generation | Yes (can explode) | No |\n"
                    "| Memory | Candidate sets | Compressed prefix tree |")
