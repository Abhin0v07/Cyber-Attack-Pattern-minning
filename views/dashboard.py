import pandas as pd
import plotly.express as px
import streamlit as st

from core import nav, ui
from core.data import DS1, DS2, load_clean

EXPERIMENTS = [
    ("EXP 01", "association-rules", "Association rules on a large dataset", "Apriori & FP-Growth over 30,000 connections"),
    ("EXP 02", "j48", "Classification with J48", "C4.5 entropy tree, IF-THEN rules, gain ratio"),
    ("EXP 03", "naive-bayes", "Classification with Naive Bayes", "Priors, likelihood tables, worked posterior"),
    ("EXP 04", "regression", "Regression on two datasets", "Traffic volume & incident financial loss"),
    ("EXP 05", "clustering", "Clustering on two datasets", "K-Means, hierarchical, DBSCAN"),
    ("DWM", "warehouse", "Data warehouse & OLAP", "Star schema, roll-up, drill-down, slice, dice, pivot"),
    ("DWM", "preprocessing", "Preprocessing", "Cleaning, normalisation, binning, PCA"),
    ("EVAL", "model-comparison", "Model comparison", "J48 vs Naive Bayes vs 4 other classifiers"),
]


def render():
    t = load_clean(DS1)
    inc = load_clean(DS2)
    attacks = t[t["attack_category"] != "Normal"]
    share = len(attacks) / len(t)

    ui.hero("Threat pattern report · August 2026 capture",
            "Cyberattack Pattern Mining Across Networks",
            "Data warehousing and data mining applied to traffic from five networks and 5,000 security incidents: "
            "which attacks hit which network, when they happen, what gives them away and what they cost.")
    st.markdown(f'<p class="lead"><b>{share:.1%}</b> of the {len(t):,} connections seen across five networks were '
                f'malicious, and the attack mix differs sharply from one network to the next.</p>',
                unsafe_allow_html=True)
    ui.kpis([
        ("Connections", f"{len(t):,}", "DS1, cleaned", ""),
        ("Malicious", f"{len(attacks):,}", f"{share:.1%} of traffic", "red"),
        ("Networks", t["network"].nunique(), f"{t['attack_type'].nunique() - 1} attack signatures", ""),
        ("Incidents", f"{len(inc):,}", "DS2, 2019 – 2025", ""),
        ("Incident loss", f"${inc['financial_loss_musd'].sum():,.0f}M", "total, all incidents", "red"),
    ])

    ui.section("01", "Where attacks land", "share of each network's traffic")
    c1, c2 = st.columns([1.7, 1])
    with c1:
        mix = (pd.crosstab(t["network"], t["attack_category"], normalize="index") * 100).round(1)
        order = mix["Normal"].sort_values().index.tolist()
        long = mix.reset_index().melt(id_vars="network", var_name="category", value_name="percent")
        fig = px.bar(long, y="network", x="percent", color="category", orientation="h",
                     color_discrete_map=ui.ATTACK_COLORS, category_orders={"network": order,
                     "category": ["Normal", "DoS", "Probe", "R2L", "U2R", "Botnet"]})
        fig.update_layout(xaxis_title="% of connections", yaxis_title=None, bargap=.35)
        ui.show(fig, 340)
    with c2:
        top = attacks.groupby("network")["attack_category"].agg(lambda s: s.value_counts().index[0])
        rate = (t.assign(a=t["attack_category"] != "Normal").groupby("network")["a"].mean() * 100).round(1)
        tbl = pd.DataFrame({"attack %": rate, "dominant attack": top}).sort_values("attack %", ascending=False)
        st.dataframe(tbl, width="stretch", height=230)
        st.caption("Each network has its own threat profile, so rules and models are compared per network on the "
                   "Association Rules page.")

    ui.section("02", "When they happen", "attack connections by day and hour (UTC)")
    c1, c2 = st.columns([1.7, 1])
    with c1:
        grid = pd.crosstab(attacks["timestamp"].dt.hour, attacks["timestamp"].dt.day)
        fig = px.imshow(grid, aspect="auto", color_continuous_scale=ui.SEQ,
                        labels=dict(x="day of August", y="hour", color="attacks"))
        fig.update_layout(coloraxis_showscale=False)
        ui.show(fig, 330)
    with c2:
        src = attacks["src_country"].value_counts().head(8).sort_values()
        fig = px.bar(x=src.values, y=src.index, orientation="h", labels={"x": "attack connections", "y": ""},
                     title="Top sources", color_discrete_sequence=[ui.INK])
        ui.show(fig, 330)
    st.caption("The bright band on days 12–14 is a DoS flood campaign. Most attacks arrive in the evening and at night.")

    ui.section("03", "What it costs", "DS2 incidents, US$ million")
    c1, c2 = st.columns([1, 1.4])
    with c1:
        loss = inc.groupby("attack_vector")["financial_loss_musd"].sum().sort_values()
        fig = px.bar(x=loss.values, y=loss.index, orientation="h", labels={"x": "total loss ($M)", "y": ""},
                     color_discrete_sequence=[ui.ACCENT])
        ui.show(fig, 360)
    with c2:
        m = inc.pivot_table(index="sector", columns="attack_vector", values="financial_loss_musd", aggfunc="mean")
        fig = px.imshow(m.round(1), text_auto=True, aspect="auto", color_continuous_scale=ui.SEQ,
                        labels=dict(color="avg $M"))
        fig.update_layout(coloraxis_showscale=False, xaxis_title=None, yaxis_title=None)
        fig.update_xaxes(tickangle=35)
        ui.show(fig, 360)

    ui.section("04", "Findings")
    peak = attacks["timestamp"].dt.hour.value_counts().idxmax()
    bot_net = attacks.loc[attacks["attack_category"] == "Botnet", "network"].value_counts().idxmax()
    r2l_net = attacks.loc[attacks["attack_category"] == "R2L", "network"].value_counts().idxmax()
    costly = inc.groupby("sector")["financial_loss_musd"].mean().idxmax()
    items = [
        f"<b>{rate.idxmax()}</b> carries the highest share of malicious traffic ({rate.max():.0f}%), and most of it is DoS.",
        f"Botnet traffic is concentrated in <b>{bot_net}</b>, where IoT devices speak telnet and MQTT.",
        f"Credential attacks (R2L) are aimed mainly at <b>{r2l_net}</b>.",
        f"Attack volume peaks around <b>{peak:02d}:00 UTC</b>, outside business hours; normal traffic peaks at midday.",
        f"<b>{costly}</b> has the highest average loss per incident, and ransomware is the costliest attack vector.",
    ]
    st.markdown('<ol class="find">' + "".join(f"<li><span>{i}</span></li>" for i in items) + "</ol>",
                unsafe_allow_html=True)

    ui.section("05", "Experiment index")
    cols = st.columns(2)
    for i, (code, path, title, desc) in enumerate(EXPERIMENTS):
        with cols[i % 2]:
            p = nav.PAGES.get(path)
            if p is not None:
                st.page_link(p, label=f"{code} · {title}", icon=":material/arrow_forward:")
            st.caption(desc)
