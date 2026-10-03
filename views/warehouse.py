import pandas as pd
import plotly.express as px
import streamlit as st

from core import ui
from core.data import DS1, load_clean

REGION = {"India": "Asia", "China": "Asia", "North Korea": "Asia", "Iran": "Middle East", "Russia": "Europe",
          "Germany": "Europe", "Netherlands": "Europe", "USA": "Americas", "Brazil": "Americas",
          "Tor-Exit": "Anonymous", "Unknown": "Unknown"}
ZONE = {"Corporate-LAN": ("Enterprise", "High"), "University-Campus": ("Academic", "Medium"),
        "Cloud-DC": ("Cloud", "High"), "IoT-Grid": ("Operational Tech", "Critical"),
        "Banking-Core": ("Financial", "Critical")}
STAGE = {"Normal": "—", "Probe": "Reconnaissance", "DoS": "Impact", "R2L": "Initial Access",
         "U2R": "Privilege Escalation", "Botnet": "Command & Control"}

SCHEMA = """
digraph G {
  graph [rankdir=LR, bgcolor="transparent", pad=0.3, nodesep=0.35, ranksep=0.9];
  node  [shape=plaintext, fontname="Helvetica", fontcolor="#1d1c1a"];
  edge  [color="#6b6760", arrowhead=none, penwidth=1.4];
  fact [label=<<table border="1" cellborder="0" cellspacing="0" cellpadding="5" color="#1d1c1a" bgcolor="#fffdf9">
    <tr><td bgcolor="#d9480f"><font color="#ffffff"><b>FACT_CONNECTION</b></font></td></tr>
    <tr><td align="left">time_key (FK)</td></tr><tr><td align="left">network_key (FK)</td></tr>
    <tr><td align="left">source_key (FK)</td></tr><tr><td align="left">service_key (FK)</td></tr>
    <tr><td align="left">attack_key (FK)</td></tr>
    <tr><td align="left"><font color="#d9480f">duration, src_bytes, dst_bytes</font></td></tr>
    <tr><td align="left"><font color="#d9480f">packets, failed_logins, is_attack</font></td></tr></table>>];
  t [label=<<table border="1" cellborder="0" cellspacing="0" cellpadding="4" color="#1d1c1a" bgcolor="#fffdf9">
    <tr><td bgcolor="#1d1c1a"><font color="#ffffff"><b>DIM_TIME</b></font></td></tr><tr><td align="left">time_key · date · hour</td></tr>
    <tr><td align="left">day_of_week · week · month</td></tr><tr><td align="left">is_weekend · time_of_day</td></tr></table>>];
  n [label=<<table border="1" cellborder="0" cellspacing="0" cellpadding="4" color="#1d1c1a" bgcolor="#fffdf9">
    <tr><td bgcolor="#1d1c1a"><font color="#ffffff"><b>DIM_NETWORK</b></font></td></tr><tr><td align="left">network_key · network</td></tr>
    <tr><td align="left">zone · criticality</td></tr></table>>];
  s [label=<<table border="1" cellborder="0" cellspacing="0" cellpadding="4" color="#1d1c1a" bgcolor="#fffdf9">
    <tr><td bgcolor="#1d1c1a"><font color="#ffffff"><b>DIM_SOURCE</b></font></td></tr><tr><td align="left">source_key · src_country</td></tr>
    <tr><td align="left">region</td></tr></table>>];
  v [label=<<table border="1" cellborder="0" cellspacing="0" cellpadding="4" color="#1d1c1a" bgcolor="#fffdf9">
    <tr><td bgcolor="#1d1c1a"><font color="#ffffff"><b>DIM_SERVICE</b></font></td></tr><tr><td align="left">service_key · service</td></tr>
    <tr><td align="left">protocol · flag</td></tr></table>>];
  a [label=<<table border="1" cellborder="0" cellspacing="0" cellpadding="4" color="#1d1c1a" bgcolor="#fffdf9">
    <tr><td bgcolor="#1d1c1a"><font color="#ffffff"><b>DIM_ATTACK</b></font></td></tr><tr><td align="left">attack_key · attack_type</td></tr>
    <tr><td align="left">attack_category · kill_chain_stage</td></tr></table>>];
  t -> fact; n -> fact; fact -> s; fact -> v; a -> fact;
}
"""


@st.cache_data(show_spinner=False)
def build_star():
    df = load_clean(DS1).copy()
    ts = df["timestamp"]
    df["date"] = ts.dt.date.astype(str)
    df["hour"] = ts.dt.hour
    df["day_of_week"] = ts.dt.day_name()
    df["week"] = "W" + ts.dt.isocalendar().week.astype(str)
    df["month"] = ts.dt.strftime("%b %Y")
    df["is_weekend"] = ts.dt.dayofweek >= 5
    df["time_of_day"] = pd.cut(df["hour"], [-1, 5, 11, 17, 23], labels=["Night", "Morning", "Afternoon", "Evening"]).astype(str)
    df["region"] = df["src_country"].map(REGION).fillna("Unknown")
    df["zone"] = df["network"].map(lambda n: ZONE[n][0])
    df["criticality"] = df["network"].map(lambda n: ZONE[n][1])
    df["kill_chain_stage"] = df["attack_category"].map(STAGE)
    df["is_attack"] = (df["attack_category"] != "Normal").astype(int)

    def dim(cols, key):
        d = df[cols].drop_duplicates().sort_values(cols).reset_index(drop=True)
        d.insert(0, key, range(1, len(d) + 1))
        return d

    dims = {
        "DIM_TIME": dim(["date", "hour", "day_of_week", "week", "month", "is_weekend", "time_of_day"], "time_key"),
        "DIM_NETWORK": dim(["network", "zone", "criticality"], "network_key"),
        "DIM_SOURCE": dim(["src_country", "region"], "source_key"),
        "DIM_SERVICE": dim(["service", "protocol", "flag"], "service_key"),
        "DIM_ATTACK": dim(["attack_type", "attack_category", "kill_chain_stage"], "attack_key"),
    }
    fact = df[["conn_id"]].copy()
    for name, d in dims.items():
        key = d.columns[0]
        cols = list(d.columns[1:])
        fact[key] = df[cols].merge(d, on=cols, how="left")[key].values
    for m in ["duration", "src_bytes", "dst_bytes", "packets", "failed_logins", "is_attack"]:
        fact[m] = df[m].values
    return df, fact, dims


MEASURES = {"Connections (count)": ("conn_id", "count"), "Attacks (sum is_attack)": ("is_attack", "sum"),
            "Total packets": ("packets", "sum"), "Total src_bytes": ("src_bytes", "sum"),
            "Avg duration (s)": ("duration", "mean"), "Failed logins": ("failed_logins", "sum")}


def agg(df, by, measure):
    col, fn = MEASURES[measure]
    return df.groupby(by, observed=True)[col].agg(fn).reset_index(name=measure)


def render():
    ui.hero("Data Warehousing", "Data Warehouse & OLAP Cube",
            "Raw connection logs are ETL-ed into a star schema; the cube is then explored with roll-up, drill-down, "
            "slice, dice and pivot operations.")
    ui.theory("A <b>data warehouse</b> is a subject-oriented, integrated, time-variant and non-volatile collection of "
              "data supporting decisions. Here the subject is <b>network attacks</b>: a central <b>fact table</b> holds "
              "measurable events (one row per connection) and <b>dimension tables</b> describe the context "
              "(when, where, from whom, which service, what attack).")
    wide, fact, dims = build_star()

    tabs = st.tabs(["⭐ Star schema & ETL", "⬆️ Roll-up", "⬇️ Drill-down", "🔪 Slice", "🎲 Dice", "🔄 Pivot"])

    with tabs[0]:
        c1, c2 = st.columns([1.4, 1])
        with c1:
            st.graphviz_chart(SCHEMA, width="stretch")
        with c2:
            st.markdown("**ETL process**")
            st.markdown(
                "1. **Extract** – raw connection logs from 5 network sensors (CSV)\n"
                "2. **Transform** – deduplicate, impute missing bytes/duration, derive hour, week, time-of-day, "
                "region, network zone, kill-chain stage and `is_attack` flag\n"
                "3. **Load** – surrogate keys generated for each dimension, fact rows reference them\n")
            st.markdown("**Schema variants**")
            st.markdown("- **Star** (used here): denormalised dimensions, fastest joins\n"
                        "- **Snowflake**: DIM_SOURCE → DIM_REGION normalised out\n"
                        "- **Fact constellation**: FACT_CONNECTION + FACT_INCIDENT (DS2) sharing DIM_NETWORK")
            sizes = pd.DataFrame({"table": ["FACT_CONNECTION"] + list(dims),
                                  "rows": [len(fact)] + [len(d) for d in dims.values()]})
            st.dataframe(sizes, hide_index=True, width="stretch")
        pick = st.selectbox("Preview table", ["FACT_CONNECTION"] + list(dims))
        st.dataframe((fact if pick == "FACT_CONNECTION" else dims[pick]).head(100), width="stretch", height=260)

    with tabs[1]:
        ui.theory("<b>Roll-up</b> aggregates data by climbing a concept hierarchy (hour → time-of-day → day → week) "
                  "or by dimension reduction.")
        c1, c2, c3 = st.columns(3)
        hier = c1.selectbox("Hierarchy", ["Time: hour → day → week", "Location: country → region"])
        measure = c3.selectbox("Measure", list(MEASURES), index=1, key="ru_m")
        if hier.startswith("Time"):
            level = c2.select_slider("Level", ["hour", "time_of_day", "date", "week"], value="date")
        else:
            level = c2.select_slider("Level", ["src_country", "region"], value="region")
        d = agg(wide, [level, "attack_category"], measure)
        fig = px.bar(d, x=level, y=measure, color="attack_category", color_discrete_map=ui.ATTACK_COLORS,
                     title=f"{measure} rolled up to '{level}'")
        fig.update_layout(legend_title=None)
        ui.show(fig, 420)

    with tabs[2]:
        ui.theory("<b>Drill-down</b> is the reverse of roll-up: navigate from summarised to more detailed data "
                  "(attack category → attack type → service).")
        cat = st.selectbox("Drill into attack category", [c for c in wide["attack_category"].unique() if c != "Normal"])
        sub = wide[wide["attack_category"] == cat]
        c1, c2 = st.columns(2)
        with c1:
            d = sub.groupby("attack_type").size().reset_index(name="connections")
            ui.show(px.bar(d, x="attack_type", y="connections", title=f"{cat} → attack types",
                           color_discrete_sequence=[ui.ATTACK_COLORS.get(cat, "#d9480f")]), 360)
        with c2:
            fig = px.sunburst(sub, path=["attack_category", "attack_type", "service"],
                              title=f"{cat} → type → service", color_discrete_sequence=ui.PALETTE)
            ui.show(fig, 360)

    with tabs[3]:
        ui.theory("<b>Slice</b> fixes a single value on one dimension, producing a sub-cube "
                  "(e.g. <code>network = IoT-Grid</code>).")
        net = st.selectbox("Slice on network", sorted(wide["network"].unique()), index=3)
        sub = wide[wide["network"] == net]
        cube = pd.crosstab(sub["hour"], sub["attack_category"])
        fig = px.imshow(cube.T, aspect="auto", color_continuous_scale=ui.SEQ,
                        title=f"Slice network='{net}': hour × attack category (connections)")
        ui.show(fig, 360)
        a, b, c = st.columns(3)
        a.metric("Connections in slice", f"{len(sub):,}")
        b.metric("Attack share", f"{sub['is_attack'].mean():.1%}")
        c.metric("Dominant attack", sub[sub["is_attack"] == 1]["attack_category"].mode()[0])

    with tabs[4]:
        ui.theory("<b>Dice</b> selects a sub-cube by specifying ranges / sets on two or more dimensions.")
        c1, c2, c3 = st.columns(3)
        nets = c1.multiselect("network", sorted(wide["network"].unique()), default=["Cloud-DC", "Banking-Core"])
        protos = c2.multiselect("protocol", sorted(wide["protocol"].unique()), default=["tcp"])
        tod = c3.multiselect("time_of_day", ["Night", "Morning", "Afternoon", "Evening"], default=["Night", "Evening"])
        sub = wide[wide["network"].isin(nets) & wide["protocol"].isin(protos) & wide["time_of_day"].isin(tod)]
        st.caption(f"Sub-cube contains **{len(sub):,}** connections")
        if len(sub):
            d = sub.groupby(["network", "time_of_day", "attack_category"]).size().reset_index(name="connections")
            fig = px.bar(d, x="network", y="connections", color="attack_category", facet_col="time_of_day",
                         color_discrete_map=ui.ATTACK_COLORS, title="Diced cube")
            fig.update_layout(legend_title=None)
            ui.show(fig, 400)

    with tabs[5]:
        ui.theory("<b>Pivot</b> (rotate) re-orients the cube to present an alternative view — swap which dimensions "
                  "are on rows and columns.")
        dims_opts = ["network", "attack_category", "attack_type", "protocol", "service", "region", "time_of_day",
                     "day_of_week", "zone", "criticality", "kill_chain_stage"]
        c1, c2, c3, c4 = st.columns(4)
        r = c1.selectbox("Rows", dims_opts, index=0)
        c = c2.selectbox("Columns", dims_opts, index=1)
        measure = c3.selectbox("Measure", list(MEASURES), key="pv_m")
        swap = c4.toggle("Rotate (swap axes)")
        if swap:
            r, c = c, r
        if r == c:
            st.warning("Choose two different dimensions.")
        else:
            col, fn = MEASURES[measure]
            pv = wide.pivot_table(index=r, columns=c, values=col, aggfunc=fn, fill_value=0)
            fig = px.imshow(pv.round(1), text_auto=True, aspect="auto", color_continuous_scale=ui.SEQ,
                            title=f"{measure}: {r} × {c}")
            ui.show(fig, 440)
            st.dataframe(pv.round(2), width="stretch")
