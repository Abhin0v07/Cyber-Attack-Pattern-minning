import pandas as pd
import streamlit as st

from core import ui
from core.data import DEFAULTS, DS1, DS2, UPLOAD, load_raw, to_arff

DICTIONARY = {
    DS1: {
        "conn_id": "Unique connection identifier", "timestamp": "Connection start time (UTC, Aug 2026)",
        "network": "Monitored network the traffic was captured on", "src_country": "Geo-located source of the connection",
        "protocol": "Transport protocol (tcp / udp / icmp)", "service": "Destination service (http, ssh, dns, mqtt …)",
        "flag": "TCP connection status flag (SF = normal, S0 = no reply, REJ = rejected …)",
        "duration": "Connection length in seconds", "src_bytes": "Bytes sent source → destination",
        "dst_bytes": "Bytes sent destination → source", "packets": "Total packets exchanged",
        "count": "Connections to the same host in the past 2 s", "srv_count": "Connections to the same service in the past 2 s",
        "serror_rate": "% of connections with SYN errors", "rerror_rate": "% of connections with REJ errors",
        "same_srv_rate": "% of connections to the same service", "dst_host_count": "Distinct connections to destination host",
        "failed_logins": "Failed login attempts", "logged_in": "1 if login succeeded",
        "num_compromised": "Compromised conditions observed", "root_shell": "1 if root shell obtained",
        "attack_type": "Specific attack signature (neptune, smurf, mirai …)",
        "attack_category": "Class label: Normal, DoS, Probe, R2L, U2R, Botnet",
    },
    DS2: {
        "incident_id": "Unique incident identifier", "date": "Date the incident was reported",
        "target_country": "Country of the victim organisation", "sector": "Industry sector of the victim",
        "network": "Network type that was breached", "attack_vector": "Primary attack technique",
        "threat_actor": "Attributed adversary type", "vulnerability": "Root-cause weakness exploited",
        "defense_in_place": "Main defence the organisation had deployed", "affected_users": "Users impacted",
        "records_breached": "Data records exposed", "data_exfiltrated_gb": "Volume of data stolen (GB)",
        "detection_hours": "Time to detect", "resolution_hours": "Time to contain & recover",
        "financial_loss_musd": "Estimated financial loss (US$ million)", "severity": "Class label: Low … Critical",
    },
}

ABOUT = {
    DS1: "Connection-level intrusion records modelled on the **NSL-KDD / KDD Cup '99** schema and captured on five "
         "networks (Corporate LAN, University campus, Cloud data-centre, IoT smart-grid, Banking core). "
         "Used for association rules, J48, Naive Bayes, regression and clustering.",
    DS2: "Organisation-level security incidents (2019 – 2025) with attack vector, threat actor, exploited vulnerability, "
         "impact and financial loss. Used as the second dataset for regression, clustering and classification.",
}


def _show(name, df):
    st.markdown(ABOUT.get(name, ""))
    a, b, c, d = st.columns(4)
    a.metric("Rows", f"{len(df):,}")
    b.metric("Columns", df.shape[1])
    c.metric("Missing cells", f"{int(df.isna().sum().sum()):,}")
    d.metric("Duplicate rows", f"{int(df.duplicated().sum()):,}")
    st.dataframe(df.head(200), width="stretch", height=320)

    info = pd.DataFrame({
        "dtype": df.dtypes.astype(str), "non-null": df.notna().sum(), "unique": df.nunique(),
        "example": df.iloc[0].astype(str),
        "description": [DICTIONARY.get(name, {}).get(c, "") for c in df.columns],
    })
    x, y = st.columns([1.3, 1])
    with x:
        st.markdown("**Data dictionary**")
        st.dataframe(info, width="stretch", height=420)
    with y:
        st.markdown("**Statistical summary**")
        st.dataframe(df.describe().T.round(2), width="stretch", height=420)

    base = name.split(" (")[0].lower().replace(" ", "_")
    d1, d2 = st.columns(2)
    d1.download_button("⬇️ Download CSV", df.to_csv(index=False).encode(), f"{base}.csv", "text/csv", width="stretch")
    ids = DEFAULTS[name]["id_cols"] if name in DEFAULTS else []
    d2.download_button("⬇️ Download ARFF (for Weka)", to_arff(df.drop(columns=ids).dropna(), base).encode(),
                       f"{base}.arff", "text/plain", width="stretch")


def render():
    ui.hero("Data collection", "Datasets", "Two cyber-security datasets are bundled. You can also upload your own CSV or Excel file.")
    t1, t2, t3 = st.tabs(["🌐 DS1 · Network Intrusion Traffic", "🏢 DS2 · Global Cyber Incidents", "⬆️ Upload your own"])
    with t1:
        _show(DS1, load_raw(DS1))
    with t2:
        _show(DS2, load_raw(DS2))
    with t3:
        st.markdown(f"Upload a **CSV or Excel** file (.csv, .xlsx, .xls). It becomes available as **{UPLOAD}** in "
                    "the dataset selector of every mining page (association, J48, Naive Bayes, regression, "
                    "clustering, comparison). The first row must contain the column names.")
        f = st.file_uploader("CSV or Excel file", type=["csv", "xlsx", "xls"])
        if f is not None:
            try:
                sheet = None
                if f.name.lower().endswith((".xlsx", ".xls")):
                    xl = pd.ExcelFile(f)
                    sheet = st.selectbox("Sheet", xl.sheet_names) if len(xl.sheet_names) > 1 else xl.sheet_names[0]
                key = (f.file_id, sheet)
                if st.session_state.get("upload_key") != key:  # only (re)load when a new file / sheet is chosen
                    df = pd.read_csv(f) if sheet is None else xl.parse(sheet)
                    df.columns = [str(c).strip() for c in df.columns]
                    for c in df.columns:  # Excel often mixes numbers and text in one column
                        if df[c].dtype == object:
                            df[c] = df[c].where(df[c].isna(), df[c].astype(str))
                    st.session_state["upload_df"] = df
                    st.session_state["upload_key"] = key
                    st.session_state["upload_name"] = f.name + (f" · {sheet}" if sheet else "")
            except Exception as e:  # noqa: BLE001
                st.error(f"Could not read the file: {e}")
        if st.session_state.get("upload_df") is not None:
            st.success(f"Using **{st.session_state.get('upload_name', 'uploaded file')}** · "
                       f"{len(st.session_state['upload_df']):,} rows")
            _show(UPLOAD, st.session_state["upload_df"])
            if st.button("Remove uploaded dataset"):
                st.session_state["upload_df"] = None
                st.rerun()
