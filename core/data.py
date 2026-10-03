"""Dataset loading, cleaning and small helpers shared by every page."""
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

DS1 = "Network Intrusion Traffic (DS1)"
DS2 = "Global Cyber Incidents (DS2)"
UPLOAD = "Uploaded file"

# sensible defaults per dataset for each technique
DEFAULTS = {
    DS1: dict(
        file="network_traffic.csv", id_cols=["conn_id", "timestamp"], date_col="timestamp",
        class_target="attack_category",
        class_features=["protocol", "service", "flag", "duration", "src_bytes", "dst_bytes", "count",
                        "srv_count", "serror_rate", "rerror_rate", "same_srv_rate", "dst_host_count",
                        "failed_logins", "logged_in", "num_compromised", "root_shell"],
        reg_target="packets", reg_features=["src_bytes", "dst_bytes", "duration", "count"],
        clu_features=["count", "srv_count", "serror_rate", "rerror_rate", "same_srv_rate", "dst_host_count",
                      "src_bytes", "dst_bytes"],
        clu_label="attack_category", leak_cols=["attack_type"],
        assoc_cols=["network", "src_country", "protocol", "service", "flag", "src_bytes", "count",
                    "serror_rate", "attack_category"],
        assoc_target_prefix="attack_category=",
    ),
    DS2: dict(
        file="cyber_incidents.csv", id_cols=["incident_id", "date"], date_col="date",
        class_target="severity",
        class_features=["sector", "network", "attack_vector", "threat_actor", "vulnerability", "defense_in_place",
                        "affected_users", "records_breached", "detection_hours", "resolution_hours"],
        reg_target="financial_loss_musd",
        reg_features=["records_breached", "resolution_hours", "affected_users", "detection_hours"],
        clu_features=["affected_users", "records_breached", "detection_hours", "resolution_hours",
                      "financial_loss_musd"],
        clu_label="severity", leak_cols=["financial_loss_musd", "data_exfiltrated_gb"],
        assoc_cols=["sector", "network", "attack_vector", "threat_actor", "vulnerability", "defense_in_place",
                    "severity"],
        assoc_target_prefix="severity=",
    ),
}


@st.cache_data(show_spinner=False)
def load_raw(name: str) -> pd.DataFrame:
    cfg = DEFAULTS[name]
    path = DATA_DIR / cfg["file"]
    if not path.exists():  # regenerate if CSVs are missing
        import runpy
        runpy.run_path(str(DATA_DIR / "generate_data.py"), run_name="__main__")
    df = pd.read_csv(path, parse_dates=[cfg["date_col"]])
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Standard cleaning: drop duplicates, median-impute numerics, mode/'Unknown'-impute categoricals."""
    out = df.drop_duplicates().copy()
    for c in out.columns:
        if out[c].isna().any():
            if pd.api.types.is_numeric_dtype(out[c]):
                out[c] = out[c].fillna(out[c].median())
            else:
                out[c] = out[c].fillna("Unknown")
    return out.reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_clean(name: str) -> pd.DataFrame:
    return clean(load_raw(name))


def available_datasets():
    names = [DS1, DS2]
    if st.session_state.get("upload_df") is not None:
        names.append(UPLOAD)
    return names


def get_df(name: str, cleaned=True) -> pd.DataFrame:
    if name == UPLOAD:
        df = st.session_state["upload_df"]
        return clean(df) if cleaned else df
    return load_clean(name) if cleaned else load_raw(name)


def cfg(name: str) -> dict:
    if name in DEFAULTS:
        return DEFAULTS[name]
    df = st.session_state["upload_df"]
    num = numeric_cols(df)
    cat = categorical_cols(df)
    return dict(
        id_cols=[], date_col=None,
        class_target=cat[-1] if cat else (df.columns[-1]),
        class_features=[c for c in df.columns if c != (cat[-1] if cat else df.columns[-1])][:12],
        reg_target=num[-1] if num else None, reg_features=num[:-1][:5],
        clu_features=num[:6], clu_label=cat[-1] if cat else None, leak_cols=[],
        assoc_cols=(cat + num)[:8], assoc_target_prefix="",
    )


def numeric_cols(df):
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]


def categorical_cols(df):
    return [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])
            and not pd.api.types.is_datetime64_any_dtype(df[c])]


def dataset_picker(key: str, label="Dataset", options=None):
    options = options or available_datasets()
    return st.selectbox(label, options, key=key)


def to_arff(df: pd.DataFrame, relation: str) -> str:
    """Export a DataFrame to Weka ARFF so the same data can be run through Weka's J48 / NaiveBayes."""
    df = df[[c for c in df.columns if not pd.api.types.is_datetime64_any_dtype(df[c])]]
    lines = [f"@RELATION {relation}", ""]
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]):
            lines.append(f"@ATTRIBUTE {c} NUMERIC")
        else:
            vals = ",".join(f"'{v}'" for v in sorted(df[c].astype(str).unique()))
            lines.append(f"@ATTRIBUTE {c} {{{vals}}}")
    lines += ["", "@DATA"]
    conv = df.copy()
    for c in conv.columns:
        if not pd.api.types.is_numeric_dtype(conv[c]):
            conv[c] = "'" + conv[c].astype(str) + "'"
    lines += conv.astype(str).agg(",".join, axis=1).tolist()
    return "\n".join(lines)


def discretize(s: pd.Series, bins=3, method="quantile", labels=("low", "med", "high")) -> pd.Series:
    labels = list(labels)[:bins] if len(labels) >= bins else [f"b{i}" for i in range(bins)]
    if method == "quantile":
        r = pd.qcut(s.rank(method="first"), q=bins, labels=labels)
    else:
        r = pd.cut(s, bins=bins, labels=labels)
    return r.astype(str)


def encode_features(df: pd.DataFrame, features):
    """One-hot encode categoricals, keep numerics. Returns X (DataFrame)."""
    features = list(features)
    X = df[features].copy()
    cats = [c for c in features if c in categorical_cols(df)]
    if cats:
        X = pd.get_dummies(X, columns=cats, dtype=int, prefix_sep="=")
    return X


def fmt_int(x):
    return f"{int(x):,}"


def sample(df, n, seed=0):
    return df if len(df) <= n else df.sample(n, random_state=seed)


def safe_log1p(s):
    return np.log1p(s.clip(lower=0))
