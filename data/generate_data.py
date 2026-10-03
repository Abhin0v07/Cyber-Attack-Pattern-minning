"""
Generates the two datasets used by the Cyberattack Pattern Mining app.

DS1  network_traffic.csv   ~30,000 connection records captured across 5 networks,
                           modelled on the NSL-KDD / KDD Cup'99 intrusion-detection schema
                           (protocol, service, flag, bytes, counts, error rates, attack labels).
DS2  cyber_incidents.csv   ~5,000 organisation-level security incidents (2019-2025) with
                           sector, attack vector, threat actor, impact and financial loss.

Both are generated with a fixed random seed so results are reproducible.
Small amounts of missing values and duplicate rows are injected on purpose so the
Preprocessing page has something real to clean.

Run:  python data/generate_data.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
rng = np.random.default_rng(42)


def pick(options, probs, n):
    probs = np.asarray(probs, dtype=float)
    return rng.choice(options, size=n, p=probs / probs.sum())


# ----------------------------------------------------------------------------------------------
# DS1 - Network intrusion traffic across networks
# ----------------------------------------------------------------------------------------------
NETWORKS = ["Corporate-LAN", "University-Campus", "Cloud-DC", "IoT-Grid", "Banking-Core"]
NET_WEIGHTS = [0.24, 0.20, 0.22, 0.16, 0.18]
CATEGORIES = ["Normal", "DoS", "Probe", "R2L", "U2R", "Botnet"]
NET_MIX = {  # attack-category mix inside each network
    "Corporate-LAN":     [0.50, 0.18, 0.14, 0.10, 0.03, 0.05],
    "University-Campus": [0.45, 0.15, 0.25, 0.08, 0.02, 0.05],
    "Cloud-DC":          [0.42, 0.35, 0.12, 0.06, 0.02, 0.03],
    "IoT-Grid":          [0.38, 0.14, 0.10, 0.04, 0.01, 0.33],
    "Banking-Core":      [0.52, 0.12, 0.10, 0.21, 0.04, 0.01],
}
SUBTYPES = {
    "Normal": (["normal"], [1]),
    "DoS": (["neptune", "smurf", "teardrop", "http_flood"], [0.50, 0.25, 0.10, 0.15]),
    "Probe": (["portsweep", "nmap", "satan", "ipsweep"], [0.30, 0.25, 0.25, 0.20]),
    "R2L": (["guess_passwd", "warezclient", "ftp_write", "sql_injection"], [0.45, 0.20, 0.10, 0.25]),
    "U2R": (["buffer_overflow", "rootkit", "loadmodule"], [0.50, 0.30, 0.20]),
    "Botnet": (["mirai", "zeus", "c2_beacon"], [0.60, 0.25, 0.15]),
}
COUNTRIES = ["India", "USA", "China", "Russia", "Brazil", "Germany", "Netherlands", "Iran", "North Korea", "Tor-Exit"]
COUNTRY_MIX = {
    "Normal": [45, 20, 5, 3, 7, 10, 8, 1, 0.2, 0.8],
    "DoS":    [6, 12, 25, 20, 12, 5, 10, 3, 1, 6],
    "Probe":  [5, 15, 30, 15, 6, 4, 10, 4, 1, 10],
    "R2L":    [4, 6, 15, 25, 5, 3, 5, 10, 7, 20],
    "U2R":    [2, 4, 12, 25, 2, 2, 3, 15, 15, 20],
    "Botnet": [15, 8, 20, 15, 20, 3, 6, 4, 2, 7],
}
SERVICE_PROTO = {
    "http": "tcp", "https": "tcp", "ftp": "tcp", "ssh": "tcp", "smtp": "tcp", "telnet": "tcp",
    "smb": "tcp", "rdp": "tcp", "mqtt": "tcp", "dns": "udp", "ecr_i": "icmp", "eco_i": "icmp",
}
ALL_SERVICES = list(SERVICE_PROTO) + ["private"]


def lognorm(mu, sigma, n):
    return rng.lognormal(mu, sigma, n)


def base(n):
    return dict(
        service=np.full(n, "http", dtype=object), flag=np.full(n, "SF", dtype=object),
        duration=np.zeros(n), src_bytes=np.zeros(n), dst_bytes=np.zeros(n),
        count=np.ones(n), srv_count=np.ones(n), serror_rate=rng.beta(1, 40, n),
        rerror_rate=rng.beta(1, 40, n), same_srv_rate=rng.beta(12, 1.5, n),
        dst_host_count=rng.binomial(255, 0.3, n), failed_logins=np.zeros(n),
        logged_in=np.zeros(n), num_compromised=np.zeros(n), root_shell=np.zeros(n),
    )


def gen_subtype(sub, n, nets):
    d = base(n)
    if sub == "normal":
        iot = nets == "IoT-Grid"
        d["service"] = pick(["http", "https", "dns", "smtp", "ssh", "ftp", "smb", "rdp", "mqtt"],
                            [25, 30, 15, 8, 7, 4, 5, 3, 3], n)
        d["service"][iot & (rng.random(n) < 0.5)] = "mqtt"
        d["flag"] = pick(["SF", "REJ", "RSTO", "S0"], [93, 3, 2, 2], n)
        d["duration"] = rng.exponential(3, n)
        d["src_bytes"] = lognorm(5.8, 1.0, n)
        d["dst_bytes"] = lognorm(8.0, 1.6, n)
        d["count"] = rng.poisson(6, n) + 1
        d["srv_count"] = np.minimum(d["count"], rng.poisson(5, n) + 1)
        d["failed_logins"] = (rng.random(n) < 0.02).astype(int)
        d["logged_in"] = (rng.random(n) < 0.85).astype(int)
    elif sub == "neptune":
        d["service"] = pick(["private", "http", "smtp", "ftp", "telnet"], [60, 20, 8, 6, 6], n)
        d["flag"] = pick(["S0", "REJ"], [90, 10], n)
        d["count"] = np.minimum(rng.poisson(180, n) + 50, 511)
        d["srv_count"] = rng.poisson(12, n) + 1
        d["serror_rate"] = rng.beta(40, 1, n)
        d["same_srv_rate"] = rng.beta(1, 12, n)
        d["dst_host_count"] = np.full(n, 255)
    elif sub == "smurf":
        d["service"] = np.full(n, "ecr_i", dtype=object)
        d["src_bytes"] = pick([1032, 520], [75, 25], n).astype(float)
        d["count"] = np.minimum(rng.poisson(480, n), 511)
        d["srv_count"] = d["count"]
        d["same_srv_rate"] = np.ones(n)
        d["dst_host_count"] = np.full(n, 255)
    elif sub == "teardrop":
        d["service"] = np.full(n, "private", dtype=object)
        d["src_bytes"] = np.full(n, 28.0)
        d["count"] = rng.poisson(40, n) + 5
        d["srv_count"] = d["count"]
    elif sub == "http_flood":
        d["service"] = pick(["http", "https"], [60, 40], n)
        d["duration"] = rng.exponential(0.5, n)
        d["src_bytes"] = lognorm(5.5, 0.3, n)
        d["dst_bytes"] = lognorm(9.0, 0.5, n)
        d["count"] = np.minimum(rng.poisson(250, n), 511)
        d["srv_count"] = d["count"]
        d["same_srv_rate"] = rng.beta(30, 1, n)
        d["logged_in"] = np.ones(n)
    elif sub in ("portsweep", "nmap", "satan", "ipsweep"):
        d["service"] = pick(ALL_SERVICES, [1] * len(ALL_SERVICES), n)
        if sub == "ipsweep":
            d["service"] = np.full(n, "eco_i", dtype=object)
        d["flag"] = pick(["REJ", "RSTO", "S0", "SF"], [40, 25, 15, 20], n)
        d["src_bytes"] = rng.poisson(5, n).astype(float)
        d["dst_bytes"] = np.where(rng.random(n) < 0.1, rng.poisson(40, n), 0).astype(float)
        d["count"] = rng.poisson({"portsweep": 3, "nmap": 20, "satan": 90, "ipsweep": 5}[sub], n) + 1
        d["srv_count"] = np.maximum(1, (d["count"] * rng.beta(1, 8, n)).round())
        d["rerror_rate"] = rng.beta(6, 3, n)
        d["serror_rate"] = rng.beta(2, 6, n)
        d["same_srv_rate"] = rng.beta(1, 8, n)
        d["dst_host_count"] = rng.integers(150, 256, n) if sub in ("ipsweep", "satan") else rng.integers(1, 40, n)
    elif sub == "guess_passwd":
        d["service"] = pick(["telnet", "ftp", "ssh", "rdp"], [35, 20, 35, 10], n)
        d["flag"] = pick(["SF", "RSTO"], [60, 40], n)
        d["duration"] = rng.exponential(4, n)
        d["src_bytes"] = lognorm(4.5, 0.4, n)
        d["dst_bytes"] = lognorm(5.0, 0.5, n)
        d["failed_logins"] = rng.poisson(3, n) + 1
        d["count"] = rng.poisson(2, n) + 1
    elif sub == "warezclient":
        d["service"] = np.full(n, "ftp", dtype=object)
        d["duration"] = rng.exponential(300, n)
        d["src_bytes"] = lognorm(10.0, 1.0, n)
        d["dst_bytes"] = lognorm(6.0, 1.0, n)
        d["logged_in"] = np.ones(n)
    elif sub == "ftp_write":
        d["service"] = np.full(n, "ftp", dtype=object)
        d["duration"] = rng.exponential(30, n)
        d["src_bytes"] = lognorm(7.0, 1.0, n)
        d["dst_bytes"] = lognorm(6.0, 1.0, n)
        d["logged_in"] = np.ones(n)
        d["num_compromised"] = rng.poisson(0.5, n)
    elif sub == "sql_injection":
        d["service"] = pick(["http", "https"], [55, 45], n)
        d["duration"] = rng.exponential(1, n)
        d["src_bytes"] = lognorm(7.2, 0.5, n)
        d["dst_bytes"] = lognorm(9.5, 1.2, n)
        d["logged_in"] = np.ones(n)
        d["count"] = rng.poisson(15, n) + 1
        d["srv_count"] = d["count"]
        d["num_compromised"] = rng.poisson(0.3, n)
    elif sub in ("buffer_overflow", "rootkit", "loadmodule"):
        d["service"] = pick(["telnet", "ftp", "ssh"], [70, 10, 20], n)
        d["duration"] = rng.exponential(120, n) + 10
        d["src_bytes"] = lognorm(7.5, 1.0, n)
        d["dst_bytes"] = lognorm(8.5, 1.0, n)
        d["logged_in"] = np.ones(n)
        d["root_shell"] = (rng.random(n) < {"buffer_overflow": 0.8, "rootkit": 0.6, "loadmodule": 0.7}[sub]).astype(int)
        d["num_compromised"] = rng.poisson(2, n) + 1
        d["failed_logins"] = (rng.random(n) < 0.2).astype(int)
    elif sub == "mirai":
        d["service"] = pick(["telnet", "private", "mqtt"], [45, 40, 15], n)
        d["flag"] = pick(["SF", "S0"], [60, 40], n)
        d["src_bytes"] = lognorm(4.0, 0.5, n)
        d["count"] = np.minimum(rng.poisson(120, n), 511)
        d["srv_count"] = (d["count"] * rng.beta(8, 2, n)).round() + 1
        d["failed_logins"] = np.where(d["service"] == "telnet", rng.poisson(1.5, n), 0)
        d["dst_host_count"] = rng.integers(100, 256, n)
    elif sub == "zeus":
        d["service"] = pick(["https", "http"], [70, 30], n)
        d["duration"] = rng.exponential(40, n)
        d["src_bytes"] = lognorm(6.5, 0.5, n)
        d["dst_bytes"] = lognorm(7.0, 0.6, n)
        d["count"] = rng.poisson(3, n) + 1
        d["logged_in"] = np.ones(n)
    elif sub == "c2_beacon":
        d["service"] = pick(["dns", "https"], [60, 40], n)
        d["duration"] = rng.exponential(0.3, n)
        d["src_bytes"] = lognorm(5.0, 0.2, n)
        d["dst_bytes"] = lognorm(5.5, 0.3, n)
        d["count"] = rng.poisson(4, n) + 1
        d["same_srv_rate"] = rng.beta(30, 1, n)
    return d


def make_network_traffic(n=30_000):
    nets = pick(NETWORKS, NET_WEIGHTS, n)
    cats = np.empty(n, dtype=object)
    for net in NETWORKS:
        m = nets == net
        cats[m] = pick(CATEGORIES, NET_MIX[net], m.sum())

    frames = []
    for cat in CATEGORIES:
        idx = np.where(cats == cat)[0]
        subs, p = SUBTYPES[cat]
        sub_arr = pick(subs, p, len(idx))
        for sub in subs:
            sidx = idx[sub_arr == sub]
            if len(sidx) == 0:
                continue
            d = gen_subtype(sub, len(sidx), nets[sidx])
            f = pd.DataFrame(d)
            f["network"] = nets[sidx]
            f["attack_category"] = cat
            f["attack_type"] = sub
            f["src_country"] = pick(COUNTRIES, COUNTRY_MIX[cat], len(sidx))
            frames.append(f)
    df = pd.concat(frames, ignore_index=True)

    # protocol derived from service
    df["protocol"] = df["service"].map(SERVICE_PROTO)
    priv = df["service"] == "private"
    df.loc[priv, "protocol"] = pick(["udp", "tcp"], [60, 40], priv.sum())
    df.loc[df["attack_type"] == "neptune", "protocol"] = "tcp"
    df.loc[df["attack_type"] == "teardrop", "protocol"] = "udp"

    # timestamps: August 2026. Normal traffic follows business hours, attacks peak at night;
    # a DoS campaign hits days 12-14 and botnet activity ramps up over the month.
    m = len(df)
    day = rng.integers(1, 32, m)
    dos = (df["attack_category"] == "DoS").to_numpy()
    campaign = dos & (rng.random(m) < 0.45)
    day[campaign] = rng.integers(12, 15, campaign.sum())
    bot = (df["attack_category"] == "Botnet").to_numpy()
    day[bot] = np.clip(np.round(31 * rng.beta(2.5, 1, bot.sum())), 1, 31).astype(int)
    normal = (df["attack_category"] == "Normal").to_numpy()
    hour = np.where(normal, np.round(rng.normal(13, 3.5, m)) % 24,
                    np.where(rng.random(m) < 0.5, np.round(rng.normal(2, 2.5, m)) % 24,
                             np.round(rng.normal(20, 3, m)) % 24)).astype(int)
    minute = rng.integers(0, 60, m)
    second = rng.integers(0, 60, m)
    df["timestamp"] = pd.to_datetime(dict(year=2026, month=8, day=day, hour=hour, minute=minute, second=second))

    # packets ~ bytes / mean packet size
    pkt_size = rng.uniform(400, 1400, m)
    df["packets"] = np.maximum(1, np.ceil((df["src_bytes"] + df["dst_bytes"]) / pkt_size + rng.poisson(2, m))).astype(int)

    for c in ["duration", "serror_rate", "rerror_rate", "same_srv_rate"]:
        df[c] = df[c].round(3)
    for c in ["src_bytes", "dst_bytes", "count", "srv_count", "dst_host_count", "failed_logins",
              "logged_in", "num_compromised", "root_shell"]:
        df[c] = df[c].round().astype(int)
    for c in ["serror_rate", "rerror_rate", "same_srv_rate"]:
        df[c] = df[c].clip(0, 1)

    df = df.sort_values("timestamp").reset_index(drop=True)
    df.insert(0, "conn_id", [f"C{i:06d}" for i in range(1, len(df) + 1)])
    cols = ["conn_id", "timestamp", "network", "src_country", "protocol", "service", "flag",
            "duration", "src_bytes", "dst_bytes", "packets", "count", "srv_count", "serror_rate",
            "rerror_rate", "same_srv_rate", "dst_host_count", "failed_logins", "logged_in",
            "num_compromised", "root_shell", "attack_type", "attack_category"]
    df = df[cols]

    # inject data-quality issues
    df = df.astype({"dst_bytes": "float", "duration": "float"})
    df.loc[rng.random(len(df)) < 0.015, "dst_bytes"] = np.nan
    df.loc[rng.random(len(df)) < 0.010, "duration"] = np.nan
    df.loc[rng.random(len(df)) < 0.010, "src_country"] = np.nan
    dups = df.sample(180, random_state=1)
    df = pd.concat([df, dups]).sort_values("timestamp").reset_index(drop=True)
    return df


# ----------------------------------------------------------------------------------------------
# DS2 - Global cyber incidents
# ----------------------------------------------------------------------------------------------
SECTORS = ["Finance", "Healthcare", "Government", "Education", "Retail", "Energy", "IT & Telecom"]
SECTOR_W = [20, 17, 14, 12, 13, 9, 15]
VECTORS = ["Phishing", "Ransomware", "DDoS", "SQL Injection", "Malware", "Insider Threat",
           "Zero-Day Exploit", "Credential Stuffing", "Man-in-the-Middle"]
SECTOR_VECTOR = {
    "Finance":      [20, 14, 15, 10, 8, 8, 3, 17, 5],
    "Healthcare":   [25, 35, 5, 5, 12, 8, 2, 5, 3],
    "Government":   [18, 10, 15, 5, 12, 8, 20, 4, 8],
    "Education":    [30, 15, 25, 8, 10, 3, 1, 6, 2],
    "Retail":       [15, 12, 10, 25, 8, 5, 2, 20, 3],
    "Energy":       [12, 25, 8, 3, 25, 5, 18, 2, 2],
    "IT & Telecom": [14, 14, 18, 12, 12, 6, 12, 8, 4],
}
SECTOR_NET = {
    "Finance": (["Banking-Core", "Cloud-DC", "Corporate-LAN"], [60, 30, 10]),
    "Healthcare": (["Corporate-LAN", "Cloud-DC", "IoT-Grid"], [50, 30, 20]),
    "Government": (["Corporate-LAN", "Cloud-DC"], [60, 40]),
    "Education": (["University-Campus", "Cloud-DC"], [80, 20]),
    "Retail": (["Cloud-DC", "Corporate-LAN"], [60, 40]),
    "Energy": (["IoT-Grid", "Corporate-LAN"], [70, 30]),
    "IT & Telecom": (["Cloud-DC", "Corporate-LAN"], [70, 30]),
}
VECTOR_ACTOR = {
    "Phishing": [20, 5, 60, 0, 15], "Ransomware": [8, 2, 80, 0, 10], "DDoS": [20, 50, 25, 0, 5],
    "SQL Injection": [5, 20, 60, 0, 15], "Malware": [30, 5, 50, 0, 15], "Insider Threat": [0, 0, 5, 95, 0],
    "Zero-Day Exploit": [70, 5, 15, 0, 10], "Credential Stuffing": [2, 3, 85, 0, 10],
    "Man-in-the-Middle": [20, 5, 50, 0, 25],
}
ACTORS = ["Nation-State", "Hacktivist", "Cybercriminal", "Insider", "Unknown"]
VULNS = ["Social Engineering", "Unpatched Software", "Weak Passwords", "Misconfiguration", "Zero-Day", "Excess Privileges"]
VECTOR_VULN = {
    "Phishing": [85, 5, 10, 0, 0, 0], "Ransomware": [35, 40, 25, 0, 0, 0], "DDoS": [0, 40, 0, 60, 0, 0],
    "SQL Injection": [0, 50, 0, 50, 0, 0], "Malware": [50, 45, 5, 0, 0, 0], "Insider Threat": [0, 0, 10, 20, 0, 70],
    "Zero-Day Exploit": [0, 0, 0, 0, 100, 0], "Credential Stuffing": [0, 0, 90, 10, 0, 0],
    "Man-in-the-Middle": [5, 15, 10, 70, 0, 0],
}
DEFENSES = ["Firewall", "AI-based IDS", "MFA", "Antivirus", "Encryption", "VPN"]
TARGET_COUNTRIES = ["USA", "India", "UK", "Germany", "Japan", "Brazil", "Australia", "France", "China", "Russia"]
V_USERS = {"DDoS": 3.0, "Insider Threat": 0.1, "Phishing": 0.6, "Zero-Day Exploit": 1.5, "Ransomware": 1.2}
V_RECORDS = {"DDoS": 0.02, "Insider Threat": 1.8, "SQL Injection": 2.5, "Credential Stuffing": 1.6, "Ransomware": 1.4}
V_DETECT = {"Zero-Day Exploit": 4.0, "Insider Threat": 4.5, "DDoS": 0.05, "Ransomware": 0.4, "Man-in-the-Middle": 2.0}
V_RESOLVE = {"Ransomware": 2.5, "Zero-Day Exploit": 2.0, "DDoS": 0.3, "Phishing": 0.7}
SECTOR_LOSS = {"Finance": 1.4, "Healthcare": 1.35, "Government": 1.1, "Education": 0.7, "Retail": 1.0,
               "Energy": 1.25, "IT & Telecom": 1.1}


def make_incidents(n=5_000):
    sector = pick(SECTORS, SECTOR_W, n)
    vector = np.empty(n, dtype=object)
    network = np.empty(n, dtype=object)
    for s in SECTORS:
        m = sector == s
        vector[m] = pick(VECTORS, SECTOR_VECTOR[s], m.sum())
        opts, p = SECTOR_NET[s]
        network[m] = pick(opts, p, m.sum())
    actor = np.empty(n, dtype=object)
    vuln = np.empty(n, dtype=object)
    for v in VECTORS:
        m = vector == v
        actor[m] = pick(ACTORS, VECTOR_ACTOR[v], m.sum())
        vuln[m] = pick(VULNS, VECTOR_VULN[v], m.sum())
    defense = pick(DEFENSES, [25, 15, 18, 20, 12, 10], n)
    # credential stuffing rarely succeeds against MFA -> reassign
    cs_mfa = (vector == "Credential Stuffing") & (defense == "MFA")
    defense[cs_mfa] = pick(["Firewall", "Antivirus", "VPN"], [1, 1, 1], cs_mfa.sum())

    mult = lambda table, default=1.0: np.array([table.get(v, default) for v in vector])
    users = np.round(lognorm(8.0, 1.4, n) * mult(V_USERS)).astype(int) + 1
    records = np.round(lognorm(10.0, 1.6, n) * mult(V_RECORDS)).astype(int)
    detect = lognorm(3.0, 0.9, n) * mult(V_DETECT) * np.where(defense == "AI-based IDS", 0.4, 1.0)
    resolve = 0.5 * detect + lognorm(3.5, 0.7, n) * mult(V_RESOLVE)
    exfil = records * rng.uniform(1e-5, 3e-5, n)
    ransom = (vector == "Ransomware").astype(float)
    smult = np.array([SECTOR_LOSS[s] for s in sector])
    loss = (0.15 + 4e-6 * records + 0.006 * resolve + 8e-7 * users + 1.2 * ransom) * smult \
        * np.exp(rng.normal(0, 0.25, n))

    sev = np.where((loss > 5) | (records > 1_000_000), "Critical",
                   np.where(loss > 1.5, "High", np.where(loss > 0.5, "Medium", "Low")))
    year = pick(list(range(2019, 2026)), [8, 11, 14, 15, 16, 17, 19], n)
    month = rng.integers(1, 13, n)
    dayy = rng.integers(1, 29, n)
    df = pd.DataFrame({
        "incident_id": [f"INC-{i:05d}" for i in range(1, n + 1)],
        "date": pd.to_datetime(dict(year=year, month=month, day=dayy)),
        "target_country": pick(TARGET_COUNTRIES, [22, 14, 10, 9, 8, 8, 7, 7, 8, 7], n),
        "sector": sector, "network": network, "attack_vector": vector, "threat_actor": actor,
        "vulnerability": vuln, "defense_in_place": defense,
        "affected_users": users, "records_breached": records,
        "data_exfiltrated_gb": exfil.round(3), "detection_hours": detect.round(1),
        "resolution_hours": resolve.round(1), "financial_loss_musd": loss.round(3), "severity": sev,
    })
    df = df.sort_values("date").reset_index(drop=True)
    df["incident_id"] = [f"INC-{i:05d}" for i in range(1, n + 1)]
    df.loc[rng.random(n) < 0.02, "detection_hours"] = np.nan
    df.loc[rng.random(n) < 0.01, "threat_actor"] = np.nan
    return df


if __name__ == "__main__":
    nt = make_network_traffic()
    nt.to_csv(OUT / "network_traffic.csv", index=False)
    print("network_traffic.csv", nt.shape)
    print(nt["attack_category"].value_counts())
    inc = make_incidents()
    inc.to_csv(OUT / "cyber_incidents.csv", index=False)
    print("cyber_incidents.csv", inc.shape)
    print(inc["severity"].value_counts())
