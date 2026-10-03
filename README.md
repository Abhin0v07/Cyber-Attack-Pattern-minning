# 🛡️ Cyberattack Pattern Mining Across Networks

**Data Warehousing & Mining (DWM) mini project**: an interactive Streamlit app that mines attack patterns from
network traffic captured on five networks and from 5,000 organisation-level security incidents.

## Experiments covered

| # | Experiment | Page | Details |
|---|------------|------|---------|
| 1 | Association rule mining on a large dataset | 🔗 Association Rules | Apriori and FP-Growth (mlxtend) on 30,000 transactions, a from-scratch level-wise Apriori showing C<sub>k</sub>/L<sub>k</sub> and pruning, rules compared across networks, and a run-time benchmark |
| 2 | Classification using J48 | 🌳 J48 Decision Tree | C4.5-style entropy tree with `-M` and pruning controls, tree diagram, IF-THEN rules, info gain / gain ratio table, Weka-style per-class accuracy, 10-fold CV |
| 3 | Classification using Naive Bayes | 🎲 Naive Bayes | Weka-style NB (Gaussian for numeric, Laplace-smoothed tables for nominal), priors, likelihood tables and a worked posterior calculation |
| 4 | Regression on two datasets | 📈 Regression | Simple, multiple, polynomial and ridge regression on DS1 (packets) and DS2 (financial loss), with the least-squares working, R², adjusted R², RMSE, MAE and residual plots |
| 5 | Clustering on two datasets | 🧩 Clustering | K-Means (elbow and silhouette), agglomerative with dendrogram, DBSCAN (k-distance plot, noise points as anomalies), plus purity, ARI and NMI against the true labels |
| + | Data warehouse and OLAP | 🏛️ Data Warehouse & OLAP | Star schema (1 fact table and 5 dimensions), ETL, and roll-up, drill-down, slice, dice and pivot |
| + | Preprocessing | 🧹 Preprocessing | Missing values, duplicates, IQR and z-score outliers, min-max / z-score / decimal scaling, binning and smoothing, correlation, information gain and PCA |
| + | Evaluation | 🏆 Model Comparison | ZeroR, J48, NB, k-NN, logistic regression and random forest compared side by side |

## Datasets (`data/`)

* **DS1 `network_traffic.csv`**: 30,180 connection records following the NSL-KDD / KDD Cup '99 schema. They come from
  Corporate-LAN, University-Campus, Cloud-DC, IoT-Grid and Banking-Core, and are labelled Normal, DoS, Probe, R2L,
  U2R or Botnet, with 18 attack signatures in total.
* **DS2 `cyber_incidents.csv`**: 5,000 incidents from 2019 to 2025. Each has a sector, attack vector, threat actor,
  vulnerability, defence, impact and financial loss.

Both files are made by `data/generate_data.py` with a fixed seed, so every run gives the same data. A small number of
missing values and duplicate rows are added on purpose so the preprocessing steps have something to fix. You can also
upload any CSV or Excel file (.csv, .xlsx, .xls, with a sheet picker) on the **Datasets** page and run every technique on it. Every dataset can be downloaded as **ARFF**
so you can check the J48 and Naive Bayes results in Weka.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy (Streamlit Community Cloud)

1. Push this folder to a public GitHub repository.
2. Go to https://share.streamlit.io, choose **Create app**, pick the repo and branch, and set the main file to `app.py`.
3. Click **Deploy**. You get a public `*.streamlit.app` URL that anyone can open.
