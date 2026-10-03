"""Model helpers: Weka-style Naive Bayes, J48 tree rendering / rule extraction, evaluation tables."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import (accuracy_score, cohen_kappa_score, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split

from core.ui import ATTACK_COLORS, PALETTE


# ----------------------------------------------------------------------------------------------
# Naive Bayes that mirrors weka.classifiers.bayes.NaiveBayes:
# Gaussian likelihood for numeric attributes, Laplace-smoothed frequency tables for nominal ones.
# ----------------------------------------------------------------------------------------------
class MixedNB(BaseEstimator, ClassifierMixin):
    def __init__(self, numeric=(), nominal=(), alpha=1.0):
        self.numeric = numeric
        self.nominal = nominal
        self.alpha = alpha

    def fit(self, X: pd.DataFrame, y):
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        n = len(y)
        self.class_count_ = np.array([(y == c).sum() for c in self.classes_])
        self.prior_ = (self.class_count_ + self.alpha) / (n + self.alpha * len(self.classes_))
        self.mean_, self.std_ = {}, {}
        for col in self.numeric:
            v = X[col].to_numpy(dtype=float)
            floor = max(v.std() * 1e-3, 1e-6)
            self.mean_[col] = np.array([v[y == c].mean() for c in self.classes_])
            self.std_[col] = np.array([max(v[y == c].std(), floor) for c in self.classes_])
        self.values_, self.cpt_ = {}, {}
        for col in self.nominal:
            v = X[col].astype(str).to_numpy()
            vals = np.unique(v)
            self.values_[col] = vals
            tab = np.array([[((y == c) & (v == val)).sum() for val in vals] for c in self.classes_], dtype=float)
            self.cpt_[col] = (tab + self.alpha) / (tab.sum(axis=1, keepdims=True) + self.alpha * len(vals))
        return self

    def _log_terms(self, X: pd.DataFrame):
        """Per-attribute log-likelihood arrays (n_samples × n_classes)."""
        terms = {}
        for col in self.numeric:
            v = X[col].to_numpy(dtype=float)[:, None]
            m, s = self.mean_[col][None, :], self.std_[col][None, :]
            terms[col] = -0.5 * np.log(2 * np.pi * s ** 2) - (v - m) ** 2 / (2 * s ** 2)
        for col in self.nominal:
            v = X[col].astype(str).to_numpy()
            idx = {val: i for i, val in enumerate(self.values_[col])}
            k = len(self.values_[col])
            unseen = np.log(self.alpha / (self.class_count_ + self.alpha * (k + 1)))
            codes = pd.Series(v).map(idx).to_numpy(dtype=float)
            seen = ~np.isnan(codes)
            out = np.empty((len(v), len(self.classes_)))
            out[seen] = np.log(self.cpt_[col][:, codes[seen].astype(int)]).T
            out[~seen] = unseen
            terms[col] = out
        return terms

    def predict_log_joint(self, X):
        total = np.log(self.prior_)[None, :].repeat(len(X), axis=0)
        for t in self._log_terms(X).values():
            total = total + t
        return total

    def predict_proba(self, X):
        lj = self.predict_log_joint(X)
        lj -= lj.max(axis=1, keepdims=True)
        p = np.exp(lj)
        return p / p.sum(axis=1, keepdims=True)

    def predict(self, X):
        return self.classes_[self.predict_log_joint(X).argmax(axis=1)]


# ----------------------------------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------------------------------
def evaluate(model, X, y, mode="split", split=0.66, folds=10, seed=42):
    """Returns dict with y_true, y_pred, proba, classes, fitted model (on training part / full data)."""
    y = np.asarray(y)
    if mode == "split":
        Xtr, Xte, ytr, yte = train_test_split(X, y, train_size=split, random_state=seed, stratify=y)
        model.fit(Xtr, ytr)
        pred = model.predict(Xte)
        proba = model.predict_proba(Xte)
        return dict(y_true=yte, y_pred=pred, proba=proba, classes=model.classes_, model=model,
                    n_train=len(ytr), n_test=len(yte))
    cv = StratifiedKFold(folds, shuffle=True, random_state=seed)
    proba = cross_val_predict(model, X, y, cv=cv, method="predict_proba")
    model.fit(X, y)
    classes = model.classes_
    pred = classes[proba.argmax(axis=1)]
    return dict(y_true=y, y_pred=pred, proba=proba, classes=classes, model=model, n_train=len(y), n_test=len(y))


def summary(res):
    yt, yp = res["y_true"], res["y_pred"]
    out = dict(
        accuracy=accuracy_score(yt, yp), kappa=cohen_kappa_score(yt, yp),
        precision=precision_score(yt, yp, average="weighted", zero_division=0),
        recall=recall_score(yt, yp, average="weighted", zero_division=0),
        f1=f1_score(yt, yp, average="weighted", zero_division=0),
    )
    try:
        out["roc_auc"] = roc_auc_score(yt, res["proba"], multi_class="ovr", average="weighted", labels=res["classes"]) \
            if len(res["classes"]) > 2 else roc_auc_score(yt == res["classes"][1], res["proba"][:, 1])
    except ValueError:
        out["roc_auc"] = np.nan
    return out


def per_class(res):
    """Weka-style 'Detailed Accuracy By Class'."""
    yt, yp, classes = res["y_true"], res["y_pred"], res["classes"]
    cm = confusion_matrix(yt, yp, labels=classes)
    rows = []
    for i, c in enumerate(classes):
        tp = cm[i, i]
        fn = cm[i].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = cm.sum() - tp - fn - fp
        prec = tp / (tp + fp) if tp + fp else 0
        rec = tp / (tp + fn) if tp + fn else 0
        try:
            auc = roc_auc_score(yt == c, res["proba"][:, i])
        except ValueError:
            auc = np.nan
        rows.append({"class": c, "TP rate": rec, "FP rate": fp / (fp + tn) if fp + tn else 0, "precision": prec,
                     "recall": rec, "F-measure": 2 * prec * rec / (prec + rec) if prec + rec else 0,
                     "ROC area": auc, "support": int(cm[i].sum())})
    return pd.DataFrame(rows), cm


# ----------------------------------------------------------------------------------------------
# Decision tree → Graphviz / IF-THEN rules
# ----------------------------------------------------------------------------------------------
def _cond(feature, thr, left):
    if "=" in feature:  # one-hot encoded nominal attribute
        a, v = feature.split("=", 1)
        return f"{a} {'≠' if left else '='} {v}"
    return f"{feature} {'≤' if left else '>'} {thr:,.3g}"


def tree_rules(tree, features, classes):
    t = tree.tree_
    rules = []

    def walk(node, conds):
        if t.children_left[node] == -1:
            dist = t.value[node][0]
            dist = dist / dist.sum()
            k = int(dist.argmax())
            rules.append(dict(conditions=conds, cls=classes[k], confidence=float(dist[k]),
                              samples=int(t.n_node_samples[node])))
            return
        f, thr = features[t.feature[node]], t.threshold[node]
        walk(t.children_left[node], conds + [_cond(f, thr, True)])
        walk(t.children_right[node], conds + [_cond(f, thr, False)])

    walk(0, [])
    return sorted(rules, key=lambda r: -r["samples"])


def _color(cls, i):
    return ATTACK_COLORS.get(str(cls), PALETTE[i % len(PALETTE)])


def tree_dot(tree, features, classes, max_depth=4):
    t = tree.tree_
    cls_idx = {c: i for i, c in enumerate(classes)}
    lines = ['digraph T {', 'graph [bgcolor="transparent", ranksep=0.35, nodesep=0.2];',
             'node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=10, fontcolor="#1d1c1a", '
             'color="#1d1c1a"];', 'edge [color="#6b6760", fontcolor="#6b6760", fontname="Helvetica", fontsize=9];']

    def walk(node, depth):
        dist = t.value[node][0]
        k = int(dist.argmax())
        cls = classes[k]
        n = int(t.n_node_samples[node])
        pur = dist[k] / dist.sum()
        leaf = t.children_left[node] == -1
        if leaf or depth >= max_depth:
            label = f"{cls}\\n({n:,} · {pur:.0%})" + ("" if leaf else "\\n…")
            lines.append(f'n{node} [label="{label}", fillcolor="#ffffff", color="{_color(cls, cls_idx[cls])}", penwidth=2.4];')
            return
        f = features[t.feature[node]]
        name = f.split("=")[0] + " = " + f.split("=", 1)[1] + " ?" if "=" in f else f"{f} ≤ {t.threshold[node]:,.3g} ?"
        lines.append(f'n{node} [label="{name}\\n{n:,} samples", fillcolor="#fffdf9"];')
        for child, lab in ((t.children_left[node], "no" if "=" in f else "yes"),
                           (t.children_right[node], "yes" if "=" in f else "no")):
            walk(child, depth + 1)
            lines.append(f'n{node} -> n{child} [label="{lab}"];')

    walk(0, 0)
    lines.append("}")
    return "\n".join(lines)


# ----------------------------------------------------------------------------------------------
# Information gain / gain ratio (C4.5 split criterion) on nominal/discretised attributes
# ----------------------------------------------------------------------------------------------
def entropy(labels):
    p = pd.Series(labels).value_counts(normalize=True).to_numpy()
    return float(-(p * np.log2(p)).sum())


def gain_table(df: pd.DataFrame, target: str, attrs):
    H = entropy(df[target])
    rows = []
    for a in attrs:
        col = df[a]
        if pd.api.types.is_numeric_dtype(col) and col.nunique() > 5:
            col = pd.qcut(col.rank(method="first"), 4, labels=["q1", "q2", "q3", "q4"]).astype(str)
        w = col.value_counts(normalize=True)
        cond = sum(w[v] * entropy(df.loc[col == v, target]) for v in w.index)
        split_info = float(-(w * np.log2(w)).sum())
        ig = H - cond
        rows.append({"attribute": a, "values": len(w), "Info(D)": H, "Info_A(D)": cond, "Gain": ig,
                     "SplitInfo": split_info, "GainRatio": ig / split_info if split_info else 0})
    return pd.DataFrame(rows).sort_values("GainRatio", ascending=False)
