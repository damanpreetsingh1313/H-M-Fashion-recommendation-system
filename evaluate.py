"""Temporal hold-out evaluation (train on the past, predict the final week), metric: MAP@12 and Recall@12."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .model import COMPONENTS, HybridRecommender


def _apk(actual: set, pred: list, k: int = 12) -> float:
    hits, score = 0, 0.0
    for i, p in enumerate(pred[:k]):
        if p in actual:
            hits += 1
            score += hits / (i + 1)
    return score / min(len(actual), k) if actual else 0.0


def evaluate(articles, tx, reviews, calendar, learn_calendar=None, n_users=400, holdout_days=7, seed=0):
    tx = tx.copy()
    tx["t_dat"] = pd.to_datetime(tx.t_dat)
    cutoff = tx.t_dat.max() - pd.Timedelta(days=holdout_days)
    train, test = tx[tx.t_dat <= cutoff], tx[tx.t_dat > cutoff]
    rv = reviews[pd.to_datetime(reviews.date) <= cutoff] if reviews is not None and len(reviews) else None
    model = HybridRecommender(calendar, learn_calendar).fit(articles, train, rv)

    truth = test.groupby("customer_id").article_id.apply(set)
    users = [u for u in truth.index if u in model.hist]
    users = list(np.random.default_rng(seed).permutation(users)[:n_users])

    only = lambda c: {k: float(k == c) for k in COMPONENTS}
    configs = {
        "Popularity only": only("popularity"),
        "Content only": only("content"),
        "Collaborative only": only("collab"),
        "Hybrid (no season)": {"collab": .35, "content": .30, "season": 0, "popularity": .15, "reviews": .10},
        "Hybrid + season": {"collab": .30, "content": .25, "season": .25, "popularity": .10, "reviews": .10},
    }
    scores = {n: [] for n in configs}
    recall = {n: [] for n in configs}
    when = cutoff + pd.Timedelta(days=1)
    for u in users:
        comp, _, seen = model.components(u, date=when)
        mask = np.ones(len(model.articles), dtype=bool)
        mask[list(seen)] = False
        for name, w in configs.items():
            top, _, _ = model.rank(comp, w, mask, 12)
            pred = model.articles.article_id.values[top].tolist()
            scores[name].append(_apk(truth[u], pred))
            recall[name].append(len(truth[u] & set(pred)) / len(truth[u]))
    res = pd.DataFrame({"MAP@12": {n: np.mean(v) for n, v in scores.items()},
                        "Recall@12": {n: np.mean(v) for n, v in recall.items()}})
    res.attrs["n_users"] = len(users)
    return res
