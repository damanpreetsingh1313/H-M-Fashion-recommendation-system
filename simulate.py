"""Data simulator: demo catalogue, season-aware transactions and synthetic reviews.

The simulator is used to (a) run the app without the Kaggle files, (b) generate *new* data for a
chosen season definition ("what if summer lasted 5 months?"), and (c) seed reviews, because the
public H&M dataset contains no review text.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .season import SEASONS, SeasonCalendar, prior_matrix

N_CLUSTERS = 5
TEXT_COLS = ["prod_name", "product_type_name", "garment_group_name", "detail_desc"]

# (product type, product group, garment group)
TYPES = [
    ("T-shirt", "Garment Upper body", "Jersey Basic"), ("Vest top", "Garment Upper body", "Jersey Fancy"),
    ("Sweater", "Garment Upper body", "Knitwear"), ("Hoodie", "Garment Upper body", "Jersey Fancy"),
    ("Cardigan", "Garment Upper body", "Knitwear"), ("Shirt", "Garment Upper body", "Blouses & Shirts"),
    ("Blouse", "Garment Upper body", "Blouses & Shirts"), ("Jacket", "Garment Upper body", "Outdoor"),
    ("Coat", "Garment Upper body", "Outdoor"), ("Puffer jacket", "Garment Upper body", "Outdoor"),
    ("Raincoat", "Garment Upper body", "Outdoor"), ("Dress", "Garment Full body", "Dresses Ladies"),
    ("Skirt", "Garment Lower body", "Skirts"), ("Trousers", "Garment Lower body", "Trousers"),
    ("Jeans", "Garment Lower body", "Trousers Denim"), ("Shorts", "Garment Lower body", "Shorts"),
    ("Leggings", "Garment Lower body", "Jersey Basic"), ("Bikini top", "Swimwear", "Swimwear"),
    ("Swimwear bottom", "Swimwear", "Swimwear"), ("Sandals", "Shoes", "Shoes"),
    ("Boots", "Shoes", "Shoes"), ("Sneakers", "Shoes", "Shoes"), ("Scarf", "Accessories", "Accessories"),
    ("Beanie", "Accessories", "Accessories"), ("Gloves", "Accessories", "Accessories"),
    ("Sunglasses", "Accessories", "Accessories"), ("Sun hat", "Accessories", "Accessories"),
    ("Bag", "Accessories", "Accessories"), ("Socks", "Socks & Tights", "Socks and Tights"),
    ("Pyjama set", "Nightwear", "Nightwear"),
]
COLOURS = ["Black", "White", "Dark Blue", "Light Blue", "Beige", "Grey", "Red", "Pink", "Green",
           "Yellow", "Orange", "Dark Green"]
ADJ = ["Slim", "Relaxed", "Classic", "Oversized", "Soft", "Cosy", "Light", "Essential", "Premium", "Printed"]
INDEX_GROUPS = ["Ladieswear", "Menswear", "Divided", "Sport", "Baby/Children"]
DESC = {"Winter": "warm lined insulated", "Summer": "lightweight breathable airy", "Monsoon": "quick-dry water-repellent"}

POS = ["Love it, great quality and fits perfectly.", "Very comfortable and looks great, highly recommend.",
       "Perfect for the weather, would buy again.", "Soft fabric and a lovely colour."]
MID = ["It is okay, nothing special.", "Decent for the price but the fit is a bit off.",
       "Fabric is fine, colour slightly different from the photo."]
NEG = ["Poor quality, the fabric feels cheap.", "Terrible fit and it shrank after one wash.",
       "Very disappointed, not worth the price."]


def article_text(articles: pd.DataFrame) -> pd.Series:
    t = articles[TEXT_COLS[0]].fillna("").astype(str)
    for c in TEXT_COLS[1:]:
        t = t + " " + articles[c].fillna("").astype(str)
    return t.str.lower()


def make_demo_catalog(n_per_type: int = 25, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows, aid = [], 100_000_000
    for ptype, pgroup, ggroup in TYPES:
        for _ in range(n_per_type):
            colour, adj = rng.choice(COLOURS), rng.choice(ADJ)
            rows.append(dict(
                article_id=aid, prod_name=f"{adj} {ptype}", product_type_name=ptype,
                product_group_name=pgroup, colour_group_name=colour,
                department_name=f"{ggroup} dept", index_group_name=rng.choice(INDEX_GROUPS),
                garment_group_name=ggroup,
                detail_desc=f"{adj} {ptype.lower()} in {colour.lower()}. {rng.choice(list(DESC.values()))} fabric."))
            aid += 1
    return pd.DataFrame(rows)


def make_customers(n: int = 1500) -> list:
    return [f"C{i:05d}" for i in range(n)]


def _cluster_of(ids) -> np.ndarray:
    return (pd.util.hash_pandas_object(pd.Series(ids), index=False).values % N_CLUSTERS).astype(int)


def _cluster_weights(articles: pd.DataFrame) -> np.ndarray:
    rng = np.random.default_rng(123)
    types = articles.product_type_name.unique()
    W = np.ones((N_CLUSTERS, len(articles)))
    for c in range(N_CLUSTERS):
        fav = rng.choice(types, size=min(8, len(types)), replace=False)
        W[c, articles.product_type_name.isin(fav).values] *= 4
        W[c, (articles.index_group_name == INDEX_GROUPS[c % len(INDEX_GROUPS)]).values] *= 2
    return W


def simulate_transactions(articles, customer_ids, calendar: SeasonCalendar, start, end, n, seed=0):
    """Season-aware purchases: demand for an item depends on the season *as defined by `calendar`*."""
    rng = np.random.default_rng(seed)
    prior = prior_matrix(article_text(articles).values)
    base = np.random.default_rng(7).lognormal(0, 1, len(articles))      # stable popularity
    cw = _cluster_weights(articles)
    days = pd.date_range(start, end, freq="D")
    dates = pd.Series(rng.choice(days.values, n))
    cust = np.asarray(rng.choice(np.asarray(customer_ids), n))
    cl = _cluster_of(cust)
    seasons = calendar.seasons_of(dates).values
    aids = articles.article_id.values
    art = np.empty(n, dtype=np.int64)
    for s in np.unique(seasons):
        j = SEASONS.index(s)
        for c in range(N_CLUSTERS):
            m = (seasons == s) & (cl == c)
            if m.any():
                p = base * cw[c] * np.exp(2.5 * prior[:, j])
                art[m] = rng.choice(aids, m.sum(), p=p / p.sum())
    return pd.DataFrame({"t_dat": pd.to_datetime(dates), "customer_id": cust, "article_id": art,
                         "price": np.round(rng.uniform(0.01, 0.12, n), 4)})


def simulate_reviews(tx: pd.DataFrame, articles: pd.DataFrame, n_reviews: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    quality = pd.Series(np.random.default_rng(11).normal(0, 0.7, len(articles)), index=articles.article_id.values)
    s = tx.sample(min(n_reviews, len(tx)), random_state=seed)
    rating = np.clip(np.round(3.7 + quality.loc[s.article_id].values + rng.normal(0, 0.8, len(s))), 1, 5).astype(int)
    text = [rng.choice(POS if r >= 4 else NEG if r <= 2 else MID) for r in rating]
    return pd.DataFrame({"customer_id": s.customer_id.values, "article_id": s.article_id.values,
                         "rating": rating, "text": text, "date": s.t_dat.values})
