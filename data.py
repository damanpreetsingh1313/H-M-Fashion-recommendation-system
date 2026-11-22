"""Loading the real H&M Kaggle data + persistence of user-generated data (reviews, purchases)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

USER_DIR = Path("user_data")
REVIEW_COLS = ["customer_id", "article_id", "rating", "text", "date"]
TX_COLS = ["t_dat", "customer_id", "article_id", "price"]
ARTICLE_COLS = ["article_id", "prod_name", "product_type_name", "product_group_name", "colour_group_name",
                "department_name", "index_group_name", "garment_group_name", "detail_desc"]


def hm_files_present(data_dir: str) -> bool:
    d = Path(data_dir)
    return all((d / f).exists() for f in ["articles.csv", "customers.csv", "transactions_train.csv"])


def load_real_hm(data_dir: str, n_customers: int = 20_000, seed: int = 42):
    """Reads the Kaggle files. The 3.5 GB transactions file is streamed in chunks and reduced to a random
    sample of customers (cached as a pickle so the next start is instant)."""
    d = Path(data_dir)
    cache = d / f"sample_{n_customers}_{seed}.pkl"
    if cache.exists():
        return pd.read_pickle(cache)
    articles = pd.read_csv(d / "articles.csv")
    cust = pd.read_csv(d / "customers.csv", usecols=["customer_id"]).customer_id
    keep = set(cust.sample(min(n_customers, len(cust)), random_state=seed))
    parts = []
    reader = pd.read_csv(d / "transactions_train.csv", usecols=TX_COLS, chunksize=2_000_000,
                         dtype={"article_id": "int64"}, parse_dates=["t_dat"])
    for chunk in reader:
        parts.append(chunk[chunk.customer_id.isin(keep)])
    tx = pd.concat(parts, ignore_index=True)
    articles = articles[articles.article_id.isin(tx.article_id.unique())][ARTICLE_COLS].reset_index(drop=True)
    pd.to_pickle((articles, tx), cache)
    return articles, tx


def image_path(data_dir: str, article_id: int):
    p = Path(data_dir) / "images" / f"{article_id:010d}"[:3] / f"{article_id:010d}.jpg"
    return str(p) if p.exists() else None


# ---------------------------------------------------------------- user-generated data
def _read(name: str, cols: list, dates: list) -> pd.DataFrame:
    p = USER_DIR / name
    if not p.exists():
        return pd.DataFrame(columns=cols)
    return pd.read_csv(p, parse_dates=dates)


def load_user_reviews() -> pd.DataFrame:
    return _read("reviews.csv", REVIEW_COLS, ["date"])


def load_user_tx() -> pd.DataFrame:
    return _read("new_tx.csv", TX_COLS, ["t_dat"])


def append_rows(name: str, df: pd.DataFrame) -> None:
    USER_DIR.mkdir(exist_ok=True)
    p = USER_DIR / name
    df.to_csv(p, mode="a", header=not p.exists(), index=False)


def reset_user_data() -> None:
    for n in ["reviews.csv", "new_tx.csv"]:
        (USER_DIR / n).unlink(missing_ok=True)
