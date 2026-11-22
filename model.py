"""Season-aware hybrid recommender.

Components (each min-max scaled to 0..1, then blended with adjustable weights):
  collab      TruncatedSVD (scikit-learn) on the user x item matrix; users are 'folded in' at query time,
              so new purchases / positive reviews change recommendations instantly without retraining
  content     TF-IDF of product text, cosine similarity to a recency-weighted taste profile
  season      learned seasonal lift per product type (from the transactions, bucketed by the season
              calendar) blended with a keyword prior; the calendar can be edited at any time
  popularity  log-count of purchases in the last 8 weeks of data
  reviews     Bayesian-shrunk rating + VADER sentiment of review text per item
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

from .season import LEARN_MAP, SEASONS, SeasonCalendar, prior_matrix
from .simulate import article_text

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    _SIA = SentimentIntensityAnalyzer()

    def sentiment(text: str) -> float:          # -1..1
        return _SIA.polarity_scores(text)["compound"]
except ImportError:                               # tiny fallback so the app never breaks
    _POS = {"love", "great", "perfect", "comfortable", "soft", "recommend", "lovely", "good", "nice"}
    _NEG = {"poor", "terrible", "cheap", "disappointed", "bad", "shrank", "worst", "awful"}

    def sentiment(text: str) -> float:
        w = text.lower().replace(",", " ").replace(".", " ").split()
        s = sum(x in _POS for x in w) - sum(x in _NEG for x in w)
        return float(np.tanh(s / 2))

COMPONENTS = ["collab", "content", "season", "popularity", "reviews"]
DEFAULT_WEIGHTS = {"collab": 0.30, "content": 0.25, "season": 0.25, "popularity": 0.10, "reviews": 0.10}
_EMPTY_T = np.array([], dtype="datetime64[ns]")


def _mm(x: np.ndarray) -> np.ndarray:
    lo, hi = x.min(), x.max()
    return (x - lo) / (hi - lo) if hi > lo else np.zeros_like(x, dtype=float)


class HybridRecommender:
    def __init__(self, calendar: SeasonCalendar, learn_calendar: SeasonCalendar | None = None,
                 n_factors: int = 32, seed: int = 42):
        self.calendar = calendar                          # used for *serving*
        self.learn_calendar = learn_calendar or calendar  # used to bucket history when learning demand
        self.n_factors, self.seed = n_factors, seed

    # ------------------------------------------------------------------ training
    def fit(self, articles: pd.DataFrame, tx: pd.DataFrame, reviews: pd.DataFrame | None = None):
        self.articles = articles.drop_duplicates("article_id").reset_index(drop=True)
        self.aid2idx = pd.Series(np.arange(len(self.articles)), index=self.articles.article_id.values)
        tx = tx[tx.article_id.isin(self.aid2idx.index)].copy()
        tx["t_dat"] = pd.to_datetime(tx["t_dat"])
        tx["aidx"] = tx.article_id.map(self.aid2idx).astype(int)
        self.tx, self.max_date = tx, tx.t_dat.max()
        self.text = article_text(self.articles)
        self._fit_content()
        self._fit_cf()
        self._fit_popularity()
        self._fit_season()
        self._build_history(reviews)
        self.set_reviews(reviews)
        return self

    def _fit_content(self):
        self.tfidf = TfidfVectorizer(max_features=5000, stop_words="english", ngram_range=(1, 2), min_df=2)
        self.X = self.tfidf.fit_transform(self.text).tocsr()

    def _fit_cf(self):
        uidx, _ = pd.factorize(self.tx.customer_id)
        n = len(self.articles)
        M = sp.coo_matrix((np.ones(len(self.tx)), (uidx, self.tx.aidx.values)), shape=(uidx.max() + 1, n)).tocsr()
        M.data = np.log1p(M.data)
        k = max(2, min(self.n_factors, min(M.shape) - 1))
        self.V = TruncatedSVD(k, random_state=self.seed).fit(M).components_.T      # (n_items, k)

    def _fit_popularity(self):
        recent = self.tx[self.tx.t_dat > self.max_date - pd.Timedelta(days=56)]
        recent = recent if len(recent) else self.tx
        c = np.bincount(recent.aidx.values, minlength=len(self.articles))
        self.pop = _mm(np.log1p(c).astype(float))
        self.pop_all = _mm(np.log1p(np.bincount(self.tx.aidx.values, minlength=len(self.articles))).astype(float))

    def _fit_season(self):
        self.prior = prior_matrix(self.text.values)                                  # (n, 5)
        ptype_idx, uniq = pd.factorize(self.articles.product_type_name.fillna("unknown").values)
        s_idx = pd.Categorical(self.learn_calendar.seasons_of(self.tx.t_dat), categories=SEASONS).codes
        ct = np.zeros((len(uniq), len(SEASONS)))
        np.add.at(ct, (ptype_idx[self.tx.aidx.values], s_idx), 1)
        vol, T = ct.sum(0), len(uniq)
        share = (ct + 0.5) / (vol + 0.5 * T)                                         # share of season's sales
        overall = (ct.sum(1) + 0.5 * len(SEASONS)) / (vol.sum() + 0.5 * len(SEASONS) * T)
        lift = share / overall[:, None]                                              # >1 = over-indexes in season
        learned_t = lift ** 2 / (1 + lift ** 2)
        n_t = ct.sum(1)
        self.learned_seasons = [s for s, v in zip(SEASONS, vol) if v >= 50]
        self.learned, self.alpha = learned_t[ptype_idx], (n_t / (n_t + 200))[ptype_idx]
        self.season_table = pd.DataFrame(lift, index=uniq, columns=SEASONS)
        self.season_table["n_sales"] = n_t

    def _season_vec(self, season: str) -> np.ndarray:
        j = SEASONS.index(season)
        natives = [season] if season in self.learned_seasons else [s for s in LEARN_MAP[season] if s in self.learned_seasons]
        if not natives:
            return self.prior[:, j]
        L = np.mean([self.learned[:, SEASONS.index(s)] for s in natives], axis=0)
        return self.alpha * L + (1 - self.alpha) * self.prior[:, j]

    def _build_history(self, reviews):
        self.hist, self.disliked = {}, {}
        for u, g in self.tx.groupby("customer_id", sort=False):
            self.hist[u] = (g.aidx.values.copy(), g.t_dat.values.astype("datetime64[ns]"), np.ones(len(g)))
        if reviews is not None:
            for r in reviews.itertuples():
                self._review_to_history(r.customer_id, r.article_id, r.rating, r.date)

    # ------------------------------------------------------------------ live updates (no retraining)
    def _add_hist(self, user, idx, date, weight):
        i, t, w = self.hist.get(user, (np.array([], dtype=int), _EMPTY_T, np.array([])))
        self.hist[user] = (np.append(i, idx), np.append(t, pd.Timestamp(date).to_datetime64()), np.append(w, weight))

    def _review_to_history(self, user, aid, rating, date):
        if aid not in self.aid2idx.index:
            return
        idx = int(self.aid2idx[aid])
        if rating >= 4:
            self._add_hist(user, idx, date, 0.5)       # a liked item = positive signal
        elif rating <= 2:
            self.disliked.setdefault(user, set()).add(idx)

    def add_purchase(self, user, aid, date):
        if aid in self.aid2idx.index:
            self._add_hist(user, int(self.aid2idx[aid]), date, 1.0)

    def add_review(self, user, aid, rating, text, date):
        row = pd.DataFrame([dict(customer_id=user, article_id=aid, rating=int(rating), text=text, date=pd.Timestamp(date))])
        self.set_reviews(pd.concat([self.reviews, row], ignore_index=True))
        self._review_to_history(user, aid, rating, date)

    def set_reviews(self, reviews: pd.DataFrame | None):
        self.reviews = reviews.copy() if reviews is not None and len(reviews) else pd.DataFrame(
            columns=["customer_id", "article_id", "rating", "text", "date"])
        score = np.full(len(self.articles), 0.5)
        if len(self.reviews):
            r = self.reviews.copy()
            r["idx"] = r.article_id.map(self.aid2idx)
            r = r.dropna(subset=["idx"])
            r["rating"] = r.rating.astype(float)
            sent = r.text.fillna("").map(lambda t: (sentiment(t) + 1) / 2 if str(t).strip() else np.nan)
            rn = (r.rating - 1) / 4
            r["s"] = np.where(sent.isna(), rn, 0.7 * rn + 0.3 * sent)
            g = r.groupby("idx").s.agg(["mean", "count"])
            m = 3                                                              # Bayesian shrinkage strength
            score[g.index.astype(int)] = ((g["mean"] * g["count"] + 0.5 * m) / (g["count"] + m)).values
        self.review_score = score

    # ------------------------------------------------------------------ inference
    def components(self, user_id=None, liked_ids=(), date=None):
        date = pd.Timestamp(date) if date is not None else self.max_date
        season = self.calendar.season_of(date)
        n = len(self.articles)
        idx, ts, w = self.hist.get(user_id, (np.array([], dtype=int), _EMPTY_T, np.array([])))
        liked = [int(self.aid2idx[a]) for a in liked_ids if a in self.aid2idx.index]
        if liked:
            idx = np.concatenate([idx, liked]).astype(int)
            ts = np.concatenate([ts, np.full(len(liked), date.to_datetime64(), dtype="datetime64[ns]")])
            w = np.concatenate([w, np.full(len(liked), 1.5)])
        collab = content = np.zeros(n)
        if len(idx):
            age = np.clip((date.to_datetime64() - ts) / np.timedelta64(1, "D"), 0, None)
            profile = self.X[idx].T.dot(w * np.exp(-age / 150))                  # recency-weighted taste
            content = _mm(self.X.dot(profile))
            x = np.zeros(n)
            np.add.at(x, idx, w)
            collab = _mm(self.V @ (np.log1p(x) @ self.V))                        # fold-in user, score items
        # season fit x overall demand = 'seasonal popularity' (avoids ranking obscure in-season items first)
        seasonal = _mm(self._season_vec(season) ** 2 * (0.2 + self.pop_all))
        comp = {"collab": collab, "content": content, "season": seasonal,
                "popularity": self.pop, "reviews": self.review_score}
        seen = set(idx.tolist()) | self.disliked.get(user_id, set())
        return comp, season, seen

    def rank(self, comp, weights=None, mask=None, k=12):
        wts = {**DEFAULT_WEIGHTS, **(weights or {})}
        tot = sum(wts.values()) or 1.0
        contrib = {c: wts[c] / tot * comp[c] for c in COMPONENTS}
        score = sum(contrib.values())
        if mask is not None:
            score = np.where(mask, score, -np.inf)
        k = min(k, int(np.isfinite(score).sum()))
        top = np.argpartition(-score, k - 1)[:k] if k else np.array([], dtype=int)
        return top[np.argsort(-score[top])], score, contrib

    def recommend(self, user_id=None, liked_ids=(), date=None, k=12, weights=None, filters=None, exclude_seen=True):
        comp, season, seen = self.components(user_id, liked_ids, date)
        mask = np.ones(len(self.articles), dtype=bool)
        if exclude_seen and seen:
            mask[list(seen)] = False
        for col, vals in (filters or {}).items():
            if vals:
                mask &= self.articles[col].isin(vals).values
        top, score, contrib = self.rank(comp, weights, mask, k)
        out = self.articles.iloc[top].copy()
        out["score"] = score[top]
        for c in COMPONENTS:
            out[f"{c}_pts"] = contrib[c][top]
        names = {"collab": "Similar shoppers bought it", "content": "Matches your style",
                 "season": f"Great for {season}", "popularity": "Trending now", "reviews": "Highly reviewed"}
        out["why"] = out[[f"{c}_pts" for c in COMPONENTS]].idxmax(axis=1).str.replace("_pts", "").map(names)
        return out.reset_index(drop=True), season
