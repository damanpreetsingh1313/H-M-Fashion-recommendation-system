"""Season calendar (fully editable) + keyword prior for how 'warm' a fashion item is."""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

# Cyclic order: Winter -> Spring -> Summer -> Monsoon -> Autumn -> Winter
SEASONS = ["Winter", "Spring", "Summer", "Monsoon", "Autumn"]
WARMTH = {"Winter": 0.0, "Spring": 0.5, "Summer": 1.0, "Monsoon": 0.65, "Autumn": 0.3}

# If a season was not observed in the training data, borrow demand patterns from these
LEARN_MAP = {"Winter": ["Winter"], "Spring": ["Spring"], "Summer": ["Summer"],
             "Monsoon": ["Summer", "Autumn"], "Autumn": ["Autumn"]}

NORTH = {12: "Winter", 1: "Winter", 2: "Winter", 3: "Spring", 4: "Spring", 5: "Spring",
         6: "Summer", 7: "Summer", 8: "Summer", 9: "Autumn", 10: "Autumn", 11: "Autumn"}
SOUTH = {12: "Summer", 1: "Summer", 2: "Summer", 3: "Autumn", 4: "Autumn", 5: "Autumn",
         6: "Winter", 7: "Winter", 8: "Winter", 9: "Spring", 10: "Spring", 11: "Spring"}
INDIA = {12: "Winter", 1: "Winter", 2: "Spring", 3: "Summer", 4: "Summer", 5: "Summer",
         6: "Summer", 7: "Monsoon", 8: "Monsoon", 9: "Monsoon", 10: "Autumn", 11: "Autumn"}

PRESETS = {
    "India (long summer, short winter)": INDIA,
    "Northern Hemisphere (Europe / US)": NORTH,
    "Southern Hemisphere": SOUTH,
}


@dataclass
class SeasonCalendar:
    month_map: dict
    name: str = "custom"

    def season_of(self, date) -> str:
        return self.month_map[pd.Timestamp(date).month]

    def seasons_of(self, dates: pd.Series) -> pd.Series:
        return pd.to_datetime(dates).dt.month.map(self.month_map)

    def lengths(self) -> dict:
        return pd.Series(self.month_map).value_counts().reindex(SEASONS, fill_value=0).to_dict()

    def key(self) -> tuple:
        return tuple(sorted(self.month_map.items()))

    @classmethod
    def from_lengths(cls, winter_start_month: int, lengths: dict, name: str = "custom"):
        """e.g. winter_start_month=12, lengths={'Winter':2,'Spring':1,'Summer':4,'Monsoon':3,'Autumn':2}"""
        if sum(lengths.values()) != 12:
            raise ValueError("Season lengths must add up to 12 months")
        month_map, month = {}, winter_start_month
        for s in SEASONS:
            for _ in range(int(lengths.get(s, 0))):
                month_map[month] = s
                month = month % 12 + 1
        return cls(month_map, name)


# ------------------------------------------------------------------ item warmth prior
# 0 = very cold-weather item, 1 = very hot-weather item
KEYWORD_WARMTH = {
    "coat": 0.0, "parka": 0.0, "puffer": 0.0, "fleece": 0.0, "beanie": 0.0, "scarf": 0.0,
    "glove": 0.0, "thermal": 0.0, "wool": 0.05, "sweater": 0.15, "jumper": 0.15, "boot": 0.15,
    "hoodie": 0.2, "knit": 0.2, "tights": 0.2, "cardigan": 0.3, "jacket": 0.3, "long sleeve": 0.4,
    "skirt": 0.8, "dress": 0.75, "t-shirt": 0.85, "crop": 0.85, "tank": 0.95, "vest top": 0.95,
    "sleeveless": 0.95, "linen": 0.95, "shorts": 1.0, "sandal": 1.0, "swim": 1.0, "bikini": 1.0,
    "sunglasses": 1.0, "sun hat": 1.0, "flip": 1.0,
}
_PATS = {k: re.compile(r"\b" + re.escape(k)) for k in KEYWORD_WARMTH}
_RAIN = re.compile(r"\b(rain|waterproof|windbreaker|anorak|poncho)")


def prior_matrix(texts) -> np.ndarray:
    """(n_items x 5) knowledge-based season fit. Neutral items get 0.5 everywhere."""
    n = len(texts)
    warmth = np.full(n, np.nan)
    rain = np.zeros(n, dtype=bool)
    for i, t in enumerate(texts):
        vals = [w for k, w in KEYWORD_WARMTH.items() if _PATS[k].search(t)]
        if vals:
            warmth[i] = np.mean(vals)
        rain[i] = bool(_RAIN.search(t))
    P = np.full((n, len(SEASONS)), 0.5)
    has = ~np.isnan(warmth)
    for j, s in enumerate(SEASONS):
        P[has, j] = 1 - np.abs(warmth[has] - WARMTH[s])
    P[rain, SEASONS.index("Monsoon")] = 1.0
    return P
