"""
/**
 * popularity.py
 *
 * Popularity baseline with two modes:
 *  - "count": score = number of ratings per item
 *  - "mean":  score = average rating per item
 *
 * Parameters:
 *  - mode: "count" | "mean"
 *  - min_count: minimum ratings required for an item to be kept
 */
"""
from __future__ import annotations
import pandas as pd
from .base import BaseAlgo

class PopularityAlgo(BaseAlgo):
    """
    /**
     * Global, non-personalized baseline used for comparison and cold-start fallback.
     */"""

    def __init__(self, mode: str = "count", min_count: int = 1):
        """
        /** Store configuration. */"""
        self.mode = mode
        self.min_count = min_count
        self.item_scores_ = None

    def fit(self, ratings: pd.DataFrame) -> "PopularityAlgo":
        """
        /** Compute item-level popularity scores according to the chosen mode. */"""
        grp = ratings.groupby("item")["rating"]
        counts = grp.size().rename("count")
        means = grp.mean().rename("mean")
        items = pd.concat([counts, means], axis=1).reset_index()
        if self.mode == "count":
            items["score"] = items["count"]
        else:
            items["score"] = items["mean"]
        if self.min_count and self.min_count > 1:
            items = items[items["count"] >= self.min_count]
        self.item_scores_ = items[["item", "score"]].sort_values("score", ascending=False).reset_index(drop=True)
        return self

    def predict_for_user(self, user, candidates=None, n: int = 10) -> pd.DataFrame:
        """
        /** Return the top-n globally popular items, optionally filtered by candidates. */"""
        if self.item_scores_ is None:
            raise RuntimeError("PopularityAlgo must be fitted before prediction.")
        df = self.item_scores_
        if candidates is not None:
            df = df[df["item"].isin(set(candidates))]
        return df.head(n).copy()
