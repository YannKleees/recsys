"""
/**
 * engine.py
 *
 * Thin wrapper that is instantiated AFTER choosing an algorithm.
 * Responsibilities:
 *  - fit(): train the main algorithm (and optional fallback)
 *  - recommend(): produce top-n, with optional seen-item filtering
 */
"""
from __future__ import annotations
import pandas as pd
from .base import BaseAlgo
from ..utils.common import build_user_hist, filter_seen

class RecEngine:
    """
    /**
     * Algorithm-agnostic engine.
     *
     * Parameters:
     *  - algo: main algorithm implementing BaseAlgo
     *  - fallback: optional BaseAlgo used for cold users
     */"""

    def __init__(self, algo: BaseAlgo, fallback: BaseAlgo | None = None):
        """
        /** Store components only. */"""
        self.algo = algo
        self.fallback = fallback
        self.user_hist = {}

    def fit(self, ratings: pd.DataFrame) -> "RecEngine":
        """
        /** Train the algorithm(s) and cache user histories for seen-item filtering. */"""
        self.user_hist = build_user_hist(ratings)
        self.algo.fit(ratings)
        if self.fallback is not None:
            self.fallback.fit(ratings)
        return self

    def recommend(self, user, n: int = 10, candidates=None, filter_seen_items: bool = True) -> pd.DataFrame:
        """
        /** Recommend top-n items with cold-start fallback and optional seen-item filtering. */"""
        is_cold = user not in self.user_hist or len(self.user_hist.get(user, ())) == 0
        model = self.fallback if (is_cold and self.fallback is not None) else self.algo
        recs = model.recommend(user, n=n, candidates=candidates)
        if filter_seen_items and user in self.user_hist and not recs.empty:
            recs = filter_seen(recs, self.user_hist[user])
        return recs.head(n)
