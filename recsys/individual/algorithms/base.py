"""
/**
 * base.py
 *
 * Minimal interface for individual recommendation algorithms.
 * Expected ratings DataFrame columns: user, item, rating (float), optional: text, props.
 */
"""
from __future__ import annotations
from abc import ABC, abstractmethod
import pandas as pd

class BaseAlgo(ABC):
    """
    /**
     * Abstract base for recommenders.
     *
     * Methods:
     *  - fit(ratings): train on interactions, return self
     *  - predict_for_user(user, candidates, n): return DataFrame[item, score] sorted desc
     *  - recommend(user, n, candidates): convenience top-n wrapper
     */
    """

    @abstractmethod
    def fit(self, ratings: pd.DataFrame) -> "BaseAlgo":
        """
        /** Train the model on the provided ratings DataFrame. */"""
        ...

    @abstractmethod
    def predict_for_user(self, user, candidates=None, n: int = 10) -> pd.DataFrame:
        """
        /** Score candidate items for a user and return a DataFrame with columns ['item','score']. */"""
        ...

    def recommend(self, user, n: int = 10, candidates=None) -> pd.DataFrame:
        """
        /** Return the top-n items from predict_for_user. */"""
        res = self.predict_for_user(user, candidates=candidates, n=n)
        return res.head(n) if isinstance(res, pd.DataFrame) else res
