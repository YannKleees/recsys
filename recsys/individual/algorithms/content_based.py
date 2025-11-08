"""
/**
 * content_based.py
 *
 * TF-IDF over review text + optional numeric hotel properties.
 * Builds item vectors and derives a user profile by aggregating vectors of items
 * the user rated positively. Scores are cosine similarities.
 *
 * Dependencies:
 *  - numpy, pandas, scipy, scikit-learn
 */
"""
from __future__ import annotations
import ast
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize as sk_normalize
from .base import BaseAlgo

PROP_KEYS = ["sleep quality","value","rooms","service","cleanliness","location"]

class ContentBasedAlgo(BaseAlgo):
    """
    /**
     * TF-IDF + properties content-based recommender.
     *
     * Arguments:
     *  - min_df: minimal document frequency for TF-IDF
     *  - max_features: cap on vocabulary size
     *  - use_props: whether to incorporate numeric property features
     *  - prop_weight: multiplicative weight for the properties block
     *  - like_threshold: ratings above this threshold contribute positively
     */"""

    def __init__(
        self,
        min_df: int = 5,
        max_features: int = 50000,
        use_props: bool = True,
        prop_weight: float = 1.0,
        like_threshold: float = 3.5
    ):
        """
        /** Store configuration and initialize fields. */"""
        self.min_df = min_df
        self.max_features = max_features
        self.use_props = use_props
        self.prop_weight = prop_weight
        self.like_threshold = like_threshold

        self.vectorizer_: Optional[TfidfVectorizer] = None
        self.item_matrix_: Optional[sp.csr_matrix] = None
        self.item_index_: Dict[str, int] = {}
        self.items_: List[str] = []
        self.user_items_: Dict[str, List[Tuple[str, float]]] = {}
        self.has_text_: bool = False
        self.has_props_: bool = False

    def _build_item_texts(self, ratings: pd.DataFrame) -> pd.Series:
        """
        /** Produce a per-item text by taking the first non-empty review text per item. */"""
        if "text" not in ratings.columns:
            return pd.Series(dtype=str)
        tmp = ratings[["item", "text"]].copy()
        tmp["text"] = tmp["text"].fillna("").astype(str)
        tmp = tmp[tmp["text"].str.strip() != ""]
        return tmp.groupby("item")["text"].first()

    def _parse_props_column(self, ratings: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        /** Parse the 'props' column if present (stringified dict) and return per-item mean features. */"""
        if not self.use_props or "props" not in ratings.columns:
            return None
        df = ratings[["item", "props"]].dropna().copy()
        if df.empty:
            return None
        def to_vec(s) -> Optional[List[float]]:
            try:
                d = ast.literal_eval(s) if isinstance(s, str) else s
            except Exception:
                return None
            if not isinstance(d, dict):
                return None
            out = []
            for k in PROP_KEYS:
                v = d.get(k, None)
                try:
                    out.append(float(v) / 5.0 if v is not None else np.nan)
                except Exception:
                    out.append(np.nan)
            return out
        vecs = df["props"].apply(to_vec)
        mask = vecs.apply(lambda x: isinstance(x, list))
        df = df[mask].copy()
        if df.empty:
            return None
        mat = np.vstack(vecs[mask].to_list())
        props_df = pd.DataFrame(mat, columns=[f"prop_{k.replace(' ','_')}" for k in PROP_KEYS])
        props_df["item"] = df["item"].values
        props_item = props_df.groupby("item").mean(numeric_only=True)
        return props_item

    def fit(self, ratings: pd.DataFrame) -> "ContentBasedAlgo":
        """
        /** Train the content-based model:
         *   1) build item-level text and properties representations
         *   2) combine them into a single sparse item matrix
         *   3) cache user→(item,rating) lists for profile construction
         */"""
        item_texts = self._build_item_texts(ratings)
        self.has_text_ = item_texts.shape[0] > 0

        props_item = self._parse_props_column(ratings)
        self.has_props_ = props_item is not None and props_item.shape[0] > 0

        items_from_text = set(item_texts.index) if self.has_text_ else set()
        items_from_props = set(props_item.index) if self.has_props_ else set()
        all_items = sorted(list(items_from_text | items_from_props | set(ratings["item"].unique())))
        self.items_ = all_items
        self.item_index_ = {it: idx for idx, it in enumerate(self.items_)}

        blocks = []
        if self.has_text_:
            texts = [item_texts.get(it, "") for it in self.items_]
            self.vectorizer_ = TfidfVectorizer(min_df=self.min_df, max_features=self.max_features)
            X_text = self.vectorizer_.fit_transform(texts)
            blocks.append(X_text)
        if self.has_props_:
            P = props_item.reindex(self.items_).fillna(0.0).to_numpy(dtype=np.float32)
            X_props = sp.csr_matrix(P) * float(self.prop_weight)
            blocks.append(X_props)

        if not blocks:
            raise ValueError("ContentBasedAlgo: neither text nor properties available to build item vectors.")

        self.item_matrix_ = sp.hstack(blocks, format="csr")
        interactions = ratings[["user", "item", "rating"]].dropna()
        self.user_items_ = (
            interactions.groupby("user")
            .apply(lambda g: list(zip(g["item"].astype(str), g["rating"].astype(float))))
            .to_dict()
        )
        return self

    def _user_profile(self, user: str) -> Optional[sp.csr_matrix]:
        """
        /** Create a normalized user profile by weighted sum of liked item vectors. */"""
        if user not in self.user_items_:
            return None
        liked = []
        weights = []
        for it, r in self.user_items_[user]:
            w = max(0.0, float(r) - float(self.like_threshold))
            if w <= 0.0:
                continue
            idx = self.item_index_.get(it, None)
            if idx is None:
                continue
            liked.append(idx)
            weights.append(w)
        if not liked:
            return None
        V = self.item_matrix_[liked, :]
        w = np.array(weights, dtype=np.float32)
        prof = sp.csr_matrix(w.reshape(1, -1)) @ V
        prof = sk_normalize(prof, norm="l2", copy=False)
        return prof

    def predict_for_user(self, user, candidates=None, n: int = 10) -> pd.DataFrame:
        """
        /** Score items for a user by cosine similarity between item vectors and the user's profile. */"""
        import numpy as np  # local to keep import set minimal at module import time
        if self.item_matrix_ is None:
            raise RuntimeError("ContentBasedAlgo must be fitted before prediction.")
        prof = self._user_profile(user)
        if prof is None:
            return pd.DataFrame(columns=["item", "score"])
        if candidates is None:
            scores = self.item_matrix_.dot(prof.T).toarray().ravel()
            idx = np.argsort(-scores)[:n]
            items = [self.items_[i] for i in idx]
            vals = scores[idx]
            return pd.DataFrame({"item": items, "score": vals})
        cand_idx = [self.item_index_.get(it) for it in candidates]
        cand_pairs = [(i, it) for i, it in zip(cand_idx, candidates) if i is not None]
        if not cand_pairs:
            return pd.DataFrame(columns=["item", "score"])
        rows = [i for i, _ in cand_pairs]
        sub = self.item_matrix_[rows, :]
        scores = sub.dot(prof.T).toarray().ravel()
        items = [it for _, it in cand_pairs]
        out = pd.DataFrame({"item": items, "score": scores})
        out = out.sort_values("score", ascending=False).head(n).reset_index(drop=True)
        return out
