"""
/**
 * common.py
 *
 * Utilities for loading filtered CSV shards, building histories, and model I/O.
 */
"""
from __future__ import annotations
import os, pickle, glob
from typing import Dict, Iterable, Optional, Set, Tuple, List
import pandas as pd

def load_interactions_csv(path: str, limit: Optional[int] = None) -> pd.DataFrame:
    """
    /**
     * Load interactions from a single CSV file or a directory of CSV shards.
     * Reads columns if present: user, item, rating, text, props.
     * If 'limit' is set, stop after reading that many rows across files.
     */"""
    paths: List[str] = []
    if os.path.isdir(path):
        paths = sorted(glob.glob(os.path.join(path, "*.csv")))
    else:
        paths = [path]
    frames: List[pd.DataFrame] = []
    remaining = None if limit is None else int(limit)
    for p in paths:
        nrows = None if remaining is None else max(0, remaining)
        if nrows == 0:
            break
        df = pd.read_csv(p, nrows=nrows)
        frames.append(df)
        if remaining is not None:
            remaining -= len(df)
    if not frames:
        return pd.DataFrame(columns=["user","item","rating"])
    df = pd.concat(frames, ignore_index=True)
    cols = [c for c in ["user","item","rating","text","props"] if c in df.columns]
    return df[cols].dropna(subset=["user","item","rating"])

def build_user_hist(ratings: pd.DataFrame) -> Dict[object, Set[object]]:
    """
    /**
     * Build a mapping user -> set(items) from the ratings DataFrame.
     */"""
    return ratings.groupby("user")["item"].apply(set).to_dict()

def filter_seen(recs: pd.DataFrame, seen_items: Set[object]) -> pd.DataFrame:
    """
    /**
     * Remove already-seen items from a recommendations DataFrame.
     */"""
    if recs is None or recs.empty:
        return recs
    return recs[~recs["item"].isin(seen_items)].reset_index(drop=True)

def save_pickle(obj, path: str) -> None:
    """
    /**
     * Serialize a Python object to disk using pickle.
     */"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)

def load_pickle(path: str):
    """
    /**
     * Load a pickled Python object from disk.
     */"""
    with open(path, "rb") as f:
        return pickle.load(f)
