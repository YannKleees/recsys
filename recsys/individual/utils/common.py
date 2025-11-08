# recsys/individual/utils/common.py
"""
Utilities for loading filtered CSV shards, building histories, and model I/O.
Now memory-aware: categorical codes for user/item and float32 ratings; optionally
skip text/props columns.
"""
from __future__ import annotations
import os, pickle, glob
from typing import Dict, Optional, Set, List, Tuple
import pandas as pd

def _mem_str(bytes_cnt: int) -> str:
    for unit in ["B","KB","MB","GB","TB"]:
        if bytes_cnt < 1024.0:
            return f"{bytes_cnt:,.1f} {unit}"
        bytes_cnt /= 1024.0
    return f"{bytes_cnt:.1f} PB"

def load_interactions_csv(path: str,
                          limit: Optional[int] = None,
                          include_text: bool = True,
                          include_props: bool = True,
                          make_categorical: bool = True,
                          verbose: bool = False) -> pd.DataFrame:
    """
    Load interactions from a single CSV file or a directory of CSV shards.

    Returned columns (subset depending on flags and file contents):
      user (str/category), item (str/category), rating (float32), [text], [props]

    Args:
      path: file or directory of CSV shards
      limit: stop after reading this many rows across files
      include_text/include_props: include those columns if present
      make_categorical: convert user/item to pandas 'category'
      verbose: print memory usage

    Note: We drop rows missing user/item/rating.
    """
    # determine which columns to request
    base_cols = ["user","item","rating"]
    opt_cols = []
    if include_text: opt_cols.append("text")
    if include_props: opt_cols.append("props")
    usecols = base_cols + opt_cols

    paths: List[str] = []
    if os.path.isdir(path):
        paths = sorted(glob.glob(os.path.join(path, "*.csv")))
    else:
        paths = [path]

    frames: List[pd.DataFrame] = []
    remaining = None if limit is None else int(limit)

    for p in paths:
        if remaining is not None and remaining <= 0:
            break
        # read at most 'remaining' rows from this file
        nrows = None if remaining is None else max(0, remaining)
        if nrows == 0:
            break
        df = pd.read_csv(p, usecols=[c for c in usecols if c], nrows=nrows, dtype={"rating":"float32"})
        # enforce expected types
        if "user" in df.columns: df["user"] = df["user"].astype("string")
        if "item" in df.columns: df["item"] = df["item"].astype("string")
        frames.append(df)
        if remaining is not None:
            remaining -= len(df)

    if not frames:
        return pd.DataFrame(columns=[c for c in usecols])

    df = pd.concat(frames, ignore_index=True)

    # core subset + drop invalid rows
    cols = [c for c in usecols if c in df.columns]
    df = df[cols].dropna(subset=["user","item","rating"])

    if make_categorical:
        # categories shrink memory drastically; keep 'string' arrow for safety then category
        df["user"] = df["user"].astype("category")
        df["item"] = df["item"].astype("category")

    if verbose:
        mem = df.memory_usage(deep=True).sum()
        print(f"[load_interactions_csv] rows={len(df):,} cols={list(df.columns)} mem={_mem_str(mem)}")

    return df

def build_user_hist(ratings: pd.DataFrame) -> Dict[object, Set[object]]:
    """Map user -> set(items)."""
    return ratings.groupby("user")["item"].apply(set).to_dict()

def filter_seen(recs: pd.DataFrame, seen_items: Set[object]) -> pd.DataFrame:
    """Remove already-seen items from a recommendations DataFrame."""
    if recs is None or recs.empty:
        return recs
    return recs[~recs["item"].isin(seen_items)].reset_index(drop=True)

def save_pickle(obj, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)

def load_pickle(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)
