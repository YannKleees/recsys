# recsys/individual/algorithms/train.py
"""
Train an individual recommendation model using filtered CSV shards.
Supports "content_based" and "popularity" algorithms.

Examples:
  Content-based + popularity fallback:
    python recsys/individual/algorithms/train.py \
      --data-path data/filtered \
      --algo content_based \
      --model-out models/cb.pkl \
      --min-user-count 3 --min-item-count 3 \
      --min-df 5 --max-features 50000 --use-props --prop-weight 1.0 --like-threshold 3.5

  Popularity baseline:
    python recsys/individual/algorithms/train.py \
      --data-path data/filtered \
      --algo popularity \
      --model-out models/pop.pkl \
      --mode count --min-count 5 --min-user-count 3 --min-item-count 3
"""
from __future__ import annotations
import argparse, pandas as pd
from .engine import RecEngine
from .popularity import PopularityAlgo
from .content_based import ContentBasedAlgo
from ..utils.common import load_interactions_csv, save_pickle

def _apply_min_counts(df: pd.DataFrame, min_user: int, min_item: int) -> pd.DataFrame:
    """Filter users/items by minimum interaction counts."""
    if df.empty: return df
    if min_user and min_user > 1:
        uc = df.groupby("user").size()
        df = df[df["user"].map(uc) >= min_user]
    if min_item and min_item > 1:
        ic = df.groupby("item").size()
        df = df[df["item"].map(ic) >= min_item]
    return df.reset_index(drop=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-path", required=True, help="Path to a filtered CSV file or directory of CSV shards")
    ap.add_argument("--algo", choices=["content_based","popularity"], required=True, help="Algorithm to train")
    ap.add_argument("--model-out", required=True, help="Output path for the trained engine pickle")
    ap.add_argument("--limit", type=int, default=None, help="Only read the first N rows")
    # popularity options
    ap.add_argument("--mode", choices=["count","mean"], default="count", help="Popularity mode when algo=popularity")
    ap.add_argument("--min-count", type=int, default=1, help="Min item count when algo=popularity")
    # content-based options
    ap.add_argument("--min-df", type=int, default=5, help="TF-IDF min_df when algo=content_based")
    ap.add_argument("--max-features", type=int, default=50000, help="TF-IDF max_features when algo=content_based")
    ap.add_argument("--use-props", action="store_true", help="Use numeric properties block when algo=content_based")
    ap.add_argument("--prop-weight", type=float, default=1.0, help="Scale factor for properties block")
    ap.add_argument("--like-threshold", type=float, default=3.5, help="Ratings above this contribute to user profile")
    # global count filters
    ap.add_argument("--min-user-count", type=int, default=3, help="Drop users with < this many reviews")
    ap.add_argument("--min-item-count", type=int, default=3, help="Drop items with < this many reviews")
    # memory options
    ap.add_argument("--no-text", action="store_true", help="Ignore text column when loading (saves memory)")
    ap.add_argument("--no-props", action="store_true", help="Ignore props column when loading (saves memory)")
    ap.add_argument("--verbose-load", action="store_true", help="Print memory usage after loading")
    args = ap.parse_args()

    include_text = (args.algo == "content_based") and (not args.no_text)
    include_props = (args.algo == "content_based") and args.use_props and (not args.no_props)

    ratings = load_interactions_csv(
        args.data_path,
        limit=args.limit,
        include_text=include_text,
        include_props=include_props,
        make_categorical=True,
        verbose=args.verbose_load,
    )

    # Apply min-count filters again in case data-path wasn't produced with those settings
    before = len(ratings)
    ratings = _apply_min_counts(ratings, args.min_user_count, args.min_item_count)
    after = len(ratings)
    if args.verbose_load:
        print(f"[train] interactions after ≥{args.min_user_count}/≥{args.min_item_count} filtering: {after:,} (from {before:,})")

    if args.algo == "popularity":
        algo = PopularityAlgo(mode=args.mode, min_count=args.min_count)
        engine = RecEngine(algo=algo, fallback=None).fit(ratings)
    else:
        cb = ContentBasedAlgo(
            min_df=args.min_df,
            max_features=args.max_features,
            use_props=args.use_props and include_props,
            prop_weight=args.prop_weight,
            like_threshold=args.like_threshold,
        )
        pop = PopularityAlgo(mode="count", min_count=5)
        engine = RecEngine(algo=cb, fallback=pop).fit(ratings)

    save_pickle(engine, args.model_out)
    print(f"[OK] Saved model to: {args.model_out}")

if __name__ == "__main__":
    main()
