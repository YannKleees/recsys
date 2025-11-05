"""
/**
 * train.py
 *
 * Train an individual recommendation model using filtered CSV shards.
 * Supports "content_based" and "popularity" algorithms.
 *
 * Example:
 *   python recsys/individual/train.py \
 *     --data-path data/filtered \
 *     --algo content_based \
 *     --model-out models/cb.pkl \
 *     --min-df 5 --max-features 50000 --use-props --prop-weight 1.0 --like-threshold 3.5
 *
 *   python recsys/individual/train.py \
 *     --data-path data/filtered \
 *     --algo popularity \
 *     --model-out models/pop.pkl \
 *     --mode count --min-count 5
 */
"""
from __future__ import annotations
import argparse, pandas as pd
from .engine import RecEngine
from .algorithms.popularity import PopularityAlgo
from .algorithms.content_based import ContentBasedAlgo
from .utils.common import load_interactions_csv, save_pickle

def main():
    """
    /** CLI entry point. */"""
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-path", required=True, help="Path to a filtered CSV file or directory of CSV shards")
    ap.add_argument("--algo", choices=["content_based","popularity"], required=True, help="Algorithm to train")
    ap.add_argument("--model-out", required=True, help="Output path for the trained engine pickle")
    ap.add_argument("--limit", type=int, default=None, help="Only read the first N rows")
    ap.add_argument("--mode", choices=["count","mean"], default="count", help="Popularity mode when algo=popularity")
    ap.add_argument("--min-count", type=int, default=1, help="Min item count when algo=popularity")
    ap.add_argument("--min-df", type=int, default=5, help="TF-IDF min_df when algo=content_based")
    ap.add_argument("--max-features", type=int, default=50000, help="TF-IDF max_features when algo=content_based")
    ap.add_argument("--use-props", action="store_true", help="Use numeric properties block when algo=content_based")
    ap.add_argument("--prop-weight", type=float, default=1.0, help="Scale factor for properties block")
    ap.add_argument("--like-threshold", type=float, default=3.5, help="Ratings above this contribute to user profile")
    args = ap.parse_args()

    ratings = load_interactions_csv(args.data_path, limit=args.limit)

    if args.algo == "popularity":
        algo = PopularityAlgo(mode=args.mode, min_count=args.min_count)
        engine = RecEngine(algo=algo, fallback=None).fit(ratings)
    else:
        cb = ContentBasedAlgo(
            min_df=args.min_df,
            max_features=args.max_features,
            use_props=args.use_props,
            prop_weight=args.prop_weight,
            like_threshold=args.like_threshold,
        )
        pop = PopularityAlgo(mode="count", min_count=5)
        engine = RecEngine(algo=cb, fallback=pop).fit(ratings)

    save_pickle(engine, args.model_out)

if __name__ == "__main__":
    main()
