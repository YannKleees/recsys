"""
/**
 * predict.py
 *
 * Load a trained engine and produce top-K recommendations for a user.
 *
 * Example:
 *   python recsys/individual/predict.py \
 *     --model models/cb.pkl \
 *     --user ladispoli \
 *     --k 10 \
 *     --out recs_ladispoli.csv
 */
"""
from __future__ import annotations
import argparse, pandas as pd
from .utils.common import load_pickle

def main():
    """
    /** CLI entry point. */"""
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Path to a pickled RecEngine")
    ap.add_argument("--user", required=True, help="User ID to recommend for")
    ap.add_argument("--k", type=int, default=10, help="Number of items to return")
    ap.add_argument("--keep-seen", action="store_true", help="Do not filter out items the user has already rated")
    ap.add_argument("--out", default=None, help="Optional CSV path to write recommendations")
    args = ap.parse_args()

    engine = load_pickle(args.model)
    recs = engine.recommend(user=args.user, n=args.k, filter_seen_items=not args.keep_seen)
    if args.out:
        recs.to_csv(args.out, index=False)
    else:
        print(recs.to_csv(index=False))

if __name__ == "__main__":
    main()
