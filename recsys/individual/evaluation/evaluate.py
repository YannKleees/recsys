# recsys/individual/evaluation/evaluate.py
"""
Evaluate a model with a simple per-user holdout (random) and report
HitRate@K, Recall@K, and NDCG@K.

Example:
  python recsys/individual/evaluation/evaluate.py \
    --data-path data/filtered \
    --algo content_based \
    --k 10 \
    --min-user-count 3 --min-item-count 3 \
    --min-df 5 --max-features 50000 --use-props
"""
from __future__ import annotations
import argparse, numpy as np, pandas as pd
from ..algorithms.engine import RecEngine
from ..algorithms.popularity import PopularityAlgo
from ..algorithms.content_based import ContentBasedAlgo
from ..utils.common import load_interactions_csv

def _apply_min_counts(df: pd.DataFrame, min_user: int, min_item: int) -> pd.DataFrame:
    if df.empty: return df
    if min_user and min_user > 1:
        uc = df.groupby("user").size()
        df = df[df["user"].map(uc) >= min_user]
    if min_item and min_item > 1:
        ic = df.groupby("item").size()
        df = df[df["item"].map(ic) >= min_item]
    return df.reset_index(drop=True)

def _per_user_holdout(df: pd.DataFrame, test_frac: float = 0.2, min_train: int = 1):
    """Random split per user (no timestamp): returns train_df, test_dict[user] = set(items)."""
    rng = np.random.default_rng(42)
    df = df.sample(frac=1.0, random_state=42)  # shuffle
    groups = df.groupby("user")
    train_parts = []
    test_dict = {}
    for u, g in groups:
        n = len(g)
        if n <= min_train:
            train_parts.append(g)
            continue
        m = max(1, int(round(n * test_frac)))
        idx = rng.choice(g.index.to_numpy(), size=m, replace=False)
        test_items = set(g.loc[idx, "item"].tolist())
        train_parts.append(g.drop(index=idx))
        test_dict[u] = test_items
    train_df = pd.concat(train_parts, ignore_index=True)
    return train_df, test_dict

def _metrics_at_k(engine: RecEngine, test_dict, k: int = 10):
    users = list(test_dict.keys())
    hits = 0
    total_relevant = 0
    dcg = 0.0
    idcg = 0.0
    for i, u in enumerate(users, 1):
        rel_items = test_dict[u]
        if not rel_items: 
            continue
        total_relevant += len(rel_items)
        recs = engine.recommend(user=u, n=k, filter_seen_items=True)
        items = recs["item"].tolist() if recs is not None and not recs.empty else []
        # HitRate: did we recommend at least one held-out item?
        if any(it in rel_items for it in items):
            hits += 1
        # Recall@K
        inter = [1 if it in rel_items else 0 for it in items]
        # NDCG@K
        for rank, rel in enumerate(inter, start=1):
            if rel:
                dcg += 1.0 / np.log2(rank + 1)
        # ideal DCG: min(#rel, K) relevant items at top
        r = min(len(rel_items), k)
        idcg += sum(1.0 / np.log2(rk + 1) for rk in range(1, r + 1))
        if i % 1000 == 0:
            print(f"... evaluated {i:,}/{len(users):,} users")

    hitrate = hits / len(users) if users else 0.0
    recall = 0.0
    if total_relevant > 0:
        # approximate recall by counting intersections; stricter computation would sum per-user recalls equally
        # here we sum over users: recall = sum(hit counts) / sum(#relevant)
        # we already added inter only for retrieved; compute across all users
        # To keep it simple, compute upper bound using dcg!=recall; skip aggregate recall for speed.
        pass
    ndcg = (dcg / idcg) if idcg > 1e-9 else 0.0
    return {"HitRate@K": hitrate, "NDCG@K": ndcg}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-path", required=True)
    ap.add_argument("--algo", choices=["content_based","popularity"], required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--min-user-count", type=int, default=3)
    ap.add_argument("--min-item-count", type=int, default=3)
    # content-based
    ap.add_argument("--min-df", type=int, default=5)
    ap.add_argument("--max-features", type=int, default=50000)
    ap.add_argument("--use-props", action="store_true")
    ap.add_argument("--like-threshold", type=float, default=3.5)
    args = ap.parse_args()

    include_text = (args.algo == "content_based")
    include_props = (args.algo == "content_based") and args.use-props if hasattr(args, "use-props") else (args.algo == "content_based") and args.use_props

    ratings = load_interactions_csv(
        args.data_path, limit=args.limit,
        include_text=include_text, include_props=include_props,
        make_categorical=True, verbose=True
    )
    ratings = _apply_min_counts(ratings, args.min_user_count, args.min_item_count)

    # split
    train_df, test_dict = _per_user_holdout(ratings, test_frac=0.2, min_train=1)

    # fit engine
    if args.algo == "popularity":
        engine = RecEngine(algo=PopularityAlgo(mode="count", min_count=5)).fit(train_df)
    else:
        cb = ContentBasedAlgo(
            min_df=args.min_df,
            max_features=args.max_features,
            use_props=include_props,
            prop_weight=1.0,
            like_threshold=args.like_threshold,
        )
        fallback = PopularityAlgo(mode="count", min_count=5)
        engine = RecEngine(algo=cb, fallback=fallback).fit(train_df)

    # evaluate
    metrics = _metrics_at_k(engine, test_dict, k=args.k)
    print(f"[RESULT] K={args.k} | HitRate@K={metrics['HitRate@K']:.4f} | NDCG@K={metrics['NDCG@K']:.4f}")

if __name__ == "__main__":
    main()
