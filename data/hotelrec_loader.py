# data/hotelrec_loader.py
"""
Preprocess HotelRec by dropping ONLY rows where ALL THREE are missing:
  - rating
  - non-empty review text
  - any numeric sub-rating inside property_dict

THEN filter out all users with <min_user_count reviews and all items with
<min_item_count reviews (default: 3 and 3) computed on the kept rows.

This script streams the JSONL twice when min-count filters are enabled:
  - Pass 1: count users/items on rows that satisfy the keep_rule
  - Pass 2: write only rows whose user and item meet the thresholds

It also prints a compact sparsity analysis on the first --analysis-n KEPT rows.

Example:
  python data/hotelrec_loader.py --in data/HotelRec.txt --out-dir data/filtered \
    --shard-size 500000 --analysis-n 1000000 --min-user-count 3 --min-item-count 3
"""
from __future__ import annotations
import argparse, json, csv, os, math, hashlib, sys
from collections import Counter
from typing import Iterator, Optional, List

# HotelRec field names
F_USER = "author"
F_ITEM = "hotel_url"
F_RATING = "rating"
F_TEXT = "text"
F_PROPS = "property_dict"

PROP_KEYS = ["sleep quality","value","rooms","service","cleanliness","location"]

def safe_float(x) -> Optional[float]:
    try: return float(x)
    except Exception: return None

def nonempty_text(x) -> bool:
    return bool(str(x or "").strip())

def has_any_subrating(props) -> bool:
    if not isinstance(props, dict): return False
    return any(safe_float(props.get(k)) is not None for k in PROP_KEYS)

def keep_rule(rec: dict) -> bool:
    """Keep a row if NOT (all three are missing)."""
    r_ok = safe_float(rec.get(F_RATING)) is not None
    t_ok = nonempty_text(rec.get(F_TEXT))
    p_ok = has_any_subrating(rec.get(F_PROPS))
    return any([r_ok, t_ok, p_ok])

def fmt(n: int) -> str: return f"{n:,}"

def pct(sorted_vals: List[int], p: float) -> Optional[int]:
    if not sorted_vals: return None
    k = (len(sorted_vals) - 1) * (p / 100.0)
    a, b = int(k), int(math.ceil(k))
    if a == b: return sorted_vals[a]
    return int(round(sorted_vals[a] + (k - a) * (sorted_vals[b] - sorted_vals[a])))

def stream_jsonl(path: str, limit: Optional[int] = None) -> Iterator[dict]:
    n = 0
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if limit is not None and n >= limit: break
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            n += 1
            yield rec

def preprocess_and_analyze(in_path: str, out_dir: str, shard_size: int,
                           analysis_n: int, progress: int,
                           drop_text: bool, drop_props: bool,
                           read_limit: Optional[int],
                           min_user_count: int, min_item_count: int) -> None:
    os.makedirs(out_dir, exist_ok=True)

    # -------------------
    # Pass 1: Counting on rows that satisfy keep_rule
    # -------------------
    need_counts = (min_user_count or min_item_count)
    user_cnt, item_cnt = Counter(), Counter()
    scanned1 = kept1 = bad1 = 0

    if need_counts:
        print(f"[PASS 1] Counting users/items (limit={fmt(read_limit) if read_limit else '∞'})")
        for rec in stream_jsonl(in_path, limit=read_limit):
            scanned1 += 1
            # If JSON decode error occurred, stream_jsonl already skipped.
            if not keep_rule(rec):
                if progress and scanned1 % progress == 0:
                    print(f"... pass1 scanned {fmt(scanned1)} | qualifying {fmt(kept1)}")
                continue
            u = str(rec.get(F_USER, "")).strip()
            it = str(rec.get(F_ITEM, "")).strip()
            if not u or not it:  # must have identifiers
                continue
            user_cnt[u] += 1
            item_cnt[it] += 1
            kept1 += 1
            if progress and scanned1 % progress == 0:
                print(f"... pass1 scanned {fmt(scanned1)} | qualifying {fmt(kept1)}")
        print(f"[PASS 1 DONE] scanned {fmt(scanned1)} | qualifying {fmt(kept1)}")
        if min_user_count:
            print(f"  Users ≥{min_user_count}: {fmt(sum(1 for v in user_cnt.values() if v >= min_user_count))}")
        if min_item_count:
            print(f"  Items ≥{min_item_count}: {fmt(sum(1 for v in item_cnt.values() if v >= min_item_count))}")

    # -------------------
    # Pass 2: Write filtered shards (+ sampled sparsity)
    # -------------------
    print(f"[PASS 2] Writing shards to: {out_dir}")
    scanned2 = kept2 = sampled = 0
    shard_idx = 0
    buf: list[dict] = []

    # sampling for compact sparsity
    sp_user_cnt, sp_item_cnt, sp_pairs = Counter(), Counter(), set()

    def flush():
        nonlocal shard_idx, buf
        if not buf: return
        shard_idx += 1
        out_path = os.path.join(out_dir, f"hotelrec_filtered_{shard_idx:05d}.csv")
        with open(out_path, "w", encoding="utf-8", newline='') as csvfile:
            fieldnames = list(buf[0].keys())
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(buf)
        buf.clear()

    for rec in stream_jsonl(in_path, limit=read_limit):
        scanned2 += 1
        if not keep_rule(rec):
            if progress and scanned2 % progress == 0:
                print(f"... pass2 scanned {fmt(scanned2)} | kept {fmt(kept2)}")
            continue

        u = str(rec.get(F_USER, "")).strip()
        it = str(rec.get(F_ITEM, "")).strip()
        if not u or not it:
            continue

        if need_counts:
            if (min_user_count and user_cnt[u] < min_user_count): 
                continue
            if (min_item_count and item_cnt[it] < min_item_count): 
                continue

        row = {
            "user": u,
            "item": it,
            "rating": safe_float(rec.get(F_RATING)),
        }
        if not drop_text:
            row["text"] = str(rec.get(F_TEXT, "")).strip()
        if not drop_props:
            # keep the original dict (stringified) for downstream parsing
            row["props"] = str(rec.get(F_PROPS, None))

        buf.append(row)
        kept2 += 1

        # sparsity sample
        if sampled < analysis_n:
            if u and it:
                sp_user_cnt[u] += 1
                sp_item_cnt[it] += 1
                sp_pairs.add(hashlib.blake2b(f"{u}||{it}".encode("utf-8"), digest_size=8).digest())
            sampled += 1

        if shard_size and len(buf) >= shard_size: flush()
        if progress and scanned2 % progress == 0:
            print(f"... pass2 scanned {fmt(scanned2)} | kept {fmt(kept2)} | sample {fmt(sampled)}")

    flush()

    # compact sparsity summary
    if sampled > 0:
        n_users, n_items, n_pairs = len(sp_user_cnt), len(sp_item_cnt), len(sp_pairs)
        density = (n_pairs / (n_users * n_items)) if n_users and n_items else 0.0
        sparsity = 1.0 - density
        uf, itf = sorted(sp_user_cnt.values()), sorted(sp_item_cnt.values())
        print("\nSPARSITY (filtered sample)")
        print(f"Scanned: {fmt(scanned2)}  |  Kept: {fmt(kept2)}  |  Sampled: {fmt(sampled)}")
        print(f"Users: {fmt(n_users)}  Items: {fmt(n_items)}  Pairs: {fmt(n_pairs)}")
        print(f"Density: {density:.8f}  Sparsity: {sparsity:.8f}")
        print(f"User p50/p90/p99: {pct(uf,50)}/{pct(uf,90)}/{pct(uf,99)}")
        print(f"Item p50/p90/p99: {pct(itf,50)}/{pct(itf,90)}/{pct(itf,99)}")

    print("\nSUMMARY")
    if need_counts:
        dropped_users = sum(v for u_, v in user_cnt.items() if v < min_user_count)
        dropped_items = sum(v for it_, v in item_cnt.items() if v < min_item_count)
        print(f"Pass1 qualifying rows: {fmt(kept1)}")
        print(f"Rows belonging to users <{min_user_count}: ~{fmt(dropped_users)} (overlaps with item drops)")
        print(f"Rows belonging to items <{min_item_count}: ~{fmt(dropped_items)} (overlaps with user drops)")
    print(f"Total scanned (pass2): {fmt(scanned2)}")
    print(f"Total kept (written) : {fmt(kept2)}")
    print(f"Shards written       : {fmt(shard_idx)}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True, help="Path to HotelRec.txt (JSONL)")
    ap.add_argument("--out-dir", required=True, help="Directory for filtered CSV shards")
    ap.add_argument("--shard-size", type=int, default=500_000, help="Rows per shard")
    ap.add_argument("--analysis-n", type=int, default=1_000_000, help="Kept rows to sample for sparsity")
    ap.add_argument("--progress", type=int, default=100_000, help="Pulse every K scanned rows")
    ap.add_argument("--limit", type=int, default=None, help="Only read the first N lines (optional)")
    ap.add_argument("--drop-text", action="store_true", help="Do not keep review text in output")
    ap.add_argument("--drop-props", action="store_true", help="Do not keep property_dict in output")
    ap.add_argument("--min-user-count", type=int, default=3, help="Drop users with < this many reviews")
    ap.add_argument("--min-item-count", type=int, default=3, help="Drop items with < this many reviews")
    args = ap.parse_args()

    preprocess_and_analyze(
        in_path=args.in_path,
        out_dir=args.out_dir,
        shard_size=args.shard_size,
        analysis_n=args.analysis_n,
        progress=args.progress,
        drop_text=args.drop_text,
        drop_props=args.drop_props,
        read_limit=args.limit,
        min_user_count=args.min_user_count,
        min_item_count=args.min_item_count,
    )

if __name__ == "__main__":
    main()
