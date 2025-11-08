# data/hotelrec_loader.py
"""
Preprocess HotelRec by dropping ONLY rows where ALL THREE are missing:
  - rating
  - non-empty review text
  - any numeric sub-rating inside property_dict

THEN filter out all users with <min_user_count reviews and all items with
<min_item_count reviews (default: 3 and 3) computed on the kept rows.

This script streams the JSONL twice when min-count filters are enabled:
  - Pass 1: count users/items, create integer mappings, and save them.
  - Pass 2: write only rows whose user and item meet the thresholds
    1. A single simple file: user_id, item_id, rating
    2. Sharded detailed files: user_id, item_id, rating, review

It also prints a compact sparsity analysis on the first --analysis-n KEPT rows.

Example:
  python data/hotelrec_loader.py --in data/HotelRec.txt --out-dir data/filtered \
    --shard-size 500000 --analysis-n 1000000 --min-user-count 3 --min-item-count 3
"""
from __future__ import annotations
import argparse, json, csv, os, math, hashlib, sys
from collections import Counter
from typing import Iterator, Optional, List
import numpy as np

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
                           read_limit: Optional[int],
                           min_user_count: int, min_item_count: int,
                           drop_text: bool, drop_props: bool) -> None:
    os.makedirs(out_dir, exist_ok=True)

    # -------------------
    # Pass 1: Counting and creating integer mappings
    # -------------------
    user_map, item_map = {}, {}
    next_user_id, next_item_id = 0, 0
    user_cnt, item_cnt = Counter(), Counter()
    scanned1 = kept1 = 0

    print(f"[PASS 1] Counting users/items and creating mappings (limit={fmt(read_limit) if read_limit else 'all'})")
    for rec in stream_jsonl(in_path, limit=read_limit):
        scanned1 += 1
        if not keep_rule(rec):
            if progress and scanned1 % progress == 0:
                print(f"... pass1 scanned {fmt(scanned1)} | qualifying {fmt(kept1)}")
            continue

        u_str = str(rec.get(F_USER, "")).strip()
        it_str = str(rec.get(F_ITEM, "")).strip()
        if not u_str or not it_str: continue

        if u_str not in user_map:
            user_map[u_str] = next_user_id
            next_user_id += 1
        if it_str not in item_map:
            item_map[it_str] = next_item_id
            next_item_id += 1

        user_cnt[u_str] += 1
        item_cnt[it_str] += 1
        kept1 += 1
        if progress and scanned1 % progress == 0:
            print(f"... pass1 scanned {fmt(scanned1)} | qualifying {fmt(kept1)}")

    print(f"[PASS 1 DONE] Scanned {fmt(scanned1)} | qualifying {fmt(kept1)}")
    print(f"  Found {fmt(len(user_map))} unique users and {fmt(len(item_map))} unique items.")

    user_map_path = os.path.join(out_dir, "user_map.json")
    item_map_path = os.path.join(out_dir, "item_map.json")
    with open(user_map_path, 'w', encoding='utf-8') as f: json.dump(user_map, f)
    with open(item_map_path, 'w', encoding='utf-8') as f: json.dump(item_map, f)
    print(f"  Saved user mapping to {user_map_path}")
    print(f"  Saved item mapping to {item_map_path}")

    # -------------------
    # Pass 2: Write filtered outputs
    # -------------------
    print(f"[PASS 2] Writing outputs to: {out_dir}")
    scanned2 = kept2 = sampled = 0
    shard_idx = 0
    buf: list[dict] = []
    sp_user_cnt, sp_item_cnt, sp_pairs = Counter(), Counter(), set()
    simple_format_path = os.path.join(out_dir, "user_item_ratings.csv")

    # Define fieldnames for sharded files dynamically based on flags
    sharded_fieldnames = ["user_id", "item_id", "rating"]
    if not drop_text:
        sharded_fieldnames.append("text")
    if not drop_props:
        sharded_fieldnames.extend(PROP_KEYS)

    with open(simple_format_path, "w", encoding="utf-8", newline='') as simple_f:
        simple_writer = csv.writer(simple_f)
        simple_writer.writerow(["user_id", "item_id", "rating"])

        def flush():
            nonlocal shard_idx, buf
            if not buf: return
            shard_idx += 1
            out_path = os.path.join(out_dir, f"hotelrec_filtered_{shard_idx:05d}.csv")
            with open(out_path, "w", encoding="utf-8", newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=sharded_fieldnames)
                writer.writeheader()
                writer.writerows(buf)
            buf.clear()

        for rec in stream_jsonl(in_path, limit=read_limit):
            scanned2 += 1
            if not keep_rule(rec):
                if progress and scanned2 % progress == 0:
                    print(f"... pass2 scanned {fmt(scanned2)} | kept {fmt(kept2)}")
                continue

            u_str = str(rec.get(F_USER, "")).strip()
            it_str = str(rec.get(F_ITEM, "")).strip()
            if not u_str or not it_str: continue

            if (min_user_count > 1 and user_cnt.get(u_str, 0) < min_user_count) or \
               (min_item_count > 1 and item_cnt.get(it_str, 0) < min_item_count):
                continue

            user_id = user_map.get(u_str)
            item_id = item_map.get(it_str)
            if user_id is None or item_id is None: continue

            rating = safe_float(rec.get(F_RATING))
            simple_writer.writerow([user_id, item_id, np.float32(rating) if rating is not None else ""])

            row_for_shard = {"user_id": user_id, "item_id": item_id, "rating": rating}
            if not drop_text:
                row_for_shard["text"] = str(rec.get(F_TEXT, "")).strip()
            if not drop_props:
                props = rec.get(F_PROPS, {})
                if not isinstance(props, dict): props = {}
                for key in PROP_KEYS:
                    row_for_shard[key] = safe_float(props.get(key))

            buf.append(row_for_shard)
            kept2 += 1

            if sampled < analysis_n:
                sp_user_cnt[user_id] += 1
                sp_item_cnt[item_id] += 1
                sp_pairs.add((user_id, item_id))
                sampled += 1

            if shard_size and len(buf) >= shard_size: flush()
            if progress and scanned2 % progress == 0:
                print(f"... pass2 scanned {fmt(scanned2)} | kept {fmt(kept2)} | sample {fmt(sampled)}")
        flush()

    if sampled > 0:
        n_users, n_items, n_pairs = len(sp_user_cnt), len(sp_item_cnt), len(sp_pairs)
        density = (n_pairs / (n_users * n_items)) if n_users and n_items else 0.0
        uf, itf = sorted(sp_user_cnt.values()), sorted(sp_item_cnt.values())
        print("\nSPARSITY (filtered sample)")
        print(f"Scanned: {fmt(scanned2)}  |  Kept: {fmt(kept2)}  |  Sampled: {fmt(sampled)}")
        print(f"Users: {fmt(n_users)}  Items: {fmt(n_items)}  Pairs: {fmt(n_pairs)}")
        print(f"Density: {density:.8f}  Sparsity: {1.0 - density:.8f}")
        print(f"User reviews p50/p90/p99: {pct(uf,50)}/{pct(uf,90)}/{pct(uf,99)}")
        print(f"Item reviews p50/p90/p99: {pct(itf,50)}/{pct(itf,90)}/{pct(itf,99)}")

    print("\nSUMMARY")
    print(f"Total scanned (pass2): {fmt(scanned2)}")
    print(f"Total kept (written) : {fmt(kept2)}")
    print(f"Shards written: {fmt(shard_idx)} (in {out_dir})")
    print(f"Simple-format file written: {simple_format_path}")

def main():
    ap = argparse.ArgumentParser(description="""
Preprocesses HotelRec JSONL data into two formats while filtering by review counts.
1. A single CSV file in the root output directory: (user_id, item_id, rating).
2. Sharded CSV files in the same directory, with a schema controlled by flags.
Also generates JSON files mapping original user/item names to integer IDs.
""")
    ap.add_argument("--in", dest="in_path", required=True, help="Path to HotelRec.txt (JSONL)")
    ap.add_argument("--out-dir", required=True, help="Root directory for all output files")
    ap.add_argument("--shard-size", type=int, default=2_000_000, help="Rows per shard for the detailed-format output")
    ap.add_argument("--analysis-n", type=int, default=1_000_000, help="Number of kept rows to sample for sparsity analysis")
    ap.add_argument("--progress", type=int, default=1_000_000, help="Log progress every K scanned rows")
    ap.add_argument("--limit", type=int, default=None, help="For testing, only read the first N lines of the input file")
    ap.add_argument("--min-user-count", type=int, default=3, help="Drop users with < this many reviews (set to 1 to disable)")
    ap.add_argument("--min-item-count", type=int, default=3, help="Drop items with < this many reviews (set to 1 to disable)")
    ap.add_argument("--drop-text", action="store_true", help="Do not include the 'text' column in sharded output.")
    ap.add_argument("--drop-props", action="store_true", help="Do not include sub-rating columns in sharded output.")
    args = ap.parse_args()

    preprocess_and_analyze(
        in_path=args.in_path,
        out_dir=args.out_dir,
        shard_size=args.shard_size,
        analysis_n=args.analysis_n,
        progress=args.progress,
        read_limit=args.limit,
        min_user_count=args.min_user_count,
        min_item_count=args.min_item_count,
        drop_text=args.drop_text,
        drop_props=args.drop_props,
    )

if __name__ == "__main__":
    main()