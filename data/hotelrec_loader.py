"""
/**
 * hotelrec_loader.py
 *
 * Preprocess HotelRec by dropping ONLY rows where ALL THREE are missing:
 *   - rating
 *   - non-empty review text
 *   - any numeric sub-rating inside property_dict
 *
 * Also prints a compact sparsity analysis on the first --analysis-n KEPT rows
 * (default: 1,000,000). Streams JSONL and writes filtered CSV files.
 *
 * Example:
 *   python data/hotelrec_loader.py --in data/HotelRec.txt --out-dir data/filtered \
 *     --shard-size 500000 --analysis-n 1000000 --drop-props
 */
"""
from __future__ import annotations

import argparse, json, csv, os, math, hashlib
from collections import Counter
from typing import Iterator, Optional, List

# HotelRec field names
F_USER = "author"
F_ITEM = "hotel_url"
F_RATING = "rating"
F_TEXT = "text"
F_PROPS = "property_dict"

# known sub-rating keys
PROP_KEYS = ["sleep quality","value","rooms","service","cleanliness","location"]

# helpers
def safe_float(x) -> Optional[float]:
    """/** return float(x) or None */"""
    try: return float(x)
    except Exception: return None

def nonempty_text(x) -> bool:
    """/** True if string is non-empty after strip */"""
    return bool(str(x or "").strip())

def has_any_subrating(props) -> bool:
    """/** True if at least one known property is numeric */"""
    if not isinstance(props, dict): return False
    return any(safe_float(props.get(k)) is not None for k in PROP_KEYS)

def keep_rule(rec: dict) -> bool:
    """
    /** Keep row if NOT (all 3 missing). We DROP only when:
     *  - rating is None AND
     *  - text is empty AND
     *  - no numeric sub-rating present
     */
    """
    r_ok = safe_float(rec.get(F_RATING)) is not None
    t_ok = nonempty_text(rec.get(F_TEXT))
    p_ok = has_any_subrating(rec.get(F_PROPS))
    return any([r_ok, t_ok, p_ok])

def fmt(n: int) -> str: return f"{n:,}"

def pct(sorted_vals: List[int], p: float) -> Optional[int]:
    """/** simple percentile on sorted ints */"""
    if not sorted_vals: return None
    k = (len(sorted_vals) - 1) * (p / 100.0)
    a, b = int(k), int(math.ceil(k))
    if a == b: return sorted_vals[a]
    return int(round(sorted_vals[a] + (k - a) * (sorted_vals[b] - sorted_vals[a])))

# streaming
def stream_jsonl(path: str, limit: Optional[int] = None) -> Iterator[dict]:
    """/** yield JSON objects line-by-line; skip malformed; stop after limit if set */"""
    n = 0
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if limit is not None and n >= limit: break
            try:
                yield json.loads(line); n += 1
            except json.JSONDecodeError:
                continue

# core: preprocess and sparsity (sampled) + write shards
def preprocess_and_analyze(in_path: str, out_dir: str, shard_size: int,
                           analysis_n: int, progress: int,
                           drop_text: bool, drop_props: bool,
                           read_limit: Optional[int]) -> None:
    os.makedirs(out_dir, exist_ok=True)
    scanned = kept = 0
    shard_idx = 0
    buf: list[dict] = []

    # sparsity
    user_cnt, item_cnt, pairs = Counter(), Counter(), set()
    sampled = 0

    def flush():
        nonlocal shard_idx, buf
        if not buf: return
        shard_idx += 1
        out_path = os.path.join(out_dir, f"hotelrec_filtered_{shard_idx:05d}.csv")
        with open(out_path, "w", encoding="utf-8", newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=list(buf[0].keys()))
            writer.writeheader()
            writer.writerows(buf)
        buf.clear()

    for rec in stream_jsonl(in_path, limit=read_limit):
        scanned += 1
        if not keep_rule(rec):
            if progress and scanned % progress == 0:
                print(f"... scanned {fmt(scanned)} | kept {fmt(kept)}")
            continue

        row = {
            "user": str(rec.get(F_USER, "")).strip(),
            "item": str(rec.get(F_ITEM, "")).strip(),
            "rating": safe_float(rec.get(F_RATING)),
        }
        if not drop_text:
            row["text"] = str(rec.get(F_TEXT, "")).strip()
        if not drop_props:
            row["props"] = str(rec.get(F_PROPS, None))

        buf.append(row)
        kept += 1

        # sampled sparsity on kept rows
        if sampled < analysis_n:
            u, it = row["user"], row["item"]
            if u and it:
                user_cnt[u] += 1
                item_cnt[it] += 1
                pairs.add(hashlib.blake2b(f"{u}||{it}".encode("utf-8"), digest_size=8).digest())
            sampled += 1

        if shard_size and len(buf) >= shard_size: flush()
        if progress and scanned % progress == 0:
            print(f"... scanned {fmt(scanned)} | kept {fmt(kept)} | sample {fmt(sampled)}")

    flush()

    # compact sparsity summary
    if sampled > 0:
        n_users, n_items, n_pairs = len(user_cnt), len(item_cnt), len(pairs)
        density = (n_pairs / (n_users * n_items)) if n_users and n_items else 0.0
        sparsity = 1.0 - density
        uf, itf = sorted(user_cnt.values()), sorted(item_cnt.values())
        print("\nSPARSITY (filtered sample)")
        print(f"Scanned: {fmt(scanned)}  |  Kept: {fmt(kept)}  |  Sampled: {fmt(sampled)}")
        print(f"Users: {fmt(n_users)}  Items: {fmt(n_items)}  Pairs: {fmt(n_pairs)}")
        print(f"Density: {density:.8f}  Sparsity: {sparsity:.8f}")
        print(f"User p50/p90/p99: {pct(uf,50)}/{pct(uf,90)}/{pct(uf,99)}")
        print(f"Item p50/p90/p99: {pct(itf,50)}/{pct(itf,90)}/{pct(itf,99)}")

    print(f"Total scanned: {fmt(scanned)}")
    print(f"Total kept:    {fmt(kept)}")
    print(f"Shards:        {fmt(shard_idx)}")

# CLI
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True, help="Path to HotelRec.txt (JSONL)")
    ap.add_argument("--out-dir", required=True, help="Directory for filtered JSONL.GZ shards")
    ap.add_argument("--shard-size", type=int, default=500_000, help="Rows per shard")
    ap.add_argument("--analysis-n", type=int, default=1_000_000, help="Kept rows to sample for sparsity")
    ap.add_argument("--progress", type=int, default=100_000, help="Pulse every K scanned rows")
    ap.add_argument("--limit", type=int, default=None, help="Only read the first N lines (optional)")
    ap.add_argument("--drop-text", action="store_true", help="Do not keep review text in output")
    ap.add_argument("--drop-props", action="store_true", help="Do not keep property_dict in output")
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
    )

if __name__ == "__main__":
    main()
