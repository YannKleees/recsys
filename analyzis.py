"""/**
 * analyzis.py
 *
 * Compute sparsity and user activity buckets from a JSONL hotel reviews file,
 * streaming up to N lines.
 *
 * This module provides a command-line entry point (main) which parses
 * arguments, streams a JSONL file, normalizes user/item identifiers,
 * counts occurrences and computes sparsity/summary statistics.
 */"""

import argparse, json, math, os, sys, hashlib
from collections import Counter

def pct(vals, p):
	"""/**
	 * Compute the p-th percentile of a (sorted) list `vals` using linear
	 * interpolation. Returns None for empty lists.
	 *
	 * Params:
	 *  - vals: sorted sequence of numeric values
	 *  - p: percentile in [0,100]
	 */"""
	if not vals: return None
	k = (len(vals)-1) * (p/100.0)
	f, c = math.floor(k), math.ceil(k)
	if f == c: return vals[int(k)]
	return int(round(vals[f] + (k - f) * (vals[c] - vals[f])))

def fmt_int(n): 
	"""/**
	 * Format integer with thousands separators for human-readable output.
	 */"""
	return f"{n:,}"

def main():
	"""/**
	 * Main entry point: parse CLI args, stream the input JSONL, compute
	 * counters and summary statistics, and optionally write a markdown report.
	 */"""

	ap = argparse.ArgumentParser()
	ap.add_argument("--path", required=True, help="Path to HotelRec.txt (JSONL)")
	ap.add_argument("--n", type=int, default=1_000_000,
					help="Max lines to scan; use 0 to scan the entire file")
	ap.add_argument("--progress", type=int, default=100_000, help="Pulse every K lines")
	ap.add_argument("--drop-undefined", action="store_true",
					help="Drop authors that are empty or '/undefined'")
	ap.add_argument("--report", default=None, help="Optional path to write a markdown summary")
	args = ap.parse_args()

	path = args.path
	if not os.path.isfile(path):
		print(f"File not found: {path}", file=sys.stderr)
		sys.exit(1)

	"""/**
	 * Counters and state:
	 *  - user_cnt: reviews per user
	 *  - item_cnt: reviews per hotel
	 *  - seen_pairs: hashed (user, hotel) pairs
	 *  - n_lines/n_valid/n_bad: scanning statistics
	 */"""
	user_cnt = Counter()     # reviews per user
	item_cnt = Counter()     # reviews per hotel
	seen_pairs = set()       # hashed (user, hotel)
	n_lines = 0
	n_valid = 0
	n_bad = 0

	def norm_user(u):
		"""/**
		 * Normalize the author field; drop/normalize undefined values when requested.
		 */"""
		if u is None: return None
		if isinstance(u, str):
			u = u.strip()
			if args.drop_undefined and (not u or u.lower() in {"/undefined", "undefined", "na", "null"}):
				return None
			return u
		return str(u)

	def norm_item(i):
		"""/**
		 * Normalize the hotel identifier to a stripped string.
		 */"""
		if i is None: return None
		return str(i).strip()

	"""/**
	 * Stream the JSONL file (respecting the --n limit).
	 */"""
	hard_limit = (args.n > 0)
	limit = args.n if hard_limit else None

	with open(path, "r", encoding="utf-8", errors="replace") as f:
		for line in f:
			n_lines += 1
			if limit and n_lines > limit:
				break
			try:
				rec = json.loads(line)
			except json.JSONDecodeError:
				n_bad += 1
				continue

			u = norm_user(rec.get("author"))
			it = norm_item(rec.get("hotel_url"))
			if not u or not it:
				continue

			n_valid += 1
			user_cnt[u] += 1
			item_cnt[it] += 1

			"""/**
			 * 64-bit hash of the (user, item) pair to count unique interactions.
			 */"""
			h = hashlib.blake2b(f"{u}||{it}".encode("utf-8"), digest_size=8).digest()
			seen_pairs.add(h)

			if args.progress and n_lines % args.progress == 0:
				print(f"...processed {fmt_int(n_lines)} lines; valid {fmt_int(n_valid)}")

	"""/**
	 * Compute core metrics: counts, averages, density/sparsity.
	 */"""
	n_users = len(user_cnt)
	n_items = len(item_cnt)
	n_pairs = len(seen_pairs)

	avg_per_user = n_valid / n_users if n_users else 0.0
	avg_per_item = n_valid / n_items if n_items else 0.0
	density = (n_pairs / (n_users * n_items)) if (n_users and n_items) else 0.0
	sparsity = 1.0 - density

	"""/**
	 * Distributions and bucket summaries.
	 */"""
	user_freqs = sorted(user_cnt.values())
	item_freqs = sorted(item_cnt.values())

	def bucket_stats(freqs, label):
		if not freqs:
			return f"{label}: no data"
		return (
			f"{label}: min={freqs[0]}, p50={pct(freqs,50)}, p75={pct(freqs,75)}, "
			f"p90={pct(freqs,90)}, p95={pct(freqs,95)}, p99={pct(freqs,99)}, max={freqs[-1]}"
		)

	# User activity buckets
	u_eq1  = sum(1 for c in user_cnt.values() if c == 1)
	u_eq2  = sum(1 for c in user_cnt.values() if c == 2)
	u_ge3  = sum(1 for c in user_cnt.values() if c >= 3)
	u_ge5  = sum(1 for c in user_cnt.values() if c >= 5)
	u_ge10 = sum(1 for c in user_cnt.values() if c >= 10)

	def frac(x): 
		return (x / n_users * 100.0) if n_users else 0.0

	# Print summary
	print("\n SPARSITY SUMMARY")
	print(f"Lines scanned:      {fmt_int(n_lines)}   (bad JSON: {fmt_int(n_bad)})")
	print(f"Valid ratings:      {fmt_int(n_valid)}")
	print(f"Unique users:       {fmt_int(n_users)}")
	print(f"Unique hotels:      {fmt_int(n_items)}")
	print(f"Unique user-hotel pairs: {fmt_int(n_pairs)}")
	print(f"Avg reviews per user:  {avg_per_user:.3f}")
	print(f"Avg reviews per hotel: {avg_per_item:.3f}")
	print(f"Matrix density:     {density:.8f}")
	print(f"Matrix sparsity:    {sparsity:.8f}")
	print(bucket_stats(user_freqs, "User review count distribution"))
	print(bucket_stats(item_freqs, "Hotel review count distribution"))

	print("\n USER ACTIVITY BUCKETS")
	print(f"=1  : {fmt_int(u_eq1)} users  ({frac(u_eq1):.2f}%)")
	print(f"=2  : {fmt_int(u_eq2)} users  ({frac(u_eq2):.2f}%)")
	print(f"≥3  : {fmt_int(u_ge3)} users  ({frac(u_ge3):.2f}%)")
	print(f"≥5  : {fmt_int(u_ge5)} users  ({frac(u_ge5):.2f}%)")
	print(f"≥10 : {fmt_int(u_ge10)} users  ({frac(u_ge10):.2f}%)")


if __name__ == "__main__":
	main()
