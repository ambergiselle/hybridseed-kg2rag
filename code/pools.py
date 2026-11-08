"""
Create fixed distractor pools: 15 rag-mini-bioasq passages per BioASQ question,
sampled once with a stable per-question seed and saved to JSON.

Run from code/:  python pools.py
"""

import argparse
import hashlib  # for MD5 hashing algorithm
import json
import random


def stable_seed(qid, base_seed=42):
    """Generate a stable random seed based on the query ID using first 8 digits of MD5 hash of query ID."""
    digest = hashlib.md5(str(qid).encode("utf-8")).hexdigest()
    return base_seed + int(digest[:8], 16)


def get_distractors(qid, all_passage_ids, n, base_seed=42):
    """Pick n random distractor passages from all_passage_ids, using a stable seed based on query ID."""
    rng = random.Random(stable_seed(qid, base_seed))
    return rng.sample(sorted(all_passage_ids), n)


def load_passage_ids():
    """Load and get all rag-mini-bioasq passage IDs only"""
    from datasets import load_dataset

    corpus = load_dataset("rag-datasets/rag-mini-bioasq", "text-corpus")["passages"]
    return [r["id"] for r in corpus]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--questions", default="../data/bioasq/factoid.json")
    p.add_argument("--out", default="../data/bioasq/distractors_15.json")
    p.add_argument("--n", type=int, default=15)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    with open(args.questions) as f:
        questions = json.load(f)

    # Load all passage IDs from the rag-mini-bioasq dataset
    ids = load_passage_ids()

    # For each question, generate n distractors and store them in a dictionary
    pools = {}
    for q in questions:
        qid = str(q["id"])
        pools[qid] = get_distractors(qid, ids, args.n, args.seed)

    # Save the distractor pools to JSON file
    with open(args.out, "w") as f:
        json.dump(pools, f)

    print(f"Saved {args.n} distractors for {len(pools)} questions to {args.out}")


if __name__ == "__main__":
    main()
