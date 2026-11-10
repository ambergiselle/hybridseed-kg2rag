"""How many pooled passages have been processed by KG extraction?

Run from code/preprocess/:
    python check_kg_coverage.py --split dev
    python check_kg_coverage.py --split test
"""

import argparse
import json
import os
import sys

from datasets import load_dataset

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from pools import build_pool

DATA = "../../data/bioasq"

p = argparse.ArgumentParser()
p.add_argument("--split", choices=["dev", "test"], required=True)
p.add_argument("--distractors", default=f"{DATA}/distractors_15.json")
p.add_argument("--split", default=f"{DATA}/kgs/extract_subkgs.json")
a = p.parse_args()

with open(
    f"{DATA}/dev_200.json" if a.split == "dev" else f"{DATA}/test_1400.json"
) as f:
    questions = json.load(f)
with open(a.distractors) as f:
    distractors = json.load(f)

corpus = load_dataset("rag-datasets/rag-mini-bioasq", "text-corpus")["passages"]
passage_lookup = {row["id"]: row["passage"] for row in corpus}

cache = {}


def processed(doc_key):
    if doc_key not in cache:
        path = os.path.join(a.kg_dir, f"{doc_key}.json")
        cache[doc_key] = json.load(open(path)) if os.path.exists(path) else {}
    return cache[doc_key]


total = done = with_triplets = full_questions = 0

for q in questions:
    chunks = [
        (d, c) for d, c, _ in build_pool(q, distractors[str(q["id"])], passage_lookup)
    ]
    q_done = 0
    for doc_key, chunk_key in chunks:
        kg = processed(doc_key)
        total += 1
        if chunk_key in kg:
            done += 1
            q_done += 1
            with_triplets += bool(kg[chunk_key])
        full_questions += q_done == len(chunks)

print(f"{a.split}: {len(questions)} questions, {total} pooled passages")
print(f"processed {done}/{total}({100 * done / total:.1f}%)")
print(f"with triplets {with_triplets}/{total}({100 * with_triplets / total:.1f}%)")
print(f"fully covered questions {full_questions}/{len(questions)}")
