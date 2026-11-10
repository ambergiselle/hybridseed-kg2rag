"""Extract KG triplets for the BioASQ distractor setting (Thesis II version).

Reads the questions of one split plus the SAVED distractor file, so it extracts
exactly the passages the pipeline will use.

Output: one JSON per document in
data/bioasq/kgs/extract_subkgs/, mapping chunk_key -> [[h, r, t], ...].

Passages already processed (including ones with no triplets, saved as [])
are skipped, so the script is safe to stop and resume, and passages shared
between batches are only extracted once.

Run:
    python extract_kg_bioasq.py --split dev
    python extract_kg_bioasq.py --split test --start 0    --end 500
    python extract_kg_bioasq.py --split test --start 500  --end 1000
    python extract_kg_bioasq.py --split test --start 1000 --end 1400
"""

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from datasets import load_dataset
from llama_index.llms.openai_like import OpenAILike
from pools import build_pool
from tqdm import tqdm

DATA = "../../data/bioasq"


def extract_triplets(llm, ctx, max_chars=3000):
    """Extract triplets from a given context using the provided LLM."""
    ctx = ctx[:max_chars]  # Keep only first 3000 chars to avoid hitting max token limit

    query = f"Extract triplets informative from the text following the examples. Make sure the triplet texts are only directly from the given text! Complete directly and strictly following the instructions without any additional words, line break nor space!\n{'-' * 20}\nText: Aspirin is a nonsteroidal anti-inflammatory drug (NSAID) commonly used to reduce pain, fever, and inflammation by inhibiting cyclooxygenase enzymes.\nTriplets:<Aspirin##is a##NSAID>$$<Aspirin##inhibits##cyclooxygenase enzymes>$$<Aspirin##used to reduce##pain>$$<Aspirin##used to reduce##fever>$$<Aspirin##used to reduce##inflammation>$$\n{'-' * 20}\nText: Mutations in the BRCA1 gene are strongly associated with an increased risk of breast and ovarian cancer in affected individuals.\nTriplets:<BRCA1##associated with##breast cancer>$$<BRCA1##associated with##ovarian cancer>$$<BRCA1 mutations##increases risk of##breast cancer>$$\n{'-' * 20}\nText: {ctx}\nTriplets:"

    resp = llm.complete(query)  # Send to vLLM for triplet extraction
    resp = resp.text

    triplets = set()
    for triplet_text in resp.split("$$"):
        # Drop triplets that are too short
        if len(triplet_text) <= 6:
            continue

        # To remove <> brackets and split by '##' to get h, r, t
        triplet_text = triplet_text[1:-1]
        tokens = triplet_text.split("##")

        # To ensure only valid triplets (h,r,t)
        if len(tokens) != 3:
            continue

        h, r, t = (tok.strip() for tok in tokens)

        # Drop triplets with 'no', 'unknown', 'null' or self loops
        if (
            ("no " in h)
            or ("no " in t)
            or ("unknown" in h.lower())
            or ("unknown" in t.lower())
            or ("null" in h.lower())
            or ("null" in t.lower())
            or (h == t)
        ):
            continue
        # Hallucination checking: if neither h nor t are in the context, drop the triplet
        if (r not in ctx) and (t not in ctx):
            continue
        triplets.add((h, r, t))

    return [[h, r, t] for (h, r, t) in triplets]


def load_existing(out_dir, doc_key):
    """Load saved triplets for one document"""
    path = os.path.join(out_dir, f"{doc_key}.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split", choices=["dev", "test"], required=True)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end", type=int, default=None)
    p.add_argument("--distractors", default=f"{DATA}/distractors_15.json")
    p.add_argument("--out_dir", default=f"{DATA}/kgs/extract_subkgs")
    p.add_argument(
        "--workers", type=int, default=8, help="parallel requests sent to vLLM"
    )

    a = p.parse_args()

    # 1. Load corresponding data
    if a.split == "dev":
        split_file = f"{DATA}/dev_200.json"
    else:
        split_file = f"{DATA}/test_1400.json"

    with open(split_file) as f:
        questions = json.load(f)[a.start : a.end]
    with open(a.distractors) as f:
        distractors = json.load(f)

    corpus = load_dataset("rag-datasets/rag-mini-bioasq", "test-corpus")["passages"]
    passage_lookup = {row["id"]: row["passage"] for row in corpus}

    os.makedirs(a.out_dir, exist_ok=True)

    # 2. Collect every passage in this batch that still needs triplets
    todo, seen = [], set()

    for q in questions:
        for doc_key, chunk_key, text in build_pool(
            q, distractors[str(q["id"])], passage_lookup
        ):
            if (doc_key, chunk_key) in seen:
                continue
            seen.add((doc_key, chunk_key))
            if chunk_key not in load_existing(a.out_dir, doc_key):
                todo.append((doc_key, chunk_key, text))

    print(
        f"{a.split} questions {a.start}-{a.start + len(questions)}:"
        f"{len(seen)} unique passages, {len(todo)} need extraction"
    )

    llm = OpenAILike(
        model="meta-llama/Llama-3.1-8B-Instruct",
        api_base="http://locaclhost:8000/v1",
        api_key="dummy",
        is_chat_model=True,
        request_timeout=120,
        temperature=0.0,
        max_tokens=64,
    )

    # 3. Send LLM requests in parallel to extract triplets [h,r,t]
    t0, done, empty, failed = time.time(), 0, 0, 0

    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        futures = {
            pool.submit(extract_triplets, llm, text): (doc_key, chunk_key)
            for doc_ket, chunk_key, text in todo
        }
        for fut in tqdm(as_completed(futures), total=len(futures)):
            doc_key, chunk_key = futures[fut]
            try:
                triplets = fut.result()
            except Exception as e:  # noqa: BLE001 -- log and skip any failed passage; it is retried next run
                failed += 1
                print(f"Failed {doc_key}/{chunk_key}:{e}")
                continue
            if not triplets:
                empty += 1
            existing = load_existing(a.out_dir, doc_key)
            existing[chunk_key] = triplets
            with open(os.path.join(a.out_dir, f"{doc_key}.json"), "w") as f:
                json.dump(existing, f)
            done += 1

    print(f"Extracted {done}, no triplets {empty}, failed {failed}")
    print(f"in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
