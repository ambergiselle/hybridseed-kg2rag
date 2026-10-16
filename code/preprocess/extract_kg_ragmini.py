"""Extract KG triplets for the rag-mini-bioasq distractor setting.

For each question in data/ragmini/ragmini_subset_200.json, builds the candidate
pool (relevant passages + 15 distractors) and asks the LLM to extract
<head##relation##tail> triplets from every passage. Output: one JSON per passage
in data/ragmini/kgs/extract_subkgs/. Existing files are skipped.

Requires an OpenAI-compatible LLM server (e.g. vLLM) at localhost:8000.
Run from this folder:  python extract_kg_ragmini.py
"""

import os
import json
import random
import ast
from tqdm import tqdm
from datasets import load_dataset
from llama_index.llms.openai_like import OpenAILike

def extract_triplets(llm, ctx, max_chars=3000):
    ctx = ctx[:max_chars]
    query = f'Extract triplets informative from the text following the examples. Make sure the triplet texts are only directly from the given text! Complete directly and strictly following the instructions without any additional words, line break nor space!\n{"-"*20}\nText: Aspirin is a nonsteroidal anti-inflammatory drug (NSAID) commonly used to reduce pain, fever, and inflammation by inhibiting cyclooxygenase enzymes.\nTriplets:<Aspirin##is a##NSAID>$$<Aspirin##inhibits##cyclooxygenase enzymes>$$<Aspirin##used to reduce##pain>$$<Aspirin##used to reduce##fever>$$<Aspirin##used to reduce##inflammation>$$\n{"-"*20}\nText: Mutations in the BRCA1 gene are strongly associated with an increased risk of breast and ovarian cancer in affected individuals.\nTriplets:<BRCA1##associated with##breast cancer>$$<BRCA1##associated with##ovarian cancer>$$<BRCA1 mutations##increases risk of##breast cancer>$$\n{"-"*20}\nText: {ctx}\nTriplets:'
    resp = llm.complete(query)
    resp = resp.text
    triplets = set()
    for triplet_text in resp.split('$$'):
        if len(triplet_text) <= 6:
            continue
        triplet_text = triplet_text[1:-1]
        tokens = triplet_text.split('##')
        if len(tokens) != 3:
            continue
        h, r, t = (tok.strip() for tok in tokens)
        if ('no ' in h) or ('no ' in t) or ('unknown' in h.lower()) or ('unknown' in t.lower()) or ('null' in h.lower()) or ('null' in t.lower()) or (h == t):
            continue
        if (r not in ctx) and (t not in ctx):
            continue
        triplets.add((h, r, t))
    return [[h, r, t] for (h, r, t) in triplets]

def build_candidate_passages(sample, passage_lookup, all_passage_ids, n_distractors=15, seed=42):
    random.seed(seed + sample['id'])
    relevant = [(pid, passage_lookup[pid]) for pid in sample['relevant_passage_ids'] if pid in passage_lookup]
    relevant_set = set(sample['relevant_passage_ids'])
    distractor_candidates = [pid for pid in all_passage_ids if pid not in relevant_set]
    distractor_ids = random.sample(distractor_candidates, n_distractors)
    distractors = [(pid, passage_lookup[pid]) for pid in distractor_ids]
    return relevant + distractors

with open('../../data/ragmini/ragmini_subset_200.json') as f:
    subset = json.load(f)

corpus = load_dataset("rag-datasets/rag-mini-bioasq", "text-corpus")['passages']
passage_lookup = {row['id']: row['passage'] for row in corpus}
all_passage_ids = list(passage_lookup.keys())

llm = OpenAILike(
    model="meta-llama/Llama-3.1-8B-Instruct",
    api_base="http://localhost:8000/v1",
    api_key="dummy",
    is_chat_model=True,
    request_timeout=120,
    temperature=0.0,
    max_tokens=64
)

out_dir = '../../data/ragmini/kgs/extract_subkgs'
os.makedirs(out_dir, exist_ok=True)
count = 0

for sample in tqdm(subset):
    candidates = build_candidate_passages(sample, passage_lookup, all_passage_ids)
    for pid, ptext in candidates:
        pid_str = str(pid)
        out_path = os.path.join(out_dir, f'{pid_str}.json')
        if os.path.exists(out_path):
            continue
        try:
            ext_triplets = extract_triplets(llm, ptext)
        except Exception as e:
            print(f'Skipping passage {pid_str} due to error: {e}')
            continue
        if len(ext_triplets) == 0:
            continue
        triplets = {pid_str: ext_triplets}
        with open(out_path, 'w') as f:
            json.dump(triplets, f)
        count += 1

print(f'Newly extracted passage KGs: {count}')