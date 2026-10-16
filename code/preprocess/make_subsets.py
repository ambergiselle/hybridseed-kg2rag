"""Create the fixed, seeded question subsets used in the experiments.

    python make_subsets.py bioasq   # data/bioasq/factoid.json -> factoid_subset_200.json
    python make_subsets.py ragmini  # rag-mini-bioasq (HF)     -> ragmini_subset_200.json

Both use random.seed(42) so the subsets are reproducible.
"""

import argparse
import ast
import json
import os
import random

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data')


def bioasq_subset(size, seed):
    with open(os.path.join(DATA_DIR, 'bioasq', 'factoid.json')) as f:
        factoid = json.load(f)
    print(f'Total factoid questions available: {len(factoid)}')
    random.seed(seed)
    return random.sample(factoid, size), os.path.join(DATA_DIR, 'bioasq', f'factoid_subset_{size}.json')


def ragmini_subset(size, seed):
    from datasets import load_dataset
    qap = load_dataset('rag-datasets/rag-mini-bioasq', 'question-answer-passages')['test']
    samples = [{
        'id': row['id'],
        'question': row['question'],
        'answer': row['answer'],
        'relevant_passage_ids': ast.literal_eval(row['relevant_passage_ids']),
    } for row in qap]
    random.seed(seed)
    return random.sample(samples, size), os.path.join(DATA_DIR, 'ragmini', f'ragmini_subset_{size}.json')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('dataset', choices=['bioasq', 'ragmini'])
    p.add_argument('--size', type=int, default=200)
    p.add_argument('--seed', type=int, default=42)
    a = p.parse_args()

    subset, out_path = (bioasq_subset if a.dataset == 'bioasq' else ragmini_subset)(a.size, a.seed)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(subset, f)
    print(f'Saved {len(subset)} questions to {out_path}')
