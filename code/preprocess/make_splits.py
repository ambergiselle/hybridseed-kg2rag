"""Split the 1,600 BioASQ questions into dev (200) and test (1,400) sets
Run from code/:  python make_splits.py
"""

import json
import os

DATA = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "bioasq"
)

# Load BioAQS 1,600 Factoid Questions
with open(os.path.join(DATA, "factoid.json")) as f:
    factoid = json.load(f)

# Load 200 dev questions
with open(os.path.join(DATA, "factoid_subset_200.json")) as f:
    dev = json.load(f)

# Splits
dev_ids = {str(q["id"]) for q in dev}
test = [q for q in factoid if str(q["id"]) not in dev_ids]

# Safety Checks
assert len(dev_ids) == len(dev), "duplicate IDs in the dev set"
assert len(test) + len(dev) == len(factoid), (
    "some dev questions are not in factoid.json"
)

for name, qs in (("dev_200.json", dev), (f"test_{len(test)}.json", test)):
    with open(os.path.join(DATA, name), "w") as f:
        json.dump(qs, f)
    print(f"Saved {len(qs)} questions to {name}")
