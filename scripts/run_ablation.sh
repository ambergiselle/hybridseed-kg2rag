#!/usr/bin/env bash
# Seed-retrieval ablation on BioASQ: dense vs. BM25 vs. hybrid (RRF).
# Run from the repository root with the vLLM server already up on localhost:8000.
#   bash scripts/run_ablation.sh            # all retrievers
#   bash scripts/run_ablation.sh dense bm25 # a subset
set -euo pipefail

export PYTHONHASHSEED=0   # keep the distractor pool identical across runs

RETRIEVERS=("${@:-dense bm25 rrf}")
DATA=../data/bioasq/factoid_subset_200.json
KG=../data/bioasq/kgs/extract_subkgs

cd "$(dirname "$0")/../code"
for r in ${RETRIEVERS[@]}; do
  out=../output/bioasq/ablation/bioasq200_${r}.json
  echo "=== ${r} -> ${out}"
  python kg_rag_distractor.py --dataset bioasq --retriever "${r}" \
      --data_path "${DATA}" --kg_dir "${KG}" --result_path "${out}"
  python evaluate.py "${out}" "${DATA}" --top_k 10
done
