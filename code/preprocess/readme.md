# Data preprocessing

Run these from `code/preprocess/`.

| Step | Script | Output |
|---|---|---|
| 1. Question subsets | `python make_subsets.py bioasq` / `python make_subsets.py ragmini` | `data/bioasq/factoid_subset_200.json`, `data/ragmini/ragmini_subset_200.json` |
| 2. KG extraction (BioASQ) | `python extract_kg_bioasq.py` | `data/bioasq/kgs/extract_subkgs/*.json` |
| 3. KG extraction (rag-mini) | `python extract_kg_ragmini.py` | `data/ragmini/kgs/extract_subkgs/*.json` |

`data/bioasq/factoid.json` holds the BioASQ factoid questions (with `snippets`,
`exact_answer` and `ideal_answer`). The rag-mini-bioasq corpus is downloaded
automatically from Hugging Face (`rag-datasets/rag-mini-bioasq`).

KG extraction calls an OpenAI-compatible LLM endpoint (we use vLLM serving
`meta-llama/Llama-3.1-8B-Instruct` on `localhost:8000`). Both extraction scripts
resume where they left off.

Each KG file maps `chunk_key -> [[head, relation, tail], ...]`. For BioASQ the
file name is the PubMed URL with `/` and `:` replaced by `_`, and `chunk_key` is
the first 12 hex characters of the MD5 of the snippet text.
