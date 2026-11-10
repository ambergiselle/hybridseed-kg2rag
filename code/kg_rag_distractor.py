"""Run KG2RAG on biomedical factoid QA in the distractor setting.

Pipeline per question:
    1. Build the candidate pool: all gold snippets + 15 rag-mini-bioasq
    2. Seed retrieval over the pool             
    3. KG-guided chunk expansion                
    4. KG-based filtering / reranking        
    5. Answer generation with the LLM (served by vLLM)

With --no_kg, steps 3-4 are skipped (plain RAG): the seed chunks go straight
to the LLM.

Output JSON per question:
- answer
- supporting chunks (sp) 
- seed documents (seed_docs)
- score with evaluate.py

Example:
    python kg_rag_distractor.py --dataset bioasq --retriever rrf \
        --data_path ../data/bioasq/dev_200.json \
        --kg_dir ../data/bioasq/kgs/extract_subkgs \
        --distractor_file ../data/bioasq/distractors_15.json \
        --result_path ../output/bioasq/thesis2/dev_kg_rrf.json
"""

import argparse
import logging
import os

import openai
import ujson as json
from FlagEmbedding import FlagReranker
from llama_index.core import PromptTemplate, Settings, VectorStoreIndex
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers import ResponseMode
from llama_index.core.schema import TextNode
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai_like import OpenAILike
from pools import build_pool, get_distractors
from retrievers import RETRIEVER_CHOICES, build_retriever
from tqdm import tqdm
from util.kg_post_processor import (
    GraphFilterPostProcessor,
    KGRetrievePostProcessor,
    NaivePostprocessor,
)
from util.kg_response_synthesizer import get_response_synthesizer

# Logging setup
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("KG2RAG")
logging.getLogger("httpx").setLevel(logging.CRITICAL)

# Constants
RAGMINI_HF_DATASET = "rag-datasets/rag-mini-bioasq"

QA_PROMPT = (
    "Context information is below.\n{context_str}\n"
    "You must answer in EXACTLY this format, with no other text before or after:\n"
    "1. <entity>\n2. <entity>\n(up to 5 entities, ordered by decreasing confidence)\n"
    "Justification: <brief explanation grounded only in the given context>\n"
    "Do not write any introductory sentence. Do not write in full prose for the entity list. "
    'Start your response immediately with "1."\n'
    "Q: What gene is most commonly mutated in cystic fibrosis?\n"
    "A: 1. CFTR\n2. ABCC7\nJustification: CFTR mutations disrupt chloride channel "
    "function, the underlying cause of cystic fibrosis.\n"
    "---------------------\nQ: {query_str}\nA: "
)


# Setup functions


def init_models(args):
    """Configure the LLM and the embedding model."""
    Settings.llm = OpenAILike(
        model=args.llm_model,
        api_base=args.llm_api_base,
        api_key="dummy",
        is_chat_model=True,
        request_timeout=args.llm_timeout,
        temperature=0.0,
        max_tokens=args.max_tokens,
    )
    Settings.embed_model = HuggingFaceEmbedding(
        model_name=args.embed_model, device=args.device
    )


def read_data(path):
    """Load JSON data from the given path."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_kg(kg_dir):
    """Load every sub KG file into one dictionary: {doc_key: {chunk_key: [[h, r, t], ...]}}."""
    doc2kg = {}

    logger.info("Loading KGs from %s", kg_dir)

    for fname in tqdm(os.listdir(kg_dir)):
        if not fname.endswith(".json"):
            continue
        with open(os.path.join(kg_dir, fname), "r", encoding="utf-8") as f:
            subkg = json.load(f)

        if subkg:
            doc2kg[fname[:-5]] = subkg
    logger.info("Loaded KGs for %d documents", len(doc2kg))
    return doc2kg


def load_ragmini_corpus():
    """Load the rag-mini-bioasq corpus and return a lookup dict and list of passage IDs."""
    from datasets import load_dataset

    corpus = load_dataset(RAGMINI_HF_DATASET, "text-corpus")["passages"]
    passage_lookup = {row["id"]: row["passage"] for row in corpus}
    return passage_lookup, list(passage_lookup.keys())


# Candidate Pool Construction Functions (BioASQ gold snippets + random rag-mini-bioasq passages as distractors)


def bioasq_candidates(
    sample, passage_lookup, all_passage_ids, n_distractors, saved=None, seed=42
):
    """BioASQ gold snippets + random rag-mini-bioasq passages as distractors."""
    # Get corresponding distractor ID for each BioASQ question
    if saved is not None:
        distractor_ids = saved[str(sample["id"])]

    else:
        distractor_ids = get_distractors(
            sample["id"], all_passage_ids, n_distractors, seed
        )

    return build_pool(sample, distractor_ids, passage_lookup)


def get_question(args, sample):
    return sample["body"]


def get_candidates(args, sample, passage_lookup, all_passage_ids):
    return bioasq_candidates(
        sample,
        passage_lookup,
        all_passage_ids,
        args.n_distractors,
        saved=args.saved_distractors,
    )


# Per-question Processing Functions (KG2RAG Pipeline)


def build_chunks(candidates, kg):
    """Turn candidates into format that KG2RAG's post-processors need."""

    ents, doc_chunks, chunks_index, subkg = [], [], {}, {}

    for doc_key, chunk_key, text in candidates:
        ents.append(doc_key)
        chunks_index.setdefault(doc_key, {})[chunk_key] = text
        triplets = kg.get(doc_key, {}).get(chunk_key)
        if triplets:
            subkg.setdefault(doc_key, {})[chunk_key] = triplets

        # Wrap passage into LlamaIndex TextNode for retrieval
        doc_chunks.append(TextNode(text=text, id_=f"{doc_key}##{chunk_key}"))

    return ents, doc_chunks, chunks_index, subkg


def ordered_unique_doc_keys(nodes):
    """Return the ranked list of unique documents from retriever's ranked list of passages for retrieval evaluation."""
    seen, keys = set(), []
    for node in nodes:
        doc_key = node.node.id_.split("##")[0]
        if doc_key not in seen:
            seen.add(doc_key)
            keys.append(doc_key)
    return keys


def process_sample(args, sample, kg, reranker, passage_lookup, all_passage_ids):
    question = get_question(sample)
    candidates = get_candidates(args, sample, passage_lookup, all_passage_ids)
    ents, doc_chunks, chunks_index, subkg = build_chunks(candidates, kg)

    # Stage 1: Seed Retrieval
    index = VectorStoreIndex(doc_chunks)  # embedding = mxbai
    retriever = build_retriever(  # retriever = dense, lexical, or hybrid
        args.retriever,
        index,
        doc_chunks,
        top_k=args.top_k,
        alpha=args.alpha,
        rrf_k=args.rrf_k,
    )
    seed_doc_keys = ordered_unique_doc_keys(retriever.retrieve(question))

    # Stages 2-4: KG expansion, KG filtering, generation (KG2RAG, unchanged)
    pp_dataset = "bioasq"
    postprocessors = [
        # Step 2: KG Expansion
        KGRetrievePostProcessor(
            dataset=pp_dataset, ents=ents, doc2kg=subkg, chunks_index=chunks_index
        ),
        # Step 3: KG Filtering
        GraphFilterPostProcessor(
            dataset=pp_dataset,
            use_tpt=args.use_tpt,
            topk=args.top_k,
            ents=ents,
            doc2kg=subkg,
            chunks_index=chunks_index,
            reranker=reranker,
        ),
        # Step 4: Naive postprocessor (just returns the top-k chunks)
        NaivePostprocessor(dataset=pp_dataset),
    ]
    # Step 5: Answer Generation with the LLM (served by vLLM)
    synthesizer = get_response_synthesizer(
        response_mode=ResponseMode.COMPACT, text_qa_template=PromptTemplate(QA_PROMPT)
    )
    query_engine = RetrieverQueryEngine(
        retriever=retriever,
        response_synthesizer=synthesizer,
        node_postprocessors=postprocessors,
    )

    try:
        response = query_engine.query(question)
        answer = response.response
        supporting = [n.node.id_.split("##") for n in response.source_nodes]
    except openai.APIError as e:  # keep going if one question fails
        logger.error("Sample %s failed: %s", sample["id"], e)
        answer, supporting = "", []
    return sample["id"], answer, supporting, seed_doc_keys


# Functions to run all questions


def predict(args, data, kg, reranker, passage_lookup, all_passage_ids):
    """Run the KG2RAG pipeline on all questions with backup for every 20 questions, and return the predictions."""
    prediction = {"answer": {}, "sp": {}, "seed_docs": {}}
    n_supporting = 0
    for i, sample in enumerate(tqdm(data)):
        sid, answer, supporting, seed_docs = process_sample(
            args, sample, kg, reranker, passage_lookup, all_passage_ids
        )
        prediction["answer"][sid] = answer
        prediction["sp"][sid] = supporting
        prediction["seed_docs"][sid] = seed_docs
        n_supporting += len(supporting)
        if (i + 1) % args.checkpoint_every == 0:
            write_json(prediction, args.result_path + ".partial")
    logger.info(
        "Average number of supporting chunks: %.3f", n_supporting / max(len(data), 1)
    )
    return prediction


def write_json(obj, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f)


# Main function and argument parsing


def main(args):
    # Step 1: Load Questions
    data = read_data(args.data_path)

    # Step 2: Initialize LLM and Embedding Models
    init_models(args)

    # Step 3: Load all KG
    kg = read_kg(args.kg_dir)

    # Step 4: Load bge-reranker
    reranker = FlagReranker(model_name_or_path=args.reranker, device=args.device)

    # Step 5: Load rag-mini-bioasq corpus and distractors
    passage_lookup, all_passage_ids = load_ragmini_corpus()
    args.saved_distractors = (
        read_data(args.distractor_file) if args.distractor_file else None
    )

    # Step 6: Run KG2RAG pipeline for every questions
    prediction = predict(args, data, kg, reranker, passage_lookup, all_passage_ids)

    # Step 7: Save results
    write_json(prediction, args.result_path)
    logger.info("Prediction written to %s", args.result_path)


# Parser for Data, Retrieval, and Model Arguments
def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Data arguments: dataset, data_path, kg_dir, result_path, distractor_file, n_distractors
    g = p.add_argument_group("data")

    g.add_argument("--data_path", default="../data/bioasq/dev_200.json")
    g.add_argument("--kg_dir", default="../data/bioasq/kgs/extract_subkgs")
    g.add_argument(
        "--result_path", default="../output/bioasq/thesis2/dev_kg_dense.json"
    )
    g.add_argument(
        "--n_distractors",
        type=int,
        default=15,
        help="Distractor passages per question (only used without --distractor_file)",
    )
    g.add_argument(
        "--distractor_file",
        default="../data/bioasq/distractors_15.json",
        help='Saved distractor IDs from pools.py; pass "" to sample instead',
    )

    # Retrieval arguments: retriever, top_k, alpha, rrf_k, use_tpt, no_kg
    g = p.add_argument_group("retrieval")
    g.add_argument("--retriever", choices=RETRIEVER_CHOICES, default="dense")
    g.add_argument("--top_k", type=int, default=10, help="Number of seed chunks")
    g.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Dense weight for --retriever hybrid_alpha",
    )
    g.add_argument(
        "--rrf_k",
        type=int,
        default=60,
        help="RRF smoothing constant for --retriever rrf",
    )
    g.add_argument(
        "--use_tpt",
        action="store_true",
        help="Use triplet representation in the KG filter",
    )
    g.add_argument(
        "--no_kg",
        action="store_true",
        help="Plain RAG: skip KG expansion and KG filtering",
    )

    # Model arguments: llm_model, llm_api_base, llm_timeout, max_tokens, embed_model, reranker, device
    g = p.add_argument_group("models")
    g.add_argument("--llm_model", default="meta-llama/Llama-3.1-8B-Instruct")
    g.add_argument("--llm_api_base", default="http://localhost:8000/v1")
    g.add_argument("--llm_timeout", type=float, default=200)
    g.add_argument(
        "--max_tokens", type=int, default=256, help="Max tokens for LLM generation"
    )
    g.add_argument("--embed_model", default="mixedbread-ai/mxbai-embed-large-v1")
    g.add_argument("--reranker", default="../model/bge-reranker-large")
    g.add_argument("--device", default="cuda")

    p.add_argument(
        "--checkpoint_every",
        type=int,
        default=20,
        help="Write <result_path>.partial every N questions",
    )

    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
