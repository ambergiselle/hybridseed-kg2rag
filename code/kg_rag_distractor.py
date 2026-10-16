"""Run KG2RAG on biomedical QA in the distractor setting.

Pipeline per question:
    1. Build a candidate chunk pool (gold snippets + rag-mini-bioasq distractors).
    2. Seed retrieval over the pool               <- the stage this thesis modifies
    3. KG-guided chunk expansion                  (unchanged from KG2RAG)
    4. KG-based filtering / reranking             (unchanged from KG2RAG)
    5. Answer generation with the LLM

Supported datasets:
    bioasq   BioASQ factoid questions; candidates = native snippets + random
             rag-mini-bioasq passages as distractors.
    ragmini  rag-mini-bioasq questions; candidates = relevant passages + random
             rag-mini-bioasq passages as distractors.

Example:
    python kg_rag_distractor.py --dataset bioasq --retriever dense \
        --data_path ../data/bioasq/factoid_subset_200.json \
        --kg_dir ../data/bioasq/kgs/extract_subkgs \
        --result_path ../output/bioasq/bioasq200_dense.json
"""

import argparse
import hashlib
import logging
import os
import random as rng

import ujson as json
from FlagEmbedding import FlagReranker
from llama_index.core import PromptTemplate, Settings, VectorStoreIndex
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers import ResponseMode
from llama_index.core.schema import TextNode
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai_like import OpenAILike
from tqdm import tqdm

from retrievers import RETRIEVER_CHOICES, build_retriever
from util.kg_post_processor import GraphFilterPostProcessor, KGRetrievePostProcessor, NaivePostprocessor
from util.kg_response_synthesizer import get_response_synthesizer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('KG2RAG')
logging.getLogger('httpx').setLevel(logging.CRITICAL)

DATASETS = ('bioasq', 'ragmini')
RAGMINI_HF_DATASET = 'rag-datasets/rag-mini-bioasq'

QA_PROMPT = (
    'Context information is below.\n{context_str}\n'
    'You must answer in EXACTLY this format, with no other text before or after:\n'
    '1. <entity>\n2. <entity>\n(up to 5 entities, ordered by decreasing confidence)\n'
    'Justification: <brief explanation grounded only in the given context>\n'
    'Do not write any introductory sentence. Do not write in full prose for the entity list. '
    'Start your response immediately with "1."\n'
    'Q: What gene is most commonly mutated in cystic fibrosis?\n'
    'A: 1. CFTR\n2. ABCC7\nJustification: CFTR mutations disrupt chloride channel '
    'function, the underlying cause of cystic fibrosis.\n'
    '---------------------\nQ: {query_str}\nA: '
)


# --------------------------------------------------------------------------- #
# Setup
# --------------------------------------------------------------------------- #

def init_models(args):
    """Configure the LLM (OpenAI-compatible server, e.g. vLLM) and the embedding model."""
    Settings.llm = OpenAILike(
        model=args.llm_model,
        api_base=args.llm_api_base,
        api_key='dummy',
        is_chat_model=True,
        request_timeout=args.llm_timeout,
        temperature=0.0,
        max_tokens=args.max_tokens,
    )
    Settings.embed_model = HuggingFaceEmbedding(model_name=args.embed_model, device=args.device)


def read_data(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f'{path} not found')
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def read_kg(kg_dir):
    """Load every per-document sub-KG: {doc_key: {chunk_key: [[h, r, t], ...]}}."""
    doc2kg = {}
    logger.info('Loading KGs from %s', kg_dir)
    for fname in tqdm(os.listdir(kg_dir)):
        if not fname.endswith('.json'):
            continue
        with open(os.path.join(kg_dir, fname), 'r', encoding='utf-8') as f:
            subkg = json.load(f)
        if subkg:
            doc2kg[fname[:-5]] = subkg
    logger.info('Loaded KGs for %d documents', len(doc2kg))
    return doc2kg


def load_ragmini_corpus():
    from datasets import load_dataset
    corpus = load_dataset(RAGMINI_HF_DATASET, 'text-corpus')['passages']
    passage_lookup = {row['id']: row['passage'] for row in corpus}
    return passage_lookup, list(passage_lookup.keys())


# --------------------------------------------------------------------------- #
# Candidate pools  (each returns a list of (doc_key, chunk_key, text))
# --------------------------------------------------------------------------- #

def doc_key_from_url(url):
    return url.replace('/', '_').replace(':', '_')


def snippet_key(text):
    return hashlib.md5(text.encode('utf-8')).hexdigest()[:12]


def bioasq_candidates(sample, passage_lookup, all_passage_ids, n_distractors, seed=42):
    """BioASQ native snippets + random rag-mini-bioasq distractor passages."""
    # NOTE: hash() of a str is randomised per Python process unless PYTHONHASHSEED is set,
    # so for string IDs the distractor set differs between runs. See README "Known issues".
    seed_offset = sample['id'] if isinstance(sample['id'], int) else (hash(sample['id']) % 100000)
    rng.seed(seed + seed_offset)

    native = [(doc_key_from_url(s['document']), snippet_key(s['text']), s['text']) for s in sample['snippets']]
    distractor_ids = rng.sample(all_passage_ids, n_distractors)
    distractors = [(f'ragmini_{pid}', f'ragmini_{pid}', passage_lookup[pid]) for pid in distractor_ids]

    pool = native + distractors
    if len(pool) < 50:
        logger.warning('Sample %s: candidate pool size %d is smaller than 50', sample['id'], len(pool))
    return pool


def ragmini_candidates(sample, passage_lookup, all_passage_ids, n_distractors, seed=42):
    """rag-mini-bioasq relevant passages + random non-relevant distractor passages."""
    rng.seed(seed + sample['id'])
    relevant_ids = set(sample['relevant_passage_ids'])
    relevant = [(pid, passage_lookup[pid]) for pid in sample['relevant_passage_ids'] if pid in passage_lookup]
    pool_ids = [pid for pid in all_passage_ids if pid not in relevant_ids]
    distractors = [(pid, passage_lookup[pid]) for pid in rng.sample(pool_ids, n_distractors)]
    # rag-mini passages are single chunks, so doc_key == chunk_key.
    return [(str(pid), str(pid), text) for pid, text in relevant + distractors]


def get_question(args, sample):
    return sample['body'] if args.dataset == 'bioasq' else sample['question']


def get_candidates(args, sample, passage_lookup, all_passage_ids):
    if args.dataset == 'bioasq':
        return bioasq_candidates(sample, passage_lookup, all_passage_ids, args.n_distractors)
    return ragmini_candidates(sample, passage_lookup, all_passage_ids, args.n_distractors)


# --------------------------------------------------------------------------- #
# Per-question pipeline
# --------------------------------------------------------------------------- #

def build_chunks(candidates, kg):
    """Turn candidates into llama-index nodes plus the indexes KG2RAG's post-processors need."""
    ents, doc_chunks, chunks_index, subkg = [], [], {}, {}
    for doc_key, chunk_key, text in candidates:
        ents.append(doc_key)
        chunks_index.setdefault(doc_key, {})[chunk_key] = text
        triplets = kg.get(doc_key, {}).get(chunk_key)
        if triplets:
            subkg.setdefault(doc_key, {})[chunk_key] = triplets
        doc_chunks.append(TextNode(text=text, id_=f'{doc_key}##{chunk_key}'))
    return ents, doc_chunks, chunks_index, subkg


def ordered_unique_doc_keys(nodes):
    seen, keys = set(), []
    for node in nodes:
        doc_key = node.node.id_.split('##')[0]
        if doc_key not in seen:
            seen.add(doc_key)
            keys.append(doc_key)
    return keys


def process_sample(args, sample, kg, reranker, passage_lookup, all_passage_ids):
    question = get_question(args, sample)
    candidates = get_candidates(args, sample, passage_lookup, all_passage_ids)
    ents, doc_chunks, chunks_index, subkg = build_chunks(candidates, kg)

    # Stage 1: seed retrieval
    index = VectorStoreIndex(doc_chunks)
    retriever = build_retriever(args.retriever, index, doc_chunks, top_k=args.top_k,
                                alpha=args.alpha, rrf_k=args.rrf_k)
    seed_doc_keys = ordered_unique_doc_keys(retriever.retrieve(question))

    # Stages 2-4: KG expansion, KG filtering, generation (KG2RAG, unchanged)
    pp_dataset = 'bioasq'  # both datasets use the doc_key##chunk_key id scheme
    postprocessors = [
        KGRetrievePostProcessor(dataset=pp_dataset, ents=ents, doc2kg=subkg, chunks_index=chunks_index),
        GraphFilterPostProcessor(dataset=pp_dataset, use_tpt=args.use_tpt, topk=args.top_k, ents=ents,
                                 doc2kg=subkg, chunks_index=chunks_index, reranker=reranker),
        NaivePostprocessor(dataset=pp_dataset),
    ]
    synthesizer = get_response_synthesizer(response_mode=ResponseMode.COMPACT,
                                           text_qa_template=PromptTemplate(QA_PROMPT))
    query_engine = RetrieverQueryEngine(retriever=retriever, response_synthesizer=synthesizer,
                                        node_postprocessors=postprocessors)
    try:
        response = query_engine.query(question)
        answer = response.response
        supporting = [n.node.id_.split('##') for n in response.source_nodes]
    except Exception as e:  # keep going if one question fails (e.g. LLM timeout)
        logger.error('Sample %s failed: %s', sample['id'], e)
        answer, supporting = '', []
    return sample['id'], answer, supporting, seed_doc_keys


def predict(args, data, kg, reranker, passage_lookup, all_passage_ids):
    prediction = {'answer': {}, 'sp': {}, 'seed_docs': {}}
    n_supporting = 0
    for i, sample in enumerate(tqdm(data)):
        sid, answer, supporting, seed_docs = process_sample(args, sample, kg, reranker, passage_lookup, all_passage_ids)
        prediction['answer'][sid] = answer
        prediction['sp'][sid] = supporting
        prediction['seed_docs'][sid] = seed_docs
        n_supporting += len(supporting)
        if (i + 1) % args.checkpoint_every == 0:
            write_json(prediction, args.result_path + '.partial')
    logger.info('Average number of supporting chunks: %.3f', n_supporting / max(len(data), 1))
    return prediction


def write_json(obj, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f)


def main(args):
    data = read_data(args.data_path)
    init_models(args)
    kg = read_kg(args.kg_dir)
    reranker = FlagReranker(model_name_or_path=args.reranker, device=args.device)
    passage_lookup, all_passage_ids = load_ragmini_corpus()

    prediction = predict(args, data, kg, reranker, passage_lookup, all_passage_ids)
    write_json(prediction, args.result_path)
    logger.info('Prediction written to %s', args.result_path)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    g = p.add_argument_group('data')
    g.add_argument('--dataset', choices=DATASETS, default='bioasq')
    g.add_argument('--data_path', default='../data/bioasq/factoid_subset_200.json')
    g.add_argument('--kg_dir', default='../data/bioasq/kgs/extract_subkgs')
    g.add_argument('--result_path', default='../output/bioasq/bioasq200_dense.json')
    g.add_argument('--n_distractors', type=int, default=None,
                   help='Distractor passages per question (default: 100 for bioasq, 15 for ragmini)')

    g = p.add_argument_group('retrieval')
    g.add_argument('--retriever', choices=RETRIEVER_CHOICES, default='dense')
    g.add_argument('--top_k', type=int, default=10, help='Number of seed chunks')
    g.add_argument('--alpha', type=float, default=0.5, help='Dense weight for --retriever hybrid_alpha')
    g.add_argument('--rrf_k', type=int, default=60, help='RRF smoothing constant for --retriever rrf')
    g.add_argument('--use_tpt', action='store_true', help='Use triplet representation in the KG filter')

    g = p.add_argument_group('models')
    g.add_argument('--llm_model', default='meta-llama/Llama-3.1-8B-Instruct')
    g.add_argument('--llm_api_base', default='http://localhost:8000/v1')
    g.add_argument('--llm_timeout', type=float, default=200)
    g.add_argument('--max_tokens', type=int, default=64)
    g.add_argument('--embed_model', default='mixedbread-ai/mxbai-embed-large-v1')
    g.add_argument('--reranker', default='../model/bge-reranker-large')
    g.add_argument('--device', default='cuda')

    p.add_argument('--checkpoint_every', type=int, default=20, help='Write <result_path>.partial every N questions')

    args = p.parse_args()
    if args.n_distractors is None:
        args.n_distractors = 100 if args.dataset == 'bioasq' else 15
    return args


if __name__ == '__main__':
    main(parse_args())
