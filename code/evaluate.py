"""Evaluate KG2RAG predictions on BioASQ factoid questions.

Answer quality   Strict Accuracy (SAcc), Lenient Accuracy (LAcc), MRR over the
                 top-5 predicted entities; ROUGE-2 F1 of the justification
                 against the BioASQ ideal answer.
Retrieval        Hit Rate@k and MRR of the seed documents, where a document is
                 gold if one of its snippets contains an exact answer string.

Usage:
    python evaluate.py PRED_JSON GOLD_JSON [--top_k 10] [--compliant_only]

--compliant_only scores answer-quality metrics only on responses that follow
the numbered-list format; retrieval metrics always use every question.
"""

import argparse
import json
import re

from rouge_score import rouge_scorer

SCORER = rouge_scorer.RougeScorer(['rouge2'], use_stemmer=True)


# ----------------------------- parsing ------------------------------------- #

def split_justification(pred_text):
    if 'Justification:' in pred_text:
        list_part, justification = pred_text.split('Justification:', 1)
        return list_part, justification
    return pred_text, ''


def is_format_compliant(pred_text):
    list_part, _ = split_justification(pred_text)
    return bool(re.match(r'\d+\.', list_part.strip()))


def parse_response(pred_text):
    """Return (top-5 entities, justification). Prose answers fall back to one entity."""
    list_part, justification = split_justification(pred_text)
    entities = []
    for line in list_part.strip().split('\n'):
        m = re.match(r'\d+\.\s*(.+)', line.strip())
        if m:
            entities.append(m.group(1).strip().lower())
    if not entities:
        entities = [pred_text.strip().lower()]
        if not justification:
            justification = pred_text.strip()
    return entities[:5], justification.strip()


def normalize(s):
    return re.sub(r'[^\w\s]', '', s.lower()).strip()


# ----------------------------- metrics ------------------------------------- #

def factoid_scores(pred_entities, gold_groups):
    """SAcc, LAcc, MRR. A prediction matches if it contains any gold synonym."""
    gold = {normalize(a) for g in gold_groups for a in (g if isinstance(g, list) else [g])}
    preds = [normalize(e) for e in pred_entities]
    hit = lambda p: any(g in p for g in gold)

    sacc = int(bool(preds) and hit(preds[0]))
    lacc = int(any(hit(p) for p in preds))
    mrr = next((1.0 / r for r, p in enumerate(preds, start=1) if hit(p)), 0.0)
    return sacc, lacc, mrr


def ideal_answer_text(ideal):
    if isinstance(ideal, list):
        return ideal[0] if ideal else ''
    return ideal or ''


def answer_bearing_docs(question):
    """Doc keys whose snippet text contains an exact-answer string."""
    answers = [a.lower() for g in question['exact_answer'] for a in (g if isinstance(g, list) else [g])]
    return {
        s['document'].replace('/', '_').replace(':', '_')
        for s in question['snippets']
        if any(a in s['text'].lower() for a in answers)
    }


def hit_rate_at_k(ranked_docs, gold_docs, k):
    return int(any(d in gold_docs for d in ranked_docs[:k]))


def retrieval_mrr(ranked_docs, gold_docs):
    return next((1.0 / r for r, d in enumerate(ranked_docs, start=1) if d in gold_docs), 0.0)


# ----------------------------- driver -------------------------------------- #

def evaluate(pred_path, gold_path, top_k=10, compliant_only=False):
    with open(pred_path) as f:
        pred = json.load(f)
    with open(gold_path) as f:
        gold = {str(q['id']): q for q in json.load(f)}
    answers = pred['answer']
    seed_docs = pred.get('seed_docs', {})

    n = n_excluded = 0
    sacc_sum = lacc_sum = mrr_sum = 0.0
    rouge_sum, n_rouge = 0.0, 0
    hit_sum, rmrr_sum, n_retr = 0, 0.0, 0

    for qid, text in answers.items():
        qid = str(qid)
        if qid not in gold:
            continue
        q = gold[qid]

        if compliant_only and not is_format_compliant(text):
            n_excluded += 1
        else:
            entities, justification = parse_response(text)
            s, l, m = factoid_scores(entities, q['exact_answer'])
            sacc_sum, lacc_sum, mrr_sum, n = sacc_sum + s, lacc_sum + l, mrr_sum + m, n + 1
            ideal = ideal_answer_text(q.get('ideal_answer'))
            if justification and ideal:
                rouge_sum += SCORER.score(ideal, justification)['rouge2'].fmeasure
                n_rouge += 1

        if qid in seed_docs:
            gold_docs = answer_bearing_docs(q)
            if gold_docs:
                hit_sum += hit_rate_at_k(seed_docs[qid], gold_docs, top_k)
                rmrr_sum += retrieval_mrr(seed_docs[qid], gold_docs)
                n_retr += 1

    fmt = lambda total, count: f'{total / count:.3f}' if count else 'N/A'
    print(f'Answer quality   ({n} questions' + (f', {n_excluded} non-compliant excluded)' if compliant_only else ')'))
    print(f'  SAcc        {fmt(sacc_sum, n)}')
    print(f'  LAcc        {fmt(lacc_sum, n)}')
    print(f'  MRR         {fmt(mrr_sum, n)}')
    print(f'  ROUGE-2 F1  {fmt(rouge_sum, n_rouge)}  ({n_rouge} with justification)')
    print(f'Seed retrieval   ({n_retr} questions)')
    print(f'  Hit@{top_k:<7} {fmt(hit_sum, n_retr)}')
    print(f'  MRR         {fmt(rmrr_sum, n_retr)}')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('pred_path')
    p.add_argument('gold_path')
    p.add_argument('--top_k', type=int, default=10)
    p.add_argument('--compliant_only', action='store_true')
    a = p.parse_args()
    evaluate(a.pred_path, a.gold_path, a.top_k, a.compliant_only)
