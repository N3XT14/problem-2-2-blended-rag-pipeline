"""
Blended retriever: BM25 + dense KNN → RRF fusion.

Follows IBM's Blended-RAG architecture. Two query strategies:
  match: single field text match
  best_fields: multi-field (text + section_title ^ 2 + model_numbers)

RRF fusion is my extension, Pipeline selects the single best index/strategy combo.

Usage:
    python src/retrieve.py --query "refrigerant charge for 48AJ060"
    python src/retrieve.py --query "why does unit short cycle" --strategy match
    python src/retrieve.py --query "E04 fault code" --model-filter 48AJ060
"""

from __future__ import annotations

import argparse
import logging
from typing import Optional

import torch
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForMaskedLM, AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
logger = logging.getLogger(__name__)

ES_URL = "http://localhost:9200"
INDEX_BM25 = "hvac_bm25"
INDEX_KNN = "hvac_knn"
INDEX_SPARSE = "hvac_sparse"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
SPLADE_MODEL = "naver/splade-cocondenser-selfdistil"
RRF_K = 60
SPLADE_TOPK = 64

def _get_client() -> Elasticsearch:
    return Elasticsearch(ES_URL)

def _bm25_search(client: Elasticsearch, query: str, top_k: int, strategy: str = "best_fields", model_filter: Optional[str] = None) -> list[dict]:
    if strategy == "match":
        q: dict = {"match": {"text": query}}
    else:
        q = {"multi_match": {
            "query": query,
            "fields": ["text", "section_title^2", "model_numbers"],
            "type": "best_fields",
        }}

    body: dict = {"query": q, "size": top_k}
    if model_filter:
        body["query"] = {"bool": {"must": [q, {"term": {"model_numbers.keyword": model_filter}}]}}

    return client.search(index=INDEX_BM25, body=body)["hits"]["hits"]

def _knn_search(client: Elasticsearch, embedding: list[float], top_k: int, model_filter: Optional[str] = None) -> list[dict]:
    knn_query: dict = {
        "field": "embedding",
        "query_vector": embedding,
        "k": top_k,
        "num_candidates": top_k * 10,
    }
    if model_filter:
        knn_query["filter"] = {"term": {"model_numbers": model_filter}}

    return client.search(
        index=INDEX_KNN,
        body={"knn": knn_query, "size": top_k},
    )["hits"]["hits"]

def _splade_query_vec(query: str, tokenizer, model, topk: int) -> dict:
    device = next(model.parameters()).device
    inputs = tokenizer(query, return_tensors="pt", truncation=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        logits = model(**inputs).logits
    vec = torch.log(1 + torch.relu(logits)).max(dim=1).values.squeeze()
    nonzero = (vec > 0).sum().item()
    vals, idxs = torch.topk(vec, k=min(topk, nonzero))
    tokens = tokenizer.convert_ids_to_tokens(idxs.tolist())
    return {t: float(v) for t, v in zip(tokens, vals.tolist()) if "." not in t}

def _sparse_search(client: Elasticsearch, query_vec: dict, top_k: int, model_filter: Optional[str] = None) -> list[dict]:
    should = [
        {"rank_feature": {"field": f"sparse_tokens.{token}", "boost": weight}}
        for token, weight in query_vec.items()
    ]
    q: dict = {"bool": {"should": should, "minimum_should_match": 1}}
    body: dict = {"query": q, "size": top_k}
    if model_filter:
        body["query"] = {"bool": {"must": [q, {"term": {"model_numbers": model_filter}}]}}
    return client.search(index=INDEX_SPARSE, body=body)["hits"]["hits"]

def _rrf_merge(*hit_lists: list[dict], top_k: int, k: int = RRF_K) -> list[dict]:
    """
    Reciprocal Rank Fusion across ranked lists.
    Our extension over Pipeline's approach as IBM picks the best single combination, RRF fuses all simultaneously without needing to classify the query first.
    """
    scores: dict[str, float] = {}
    docs: dict[str, dict] = {}

    for hits in hit_lists:
        for rank, hit in enumerate(hits, start=1):
            cid = hit["_id"]
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
            docs[cid]   = hit["_source"]

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return [{"chunk_id": cid, "score": sc, **docs[cid]} for cid, sc in ranked]

_dense_model = None
_splade_tokenizer = None
_splade_model = None

def _load_dense():
    global _dense_model
    if _dense_model is None:
        _dense_model = SentenceTransformer(EMBED_MODEL)
    return _dense_model

def _load_splade():
    global _splade_tokenizer, _splade_model
    if _splade_model is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _splade_tokenizer = AutoTokenizer.from_pretrained(SPLADE_MODEL)
        _splade_model = AutoModelForMaskedLM.from_pretrained(SPLADE_MODEL).to(device)
        _splade_model.eval()
    return _splade_tokenizer, _splade_model

def retrieve(query: str, top_k: int = 5, strategy: str = "best_fields", model_filter: Optional[str] = None, use_sparse: bool = True) -> list[dict]:
    client = _get_client()
    candidate_k = top_k * 3

    embedding = _load_dense().encode(query, normalize_embeddings=True).tolist()

    bm25_hits = _bm25_search(client, query, candidate_k, strategy, model_filter)
    knn_hits = _knn_search(client, embedding, candidate_k, model_filter)

    if use_sparse:
        tokenizer, splade_model = _load_splade()
        query_vec = _splade_query_vec(query, tokenizer, splade_model, SPLADE_TOPK)
        sparse_hits = _sparse_search(client, query_vec, candidate_k, model_filter)
        logger.info(f"BM25: {len(bm25_hits)}  KNN: {len(knn_hits)}  SPLADE: {len(sparse_hits)}")
        return _rrf_merge(bm25_hits, knn_hits, sparse_hits, top_k=top_k)

    logger.info(f"BM25: {len(bm25_hits)}  KNN: {len(knn_hits)}")
    return _rrf_merge(bm25_hits, knn_hits, top_k=top_k)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--strategy", choices=["match", "best_fields"], default="best_fields")
    parser.add_argument("--model-filter", default=None)
    parser.add_argument("--no-sparse", action="store_true")
    args = parser.parse_args()

    results = retrieve(args.query, args.top_k, args.strategy, args.model_filter, use_sparse=not args.no_sparse)
    for i, r in enumerate(results, 1):
        print(f"\n--- Result {i} (score={r['score']:.4f}) ---")
        print(f"Doc: {r['doc_id']}  Page: {r['page']}  Section: {r['section_title']}")
        print(f"Type: {r['chunk_type']}  Models: {r.get('model_numbers', [])[:3]}")
        print(r["text"][:400])
