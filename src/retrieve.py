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

from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
logger = logging.getLogger(__name__)

ES_URL = "http://localhost:9200"
INDEX_BM25 = "hvac_bm25"
INDEX_KNN = "hvac_knn"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RRF_K = 60

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

def retrieve(query: str, top_k: int = 5, strategy: str = "best_fields", model_filter: Optional[str] = None) -> list[dict]:
    client = _get_client()
    model = SentenceTransformer(EMBED_MODEL)

    embedding = model.encode(query, normalize_embeddings=True).tolist()
    candidate_k = top_k * 3

    bm25_hits = _bm25_search(client, query, candidate_k, strategy, model_filter)
    knn_hits = _knn_search(client, embedding, candidate_k, model_filter)
    logger.info(f"BM25({strategy}): {len(bm25_hits)}  KNN: {len(knn_hits)}")

    return _rrf_merge(bm25_hits, knn_hits, top_k=top_k)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--strategy", choices=["match", "best_fields"], default="best_fields")
    parser.add_argument("--model-filter", default=None)
    args = parser.parse_args()

    results = retrieve(args.query, args.top_k, args.strategy, args.model_filter)
    for i, r in enumerate(results, 1):
        print(f"\n--- Result {i} (score={r['score']:.4f}) ---")
        print(f"Doc: {r['doc_id']}  Page: {r['page']}  Section: {r['section_title']}")
        print(f"Type: {r['chunk_type']}  Models: {r.get('model_numbers', [])[:3]}")
        print(r["text"][:400])
