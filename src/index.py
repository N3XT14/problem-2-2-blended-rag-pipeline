"""
Elasticsearch indexing: BM25 + dense KNN + sparse (SPLADE).

Three indexes matching IBM's Blended-RAG architecture:
  hvac_bm25: standard text search (BM25), model-number keyword analyzer
  hvac_knn: dense vector (all-MiniLM-L6-v2, 384-dim) for semantic search
  hvac_sparse: SPLADE sparse encoder (rank_features) — substitutes IBM's ELSER

Usage:  python src/index.py --chunks data/chunks.json
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
from elasticsearch import Elasticsearch, helpers
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
EMBED_DIM = 384
BATCH_SIZE = 64
SPLADE_TOPK = 128

_MAPPING_BM25 = {
    "settings": {
        "analysis": {
            "analyzer": {
                "model_number_analyzer": {
                    "type": "custom",
                    "tokenizer": "whitespace",
                    "filter": ["uppercase"],
                }
            }
        }
    },
    "mappings": {
        "properties": {
            "chunk_id": {"type": "keyword"},
            "doc_id": {"type": "keyword"},
            "page": {"type": "integer"},
            "section_title": {"type": "text"},
            "chunk_type": {"type": "keyword"},
            "text": {"type": "text", "analyzer": "standard"},
            "model_numbers": {
                "type": "text",
                "analyzer": "model_number_analyzer",
                "fielddata": True,
                "fields": {"keyword": {"type": "keyword"}},
            },
        }
    },
}

_MAPPING_KNN = {
    "mappings": {
        "properties": {
            "chunk_id": {"type": "keyword"},
            "doc_id": {"type": "keyword"},
            "page": {"type": "integer"},
            "section_title": {"type": "text"},
            "chunk_type": {"type": "keyword"},
            "text": {"type": "text"},
            "model_numbers": {"type": "keyword"},
            "embedding": {
                "type": "dense_vector",
                "dims": EMBED_DIM,
                "index": True,
                "similarity": "cosine",
            },
        }
    },
}

_MAPPING_SPARSE = {
    "mappings": {
        "properties": {
            "chunk_id": {"type": "keyword"},
            "doc_id": {"type": "keyword"},
            "page": {"type": "integer"},
            "section_title": {"type": "text"},
            "chunk_type": {"type": "keyword"},
            "text": {"type": "text"},
            "model_numbers": {"type": "keyword"},
            "sparse_tokens": {"type": "rank_features"},
        }
    },
}

def _get_client() -> Elasticsearch:
    client = Elasticsearch(ES_URL)
    info = client.info()
    logger.info(f"Connected to Elasticsearch {info['version']['number']}")
    return client

def _create_index(client: Elasticsearch, name: str, mapping: dict) -> None:
    if client.indices.exists(index=name):
        logger.info(f"Index '{name}' already exists — skipping creation")
        return
    client.indices.create(index=name, body=mapping)
    logger.info(f"Created index '{name}'")

def _bm25_actions(chunks: list[dict]):
    for c in chunks:
        yield {
            "_index": INDEX_BM25,
            "_id": c["chunk_id"],
            "_source": {
                "chunk_id": c["chunk_id"],
                "doc_id": c["doc_id"],
                "page": c["page"],
                "section_title": c["section_title"],
                "chunk_type": c["chunk_type"],
                "text": c["text"],
                "model_numbers": c.get("model_numbers", []),
            },
        }

def _knn_actions(chunks: list[dict], embeddings: np.ndarray):
    for c, emb in zip(chunks, embeddings):
        yield {
            "_index": INDEX_KNN,
            "_id": c["chunk_id"],
            "_source": {
                "chunk_id": c["chunk_id"],
                "doc_id": c["doc_id"],
                "page": c["page"],
                "section_title": c["section_title"],
                "chunk_type": c["chunk_type"],
                "text": c["text"],
                "model_numbers": c.get("model_numbers", []),
                "embedding": emb.tolist(),
            },
        }

def _sparse_actions(chunks: list[dict], sparse_vecs: list[dict]):
    for c, sv in zip(chunks, sparse_vecs):
        yield {
            "_index": INDEX_SPARSE,
            "_id": c["chunk_id"],
            "_source": {
                "chunk_id": c["chunk_id"],
                "doc_id": c["doc_id"],
                "page": c["page"],
                "section_title": c["section_title"],
                "chunk_type": c["chunk_type"],
                "text": c["text"],
                "model_numbers": c.get("model_numbers", []),
                "sparse_tokens": sv,
            },
        }

def _splade_encode(texts: list[str], tokenizer, model, topk: int) -> list[dict]:
    inputs = tokenizer(
        texts, return_tensors="pt", padding=True,
        truncation=True, max_length=512
    )
    with torch.no_grad():
        logits = model(**inputs).logits
    vecs = torch.log(1 + torch.relu(logits)).max(dim=1).values
    results = []
    for vec in vecs:
        nonzero = (vec > 0).sum().item()
        vals, idxs = torch.topk(vec, k=min(topk, nonzero))
        tokens = tokenizer.convert_ids_to_tokens(idxs.tolist())
        # ES rank_features uses '.' as path separator — drop tokens containing it
        results.append({t: round(float(v), 4) for t, v in zip(tokens, vals.tolist()) if "." not in t})
    return results


def build_index(chunks_path: Path) -> None:
    chunks = json.loads(chunks_path.read_text())
    logger.info(f"Loaded {len(chunks)} chunks from {chunks_path}")

    client = _get_client()
    _create_index(client, INDEX_BM25, _MAPPING_BM25)
    _create_index(client, INDEX_KNN, _MAPPING_KNN)
    _create_index(client, INDEX_SPARSE, _MAPPING_SPARSE)

    logger.info("Indexing BM25 ...")
    success, errors = helpers.bulk(client, _bm25_actions(chunks), raise_on_error=False)
    logger.info(f"  BM25: {success} indexed, {len(errors)} errors")

    logger.info(f"Loading dense model: {EMBED_MODEL}")
    dense_model = SentenceTransformer(EMBED_MODEL)
    texts = [c["text"] for c in chunks]
    all_embeddings = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i + BATCH_SIZE]
        embs = dense_model.encode(batch, normalize_embeddings=True, show_progress_bar=False)
        all_embeddings.append(embs)
        logger.info(f"  Dense {min(i + BATCH_SIZE, len(texts))}/{len(texts)}")
    embeddings = np.vstack(all_embeddings)

    logger.info("Indexing KNN ...")
    success, errors = helpers.bulk(client, _knn_actions(chunks, embeddings), raise_on_error=False)
    logger.info(f"  KNN: {success} indexed, {len(errors)} errors")

    logger.info(f"Loading SPLADE model: {SPLADE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(SPLADE_MODEL)
    splade_model = AutoModelForMaskedLM.from_pretrained(SPLADE_MODEL)
    splade_model.eval()

    all_sparse: list[dict] = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i + BATCH_SIZE]
        vecs = _splade_encode(batch, tokenizer, splade_model, SPLADE_TOPK)
        all_sparse.extend(vecs)
        logger.info(f"  SPLADE {min(i + BATCH_SIZE, len(texts))}/{len(texts)}")

    logger.info("Indexing sparse ...")
    success, errors = helpers.bulk(client, _sparse_actions(chunks, all_sparse), raise_on_error=False)
    logger.info(f"  Sparse: {success} indexed, {len(errors)} errors")

    client.indices.refresh(index=f"{INDEX_BM25},{INDEX_KNN},{INDEX_SPARSE}")
    logger.info("Done — all three indexes ready.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks", type=Path, default=Path("data/chunks.json"))
    args = parser.parse_args()
    build_index(args.chunks)
