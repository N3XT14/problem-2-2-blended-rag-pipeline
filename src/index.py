"""
Elasticsearch indexing: BM25 + dense KNN.

Creates two indexes matching IBM's Blended-RAG architecture:
  hvac_bm25: standard text search (BM25), model-number keyword analyzer
  hvac_knn: dense vector (all-MiniLM-L6-v2, 384-dim) for semantic search

Usage:  python src/index.py --chunks data/chunks.json
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
from elasticsearch import Elasticsearch, helpers
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
logger = logging.getLogger(__name__)

ES_URL = "http://localhost:9200"
INDEX_BM25 = "hvac_bm25"
INDEX_KNN = "hvac_knn"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_DIM = 384
BATCH_SIZE = 64


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

def build_index(chunks_path: Path) -> None:
    chunks = json.loads(chunks_path.read_text())
    logger.info(f"Loaded {len(chunks)} chunks from {chunks_path}")

    client = _get_client()
    _create_index(client, INDEX_BM25, _MAPPING_BM25)
    _create_index(client, INDEX_KNN,  _MAPPING_KNN)

    logger.info("Indexing BM25 ...")
    success, errors = helpers.bulk(client, _bm25_actions(chunks), raise_on_error=False)
    logger.info(f"  BM25: {success} indexed, {len(errors)} errors")

    logger.info(f"Loading embedding model: {EMBED_MODEL}")
    model = SentenceTransformer(EMBED_MODEL)

    texts = [c["text"] for c in chunks]
    all_embeddings = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i + BATCH_SIZE]
        embs = model.encode(batch, normalize_embeddings=True, show_progress_bar=False)
        all_embeddings.append(embs)
        logger.info(f"  Embedded {min(i + BATCH_SIZE, len(texts))}/{len(texts)}")
    embeddings = np.vstack(all_embeddings)

    logger.info("Indexing KNN ...")
    success, errors = helpers.bulk(
        client, _knn_actions(chunks, embeddings), raise_on_error=False
    )
    logger.info(f"  KNN: {success} indexed, {len(errors)} errors")

    client.indices.refresh(index=f"{INDEX_BM25},{INDEX_KNN}")
    logger.info("Done — indexes ready.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks", type=Path, default=Path("data/chunks.json"))
    args = parser.parse_args()
    build_index(args.chunks)
