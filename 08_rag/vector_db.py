"""Qdrant Cloud vector storage; all embedding computations stay in rag_utils."""
import hashlib
import os
from typing import Any
import numpy as np
from dotenv import load_dotenv
from config import ROOT, EMBED_MODEL

load_dotenv(ROOT / '.env')


def get_client():
    """Connect to Qdrant Cloud. Secrets are read only from environment variables."""
    from qdrant_client import QdrantClient
    url = os.getenv('QDRANT_URL', '').strip()
    key = os.getenv('QDRANT_API_KEY', '').strip()
    if not url or not key:
        raise RuntimeError('Set QDRANT_URL and QDRANT_API_KEY in .env (see README).')
    return QdrantClient(url=url, api_key=key, timeout=30)


def collection_name(chunks: list[dict]) -> str:
    """Stable identifier for the exact corpus and embedding model."""
    from rag_utils import cache_key
    return 'statistical_rag_' + cache_key(chunks)


def point_id(chunk_id: str) -> str:
    """Deterministic UUID string accepted as Qdrant point ID."""
    import uuid
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


def index_chunks(chunks: list[dict], vectors: np.ndarray, client=None):
    """Create a collection (if needed) and idempotently upsert precomputed embeddings.

    Returns collection name; no network call to OpenAI occurs here.
    """
    from qdrant_client import models
    arr = np.asarray(vectors, dtype=float)
    if not chunks or arr.ndim != 2 or arr.shape[0] != len(chunks) or arr.shape[1] == 0:
        raise ValueError('Need a nonempty list of chunks and one vector per chunk.')
    if not np.isfinite(arr).all():
        raise ValueError('Embedding vectors must contain only finite values.')
    client = client or get_client()
    name = collection_name(chunks)
    if not client.collection_exists(name):
        client.create_collection(
            collection_name=name,
            vectors_config=models.VectorParams(size=int(arr.shape[1]), distance=models.Distance.COSINE),
        )
    else:
        info = client.get_collection(name)
        existing = info.config.params.vectors
        if hasattr(existing, 'size') and existing.size != arr.shape[1]:
            raise ValueError('Existing collection dimension differs; change corpus/model or collection name.')
    points = [models.PointStruct(
        id=point_id(c['chunk_id']),
        vector=arr[i].tolist(),
        payload={key: str(c[key]) for key in ('chunk_id', 'source_id', 'title', 'url', 'text')},
    ) for i, c in enumerate(chunks)]
    client.upsert(collection_name=name, points=points, wait=True)
    return name


def search_collection(client, collection: str, query_vector: np.ndarray, k: int = 3, source_id: str | None = None) -> list[dict]:
    """Query Qdrant and return passage dictionaries compatible with basic RAG."""
    from qdrant_client import models
    q = np.asarray(query_vector, dtype=float).reshape(-1)
    if k < 1 or not q.size or not np.isfinite(q).all():
        raise ValueError('Require k >= 1 and a valid nonempty query vector.')
    filter_ = None
    if source_id is not None:
        filter_ = models.Filter(must=[models.FieldCondition(key='source_id', match=models.MatchValue(value=source_id))])
    result = client.query_points(
        collection_name=collection, query=q.tolist(), limit=k, query_filter=filter_, with_payload=True,
    )
    return [{**(hit.payload or {}), 'score': float(hit.score)} for hit in result.points]
