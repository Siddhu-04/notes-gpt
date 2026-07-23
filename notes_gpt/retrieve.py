from functools import lru_cache
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi

@lru_cache(maxsize=1)
def get_embedder() -> SentenceTransformer:
    return SentenceTransformer("BAAI/bge-small-en-v1.5")

@lru_cache(maxsize=1)
def get_reranker() -> CrossEncoder:
    return CrossEncoder("BAAI/bge-reranker-base")

def _all_chunks(collection) -> list[dict]:
    data = collection.get(include=["documents", "metadatas"])
    return [
        {"chunk_id": i, "text": d, **m}
        for i, d, m in zip(data["ids"], data["documents"], data["metadatas"])
    ]

def bm25_search(query: str, collection, k: int = 10) -> list[dict]:
    chunks = _all_chunks(collection)
    if not chunks:
        return []
    bm25 = BM25Okapi([c["text"].split() for c in chunks])
    scores = bm25.get_scores(query.split())
    ranked = sorted(zip(scores, chunks), key=lambda x: x[0], reverse=True)
    return [c for _, c in ranked[:k]]

def semantic_search(query: str, collection, k: int = 10) -> list[dict]:
    if collection.count() == 0:
        return []
    embedding = get_embedder().encode(query, normalize_embeddings=True).tolist()
    results = collection.query(query_embeddings=[embedding], n_results=k)
    return [
        {"chunk_id": i, "text": d, **m}
        for i, d, m in zip(results["ids"][0], results["documents"][0], results["metadatas"][0])
    ]

def hybrid_retrieve(query: str, collection, k: int = 5) -> list[dict]:
    if collection is None or collection.count() == 0:
        return []
    combined = list({
        c["chunk_id"]: c for c in bm25_search(query, collection) + semantic_search(query, collection)
    }.values())
    if not combined:
        return []
    pairs = [(query, c["text"]) for c in combined]
    scores = get_reranker().predict(pairs)
    ranked = sorted(zip(scores, combined), key=lambda x: x[0], reverse=True)
    return [c for _, c in ranked[:k]]