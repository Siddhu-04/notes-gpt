from pathlib import Path
import pymupdf
import chromadb

CHROMA_PATH = "chroma_db"
COLLECTION_NAME = "notes"

def load_pdf(path: Path) -> list[dict]:
    doc = pymupdf.open(path)
    return [
        {"text": page.get_text(), "page": i + 1, "source": path.name}
        for i, page in enumerate(doc)
        if page.get_text().strip()
    ]

def load_pdf_bytes(file_bytes: bytes, filename: str) -> list[dict]:
    doc = pymupdf.open(stream=file_bytes, filetype="pdf")
    return [
        {"text": page.get_text(), "page": i + 1, "source": filename}
        for i, page in enumerate(doc)
        if page.get_text().strip()
    ]

def chunk_pages(pages: list[dict], size: int = 800, overlap: int = 100) -> list[dict]:
    chunks = []
    for page in pages:
        text = page["text"]
        start = 0
        while start < len(text):
            chunks.append({
                "chunk_id": f"{page['source']}_p{page['page']}_{start}",
                "text": text[start:start + size],
                "page": page["page"],
                "source": page["source"],
            })
            start += size - overlap
    return chunks

def embed_chunks(chunks: list[dict], embedder) -> None:
    texts = [c["text"] for c in chunks]
    embeddings = embedder.encode(texts, normalize_embeddings=True)
    for chunk, emb in zip(chunks, embeddings):
        chunk["embedding"] = emb.tolist()

def index_to_chroma(chunks: list[dict], collection_name: str = COLLECTION_NAME, client=None):
    client = client or chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_or_create_collection(collection_name)
    if chunks:
        collection.upsert(
            ids=[c["chunk_id"] for c in chunks],
            embeddings=[c["embedding"] for c in chunks],
            documents=[c["text"] for c in chunks],
            metadatas=[{"page": c["page"], "source": c["source"]} for c in chunks],
        )
    return collection

if __name__ == "__main__":
    from retrieve import get_embedder

    all_chunks = []
    for pdf_path in Path("data").glob("*.pdf"):
        all_chunks.extend(chunk_pages(load_pdf(pdf_path)))

    embed_chunks(all_chunks, get_embedder())
    index_to_chroma(all_chunks)
    print(f"Indexed {len(all_chunks)} chunks from {len(list(Path('data').glob('*.pdf')))} PDFs")