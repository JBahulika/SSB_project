import pypdf
from langchain_text_splitters import RecursiveCharacterTextSplitter

CHUNK_SIZE = 700
CHUNK_OVERLAP = 100
DEFAULT_K = 4
MIN_SCORE = 0.25

_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
)


def extract_pdf_pages(pdf_file):
    reader = pypdf.PdfReader(pdf_file)
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        content = (page.extract_text() or "").strip()
        if content:
            pages.append({"page": i, "text": content})
    return pages


def chunk_pages(pages):
    chunks = []
    for page in pages:
        for piece in _splitter.split_text(page["text"]):
            text = piece.strip()
            if text:
                chunks.append({"text": text, "page": page.get("page")})
    return chunks


def build_index(embedder, chunks):
    import faiss
    import numpy as np

    texts = [chunk["text"] for chunk in chunks]
    embeddings = embedder.encode(texts, normalize_embeddings=True)
    embeddings = np.asarray(embeddings, dtype="float32")
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index


def search(embedder, index, chunks, query, k=DEFAULT_K):
    import numpy as np

    if not chunks:
        return []
    k = min(k, len(chunks))
    query_vec = embedder.encode([query], normalize_embeddings=True)
    scores, indices = index.search(np.asarray(query_vec, dtype="float32"), k=k)
    hits = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue
        hit = dict(chunks[int(idx)])
        hit["score"] = float(score)
        hits.append(hit)
    strong = [hit for hit in hits if hit["score"] >= MIN_SCORE]
    return strong or hits[:1]
