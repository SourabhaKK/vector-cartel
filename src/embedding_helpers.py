# pyrefly: ignore [missing-import]
from sentence_transformers import SentenceTransformer
# pyrefly: ignore [missing-import]
from langchain_core.embeddings import Embeddings
from typing import List
import torch

# EMBEDDING_MODEL = "BAAI/bge-m3"
EMBEDDING_MODEL = "nomic-ai/nomic-embed-text-v1"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[LOG] Using embedding device: {DEVICE}")

embedding_model = SentenceTransformer(EMBEDDING_MODEL, trust_remote_code=True, device=DEVICE)


class Embeddings(Embeddings):
    """LangChain-compatible wrapper around the Embedding model."""

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        # Prefix required by nomic-embed-text for passage/document encoding
        prefixed = [f"search_document: {t}" for t in texts]
        return embedding_model.encode(prefixed, normalize_embeddings=True, batch_size = 16, show_progress_bar = True).tolist()

    def embed_query(self, text: str) -> List[float]:
        # Prefix required by nomic-embed-text for query encoding
        return embedding_model.encode([f"search_query: {text}"], normalize_embeddings=True)[0].tolist()

    

# Instantiate once — import this into other modules
embeddings = Embeddings()
