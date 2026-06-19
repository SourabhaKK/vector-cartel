from sentence_transformers import SentenceTransformer
# pyrefly: ignore [missing-import]
from langchain_core.embeddings import Embeddings
from typing import List

nomic_embedding_model = SentenceTransformer("nomic-ai/nomic-embed-text-v1", trust_remote_code=True, device="cpu")


class NomicEmbeddings(Embeddings):
    """LangChain-compatible wrapper around the Nomic SentenceTransformer model."""

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        # Prefix required by nomic-embed-text for passage/document encoding
        prefixed = [f"search_document: {t}" for t in texts]
        return nomic_embedding_model.encode(prefixed, normalize_embeddings=True, batch_size = 16, show_progress_bar = True).tolist()

    def embed_query(self, text: str) -> List[float]:
        # Prefix required by nomic-embed-text for query encoding
        return nomic_embedding_model.encode([f"search_query: {text}"], normalize_embeddings=True)[0].tolist()

    

# Instantiate once — import this into other modules
jina_embeddings = NomicEmbeddings()
