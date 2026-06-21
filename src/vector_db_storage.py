# pyrefly: ignore [missing-import]
from langchain_chroma import Chroma
from embedding_helpers import embeddings
# pyrefly: ignore [missing-import]
from langchain_core.documents import Document
from typing import List
import gc

def vector_db_storage(chunks: List[Document], batch_size: int = 256, persist_directory = 'runtime_vector_db'):
    """Store chunk documents in a Chroma vector store using batched embedding.

    Args:
        chunks (List[Document]): List of LangChain Document objects to store.
        batch_size (int): Number of documents processed per embedding batch to avoidcls
            excessive memory consumption.
    """
    # Initialize a Chroma collection without pre‑embedding
    vectorstore = Chroma(embedding_function=embeddings, persist_directory=persist_directory)
    # Add documents in batches to keep memory usage reasonable
    total_batches = (len(chunks) + batch_size - 1) // batch_size
    for i in range(total_batches):
        print(f'Batch num processing: {i + 1}/{total_batches}')
        batch = chunks[i * batch_size : (i + 1) * batch_size]
        texts = [doc.page_content for doc in batch]
        metadatas = [doc.metadata for doc in batch]
        vectorstore.add_texts(texts=texts, metadatas=metadatas)
        gc.collect()
    return vectorstore