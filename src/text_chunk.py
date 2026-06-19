# rag_helpers/text_chunk.py
# pyrefly: ignore [missing-import]
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from typing import List
# pyrefly: ignore [missing-import]
from langchain_core.documents import Document

header_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=[
        ("#", "Section"),
        ("##", "Subsection"),
        ("###", "Subsubsection"),
        ("####", "Subsubsubsection"),
        ("#####", "Subsubsubsubsection"),
        ("######", "Subsubsubsubsubsection"),
    ]
)

size_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000,
    chunk_overlap=400,
    # optional: you can keep separators if you want
    separators=["\n\n", "\n", " "],
)

def chunk_text_func(text: str) -> List[Document]:
    """
    Split a markdown document into logical sections (by header) and then
    further break those sections into size‑limited chunks.
    Returns a flat list of Document chunks with preserved header metadata.
    """
    # First split on headers
    header_sections = header_splitter.split_text(text)
    # Then apply size‑based chunking to each section
    final_chunks: List[Document] = []
    for section in header_sections:
        # Pass the section's metadata so that RecursiveCharacterTextSplitter attaches it to the resulting chunks
        chunks = size_splitter.create_documents([section.page_content], metadatas=[section.metadata])
        final_chunks.extend(chunks)
    return final_chunks