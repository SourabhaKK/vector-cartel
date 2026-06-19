from typing import List, Any
from pathlib import Path
import json
import sys
import os

# pyrefly: ignore [missing-import]
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader

# pyrefly: ignore [missing-import]
from langchain_core.documents import Document

# Add src folder to Python path for notebook usage
PROJECT_ROOT = r"C:\Users\knowu\Documents\Projects\AI_Hackathon\vector-cartel"
SRC_DIR = os.path.join(PROJECT_ROOT, "src")

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from vector_db_storage import vector_db_storage
from text_chunk import chunk_text_func

from ollama_inference import (
    extract_metadata_for_attack_files,
    extract_metadata_for_advisory_files,
)


def _json_safe(value: Any) -> str | int | float | bool:
    """
    Chroma metadata cannot safely store lists/dicts directly.
    Convert lists/dicts to JSON strings and None to 'N/A'.
    """
    if value is None:
        return "N/A"

    if isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, (list, dict, tuple, set)):
        return json.dumps(list(value) if isinstance(value, set) else value, ensure_ascii=False)

    return str(value)


def _sanitize_metadata(metadata: dict) -> dict:
    """
    Make metadata safe for ChromaDB.
    Chroma accepts simple scalar metadata values, not raw lists/dicts.
    """
    return {key: _json_safe(value) for key, value in metadata.items()}


def _detect_source_type(source_path: str) -> str:
    """
    Detect corpus source type from file path/name.
    """
    lower_path = source_path.lower()

    if "attack" in lower_path or "mitre" in lower_path:
        return "MITRE_ATTACK_ICS"

    if "advisories" in lower_path or "cisa" in lower_path or "icsa-" in lower_path:
        return "CISA_ICS_ADVISORY"

    if "800-82" in lower_path or "sp800-82" in lower_path or "nist_sp_800_82" in lower_path:
        return "NIST_SP_800_82"

    if "csf" in lower_path or "cybersecurity-framework" in lower_path:
        return "NIST_CSF_2_0"

    return "OTHER"


def _extract_document_level_metadata(doc: Document) -> dict:
    """
    Extract metadata once per loaded document, not once per chunk.

    For MITRE/CISA markdown files, use the Ollama metadata extractors.
    For PDFs/NIST files, create simple metadata from path/page info.
    """
    source_path = doc.metadata.get("source", "N/A")
    source_type = _detect_source_type(source_path)
    file_name = Path(source_path).name if source_path != "N/A" else "N/A"

    base_meta = {
        "source_type": source_type,
        "source_path": source_path,
        "file_name": file_name,
        "source": source_path,
        "url": "N/A",
        "document_title": Path(file_name).stem if file_name != "N/A" else "N/A",
        "source_name": "N/A",
    }

    try:
        if source_type == "MITRE_ATTACK_ICS":
            extracted = extract_metadata_for_attack_files(doc.page_content[:4000])
            extracted_dict = extracted.model_dump()

            base_meta.update({
                "source_name": f"MITRE ATT&CK for ICS {extracted.technique_id}",
                "document_title": extracted.name,
                "technique_id": extracted.technique_id,
                "technique_name": extracted.name,
                "tactics": extracted.tactics,
                "tactic_ids": extracted.tactic_ids,
                "is_subtechnique": extracted.is_subtechnique,
                "parent_technique": extracted.parent_technique,
                "tactic_source": extracted.tactic_source,
                "url": extracted.url,
            })

        elif source_type == "CISA_ICS_ADVISORY":
            extracted = extract_metadata_for_advisory_files(doc.page_content[:4000])
            extracted_dict = extracted.model_dump()

            base_meta.update({
                "source_name": f"CISA {extracted.alert_code}",
                "document_title": extracted.title,
                "advisory_id": extracted.alert_code,
                "alert_code": extracted.alert_code,
                "vendor": extracted.vendor,
                "release_date": extracted.release_date,
                "cvss_version": extracted.cvss_version,
                "cvss_score": extracted.cvss_score,
                "sectors": extracted.sectors,
                "countries": extracted.countries,
                "url": extracted.url,
            })

        else:
            # NIST PDFs or other documents do not need LLM metadata extraction.
            if source_type == "NIST_SP_800_82":
                base_meta.update({
                    "source_name": "NIST SP 800-82",
                    "document_title": "Guide to Operational Technology Security",
                })

            elif source_type == "NIST_CSF_2_0":
                base_meta.update({
                    "source_name": "NIST Cybersecurity Framework 2.0",
                    "document_title": "NIST Cybersecurity Framework 2.0",
                })

            else:
                base_meta.update({
                    "source_name": file_name,
                    "document_title": Path(file_name).stem if file_name != "N/A" else "N/A",
                })

    except Exception as e:
        print(f"[WARN] Metadata extraction failed for {source_path}: {e}")
        base_meta.update({
            "metadata_error": str(e),
        })

    return base_meta


def iterate_chunk_vectorize(
    corpus_dir: str,
    persist_directory: str = "runtime_vector_db",
    batch_size: int = 256,
) -> List[Document]:
    """
    Load SecureOps corpus files, chunk them, attach metadata, and store embeddings in Chroma.

    Supports:
    - Markdown files: MITRE ATT&CK for ICS and CISA ICS advisories
    - PDF files: NIST SP 800-82, NIST CSF 2.0, or other PDFs

    Returns:
        List[Document]: The final chunked documents that were embedded.
    """

    print(f"[LOG] Loading markdown files from: {corpus_dir}")

    md_loader = DirectoryLoader(
        path=corpus_dir,
        glob="**/*.md",
        loader_cls=TextLoader,
        loader_kwargs={
            "encoding": "utf-8",
            "autodetect_encoding": True,
        },
        show_progress=True,
        use_multithreading=True,
    )

    md_docs = md_loader.load()

    print(f"[LOG] Markdown documents loaded: {len(md_docs)}")
    print("--" * 30)

    print(f"[LOG] Loading PDF files from: {corpus_dir}")

    pdf_loader = DirectoryLoader(
        path=corpus_dir,
        glob="**/*.pdf",
        loader_cls=PyPDFLoader,
        show_progress=True,
        use_multithreading=True,
    )

    pdf_docs = pdf_loader.load()

    print(f"[LOG] PDF pages loaded: {len(pdf_docs)}")
    print("--" * 30)

    docs = md_docs + pdf_docs

    print(f"[LOG] Total loaded documents/pages: {len(docs)}")
    print("--" * 30)

    chunked_docs: List[Document] = []

    print("[LOG] Chunking documents and attaching metadata")
    print("--" * 30)

    for doc_idx, doc in enumerate(docs):
        if isinstance(doc, str):
            doc = Document(page_content=doc, metadata={})

        source_path = doc.metadata.get("source", "N/A")
        source_type = _detect_source_type(source_path)

        print(f"[LOG] Processing document {doc_idx + 1}/{len(docs)}: {source_path}")

        document_meta = _extract_document_level_metadata(doc)

        chunks = chunk_text_func(doc.page_content)

        for chunk_idx, chunk in enumerate(chunks):
            chunk_content = chunk.page_content

            meta = {}

            # Original loader metadata, for example source/page.
            if doc.metadata:
                meta.update(doc.metadata)

            # Document-level extracted metadata.
            meta.update(document_meta)

            # Header metadata from MarkdownHeaderTextSplitter.
            if chunk.metadata:
                meta.update(chunk.metadata)

            meta.update({
                "chunk_index": chunk_idx,
                "chunk_id": f"{Path(source_path).stem}_{doc_idx}_{chunk_idx}",
            })

            # Better source citation text for retrieval answers.
            if source_type == "MITRE_ATTACK_ICS":
                technique_id = meta.get("technique_id", "N/A")
                technique_name = meta.get("technique_name", "N/A")
                metacontent = (
                    f"Source Type: MITRE ATT&CK for ICS\n"
                    f"Technique ID: {technique_id}\n"
                    f"Technique Name: {technique_name}\n"
                    f"Tactics: {meta.get('tactics', [])}\n\n"
                )

            elif source_type == "CISA_ICS_ADVISORY":
                metacontent = (
                    f"Source Type: CISA ICS Advisory\n"
                    f"Advisory ID: {meta.get('advisory_id', meta.get('alert_code', 'N/A'))}\n"
                    f"Vendor: {meta.get('vendor', 'N/A')}\n"
                    f"Release Date: {meta.get('release_date', 'N/A')}\n"
                    f"CVSS Score: {meta.get('cvss_score', 'N/A')}\n\n"
                )

            elif source_type.startswith("NIST"):
                metacontent = (
                    f"Source Type: {source_type}\n"
                    f"Source Name: {meta.get('source_name', 'N/A')}\n"
                    f"Document Title: {meta.get('document_title', 'N/A')}\n"
                    f"Page: {meta.get('page', 'N/A')}\n\n"
                )

            else:
                metacontent = (
                    f"Source Type: {source_type}\n"
                    f"Source Name: {meta.get('source_name', 'N/A')}\n\n"
                )

            final_content = metacontent + chunk_content
            safe_meta = _sanitize_metadata(meta)

            chunked_docs.append(
                Document(
                    page_content=final_content,
                    metadata=safe_meta,
                )
            )

    print(f"[LOG] Number of chunks created: {len(chunked_docs)}")
    print("--" * 30)

    print("[LOG] Storing embeddings in Chroma vector DB")
    print("--" * 30)

    vector_db_storage(
        chunks=chunked_docs,
        persist_directory=persist_directory,
        batch_size=batch_size,
    )

    print("[LOG] Embeddings stored successfully")

    return chunked_docs

if __name__ == "__main__":
    iterate_chunk_vectorize(r"c:\\Users\\knowu\\Documents\\Projects\\AI_Hackathon\\vector-cartel\\corpus\\advisories", "runtime_advisories_vector_db", 256)
    print(f'Advisories Vector DB Created')
    iterate_chunk_vectorize(r"c:\\Users\\knowu\\Documents\\Projects\\AI_Hackathon\\vector-cartel\\corpus\\attack", "runtime_attack_ics_vector_db", 256)
    print(f'Attack ICS Vector DB Created')
    iterate_chunk_vectorize(r"c:\\Users\\knowu\\Documents\\Projects\\AI_Hackathon\\vector-cartel\\corpus\\nist", "runtime_nist_vector_db", 256)
    print(f'NIST Vector DB Created')