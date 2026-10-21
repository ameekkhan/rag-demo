from pathlib import Path

from langchain_core.documents import Document
from langchain_community.document_loaders import (
    Docx2txtLoader,
    PyPDFLoader,
    TextLoader,
)

SUPPORTED_SUFFIXES = {".pdf", ".txt", ".md", ".docx"}


def load_documents(directory: Path) -> list[Document]:
    """Load supported files recursively and attach a portable source path."""
    if not directory.exists():
        raise FileNotFoundError(f"Document directory does not exist: {directory}")

    documents: list[Document] = []
    for path in sorted(directory.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        loader = _loader_for(path)
        for document in loader.load():
            document.metadata["source"] = str(path.relative_to(directory))
            document.metadata["file_type"] = path.suffix.lower().lstrip(".")
            documents.append(document)
    return documents


def _loader_for(path: Path):
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return PyPDFLoader(str(path))
    if suffix == ".docx":
        return Docx2txtLoader(str(path))
    return TextLoader(str(path), autodetect_encoding=True)
