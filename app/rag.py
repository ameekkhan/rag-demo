from functools import lru_cache

from langchain_chroma import Chroma
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import Settings, get_settings
from app.loaders import load_documents

COLLECTION_NAME = "documents"


@lru_cache
def embeddings():
    settings = get_settings()
    return OllamaEmbeddings(model=settings.embedding_model, base_url=settings.ollama_base_url)


def vector_store(settings: Settings | None = None) -> Chroma:
    settings = settings or get_settings()
    settings.chroma_persist_directory.mkdir(parents=True, exist_ok=True)
    return Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=str(settings.chroma_persist_directory),
        embedding_function=embeddings(),
    )


def ingest_documents() -> dict[str, int]:
    settings = get_settings()
    raw_documents = load_documents(settings.documents_directory)
    if not raw_documents:
        return {"documents": 0, "chunks": 0}

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    chunks = splitter.split_documents(raw_documents)
    store = vector_store(settings)
    store.delete_collection()
    vector_store(settings).add_documents(chunks)
    return {"documents": len(raw_documents), "chunks": len(chunks)}


def answer_question(question: str) -> tuple[str, list[dict[str, str]]]:
    settings = get_settings()
    matches = vector_store(settings).similarity_search(question, k=settings.retrieval_k)
    if not matches:
        return "I could not find any indexed document content to answer that question.", []

    context = "\n\n".join(
        f"[Source: {doc.metadata.get('source', 'unknown')}]\n{doc.page_content}"
        for doc in matches
    )
    prompt = f"""You answer questions only from the supplied document context.
If the answer is absent, say that clearly. Do not invent facts.

Context:
{context}

Question: {question}
Answer:"""
    llm = ChatOllama(model=settings.llm_model, base_url=settings.ollama_base_url, temperature=0)
    answer = llm.invoke(prompt).content
    sources = [
        {
            "source": str(doc.metadata.get("source", "unknown")),
            "page": str(doc.metadata.get("page", "")),
        }
        for doc in matches
    ]
    return str(answer), sources
