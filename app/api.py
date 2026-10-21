from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.config import get_settings
from app.rag import answer_question, ingest_documents

app = FastAPI(title="Document RAG API", version="0.1.0")


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)


class Source(BaseModel):
    source: str
    page: str


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]


@app.get("/health")
def health() -> dict[str, str]:
    settings = get_settings()
    return {"status": "ok", "llm_model": settings.llm_model}


@app.post("/ingest")
def ingest() -> dict[str, int | str]:
    try:
        result = ingest_documents()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"status": "indexed", **result}


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    answer, sources = answer_question(request.question)
    return AskResponse(answer=answer, sources=sources)
