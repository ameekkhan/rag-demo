# RAG - Demo

A simple local RAG application that reads PDF, DOCX, TXT, and Markdown files from a folder and answers questions with source references.

## Stack

- **LLM:** Ollama / `qwen3:8b` (configurable)
- **Embeddings:** Ollama / `nomic-embed-text`
- **Orchestration:** LangChain
- **Vector store:** Chroma
- **API:** FastAPI
- **Runtime:** Docker Compose

## Architecture

```text
documents → LangChain loaders → chunking → embeddings → Chroma
question  → embedding search → relevant chunks → Qwen3 → answer + sources
```

## Run locally

1. Copy the configuration file and put documents into `data/documents/`.

   ```bash
   cp .env.example .env
   ```

2. Start the services.

   ```bash
   docker compose up --build -d
   ```

3. Download the models once Ollama has started.

   ```bash
   docker compose exec ollama ollama pull qwen3:8b
   docker compose exec ollama ollama pull nomic-embed-text
   ```

4. Index the documents and ask a question.

   ```bash
   curl -X POST http://localhost:8000/ingest
   curl -X POST http://localhost:8000/ask \
     -H 'content-type: application/json' \
     -d '{"question":"What does the document say about refunds?"}'
   ```

Open `http://localhost:8000/docs` for interactive API documentation.

## Endpoints

| Endpoint | Purpose |
| --- | --- |
| `GET /health` | Service readiness and active LLM model |
| `POST /ingest` | Rebuild the Chroma index from `data/documents/` |
| `POST /ask` | Answer a grounded question and return retrieved sources |

## Notes

The application only answers from retrieved document chunks. It returns source filenames (and PDF page numbers when available) so users can verify the answer. To use a smaller model, set `LLM_MODEL=qwen3:4b` in `.env`.
