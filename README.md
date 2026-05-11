# AI Document Processor — 100% Local

Runs entirely on your machine via Docker.
**No API keys. No internet required. No data leaves your computer.**

---

## Requirements

- Docker + Docker Compose
- 8GB RAM minimum (16GB recommended for larger models)
- GPU optional but speeds things up a lot

---

## Start (one command)

```bash
docker compose up
```

That's it. On first run it will:
1. Pull the Ollama image
2. Pull the `llama3.2:3b` model (~2GB download, one time only)
3. Start the backend and serve the frontend

Then open **http://localhost:8000**

---

## Stop

```bash
docker compose down
```

Models stay cached in a Docker volume — next start is instant.

---

## Project structure

```
doc-processor-local/
├── backend/
│   ├── main.py            # FastAPI — calls Ollama instead of any cloud API
│   └── requirements.txt
├── frontend/
│   └── index.html         # UI with live model status indicator
├── docker-compose.yml     # Ollama + backend wired together
├── Dockerfile
└── README.md
```

---

## Available vision models

Swap models by changing `MODEL_NAME` in `docker-compose.yml`:

| Model | Size | Speed | Accuracy |
|---|---|---|---|
| `llama3.2-vision` | ~2GB | Medium | ⭐⭐⭐⭐ (recommended) |
| `llava` | ~4GB | Medium | ⭐⭐⭐ |
| `llava:13b` | ~8GB | Slow | ⭐⭐⭐⭐ |
| `moondream` | ~1.7GB | Fast | ⭐⭐ |
| `minicpm-v` | ~5GB | Medium | ⭐⭐⭐⭐ |

To pull a different model manually:
```bash
docker exec -it doc_processor_ollama ollama pull llava
```

---

## GPU acceleration (optional but recommended)

### NVIDIA
The `docker-compose.yml` already includes the GPU config. Make sure you have:
- NVIDIA Container Toolkit installed
- `nvidia-smi` returns your GPU info

If you don't have a GPU, remove the `deploy` block from `docker-compose.yml` — it runs on CPU fine, just slower.

### Mac (Apple Silicon)
Ollama automatically uses Metal (GPU acceleration) on Apple Silicon when running natively.
For Docker on Mac, CPU is used. For best performance on Mac, run Ollama natively:

```bash
brew install ollama
ollama serve &
ollama pull llama3.2-vision

# Then run only the backend:
cd backend
pip install -r requirements.txt
OLLAMA_HOST=http://localhost:11434 uvicorn main:app --reload
```

---

## How it works

```
Browser → FastAPI backend → Ollama (local LLM)
                ↑                    ↑
         Converts PDF           Runs entirely
         to image               on your hardware
```

PDFs are converted to images using PyMuPDF, then passed to the vision model.
The model reads the image and returns structured JSON.

---

## Freelance / product next steps

- Wrap in a desktop app with Electron or Tauri
- Add a queue for batch processing multiple documents
- Store results in SQLite for history/search
- Add a simple auth layer if exposing to a team on a local network
