import os
import base64
import httpx
import json
import re
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from typing import Optional
import fitz  # PyMuPDF

app = FastAPI(title="AI Document Processor (Local)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
MODEL_NAME = os.environ.get("MODEL_NAME", "llama3.2:3b")


# ---------------- PROMPTS ----------------
PROMPTS = {
    "invoice": """Extract structured invoice data.
Return ONLY valid JSON:
{"vendor_name":"","vendor_address":"","invoice_number":"","invoice_date":"","due_date":"","subtotal":"","tax":"","total":"","currency":"","line_items":[]}
No explanations, no markdown.""",

    "resume": """You are a STRICT extraction system.

RULES:
- Use ONLY visible text
- Do NOT guess anything
- If missing, return null
- Return ONLY JSON

Output:
{
  "full_name": null,
  "email": null,
  "phone": null,
  "skills": [],
  "experience": [],
  "education": []
}
""",

    "research": """Extract structured research paper data.
Return ONLY valid JSON:
{"title":"","authors":[],"summary":"","key_findings":[]}
No explanations, no markdown."""
}


# ---------------- PDF → IMAGE ----------------
def pdf_to_base64_image(file_bytes: bytes):
    """
    Convert ONLY FIRST PAGE of PDF to image
    (prevents memory overload in Ollama)
    """
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    page = doc[0]

    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
    img_bytes = pix.tobytes("png")

    return base64.b64encode(img_bytes).decode("utf-8")


# ---------------- JSON CLEANER ----------------
def extract_json(text: str):
    try:
        return json.loads(text)
    except:
        pass

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except:
            pass

    return {"raw_output": text}


# ---------------- MAIN ENDPOINT ----------------
@app.post("/api/process")
async def process_document(
    file: UploadFile = File(...),
    mode: str = Form(...),
    custom_prompt: Optional[str] = Form(None),
):

    file_bytes = await file.read()

    if len(file_bytes) > 20 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large (max 20MB)")

    is_pdf = file.filename.lower().endswith(".pdf")

    # ---------------- IMAGE PREP ----------------
    if is_pdf:
        try:
            b64_image = pdf_to_base64_image(file_bytes)
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))
    else:
        b64_image = base64.b64encode(file_bytes).decode("utf-8")

    # ---------------- PROMPT ----------------
    if mode == "custom":
        prompt = (custom_prompt or "Extract structured data") + "\nReturn ONLY JSON."
    else:
        prompt = PROMPTS.get(mode)
        if not prompt:
            raise HTTPException(status_code=400, detail="Invalid mode")

    # ---------------- OLLAMA REQUEST ----------------
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "user",
                "content": prompt,
                "images": [b64_image],
            }
        ],
        "stream": False,
        "options": {
            "temperature": 0,
            "num_predict": 1024,
        },
    }

    async def call_ollama():
        async with httpx.AsyncClient(timeout=180.0) as client:
            return await client.post(
                f"{OLLAMA_HOST}/api/chat",
                json=payload,
            )

    try:
        response = await call_ollama()

        if response.status_code != 200:
            raise HTTPException(status_code=502, detail=response.text)

    except httpx.ReadTimeout:
        raise HTTPException(status_code=504, detail="Model took too long to respond")
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))

    # ---------------- PARSE RESPONSE ----------------
    try:
        data = response.json()
        raw = data.get("message", {}).get("content", "")
    except:
        raise HTTPException(status_code=500, detail="Invalid response from Ollama")

    parsed = extract_json(raw)

    return {
        "success": True,
        "data": parsed,
        "model": MODEL_NAME,
    }


# ---------------- HEALTH CHECK ----------------
@app.get("/api/health")
async def health():
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            r = await client.get(f"{OLLAMA_HOST}/api/tags")
        return {"status": "ok", "ollama": "connected"}
    except:
        return {"status": "error", "ollama": "disconnected"}


# ---------------- FRONTEND ----------------
frontend_path = os.path.join(os.path.dirname(__file__), "..", "frontend")

if os.path.exists(frontend_path):
    app.mount("/", StaticFiles(directory=frontend_path, html=True), name="frontend")