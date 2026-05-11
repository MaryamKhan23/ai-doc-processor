import os
import httpx
import json
import re
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import io

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from typing import Optional


app = FastAPI(title="AI Document Processor (Local)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- CONFIG ----------------
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://ollama:11434")
MODEL_NAME = os.environ.get("MODEL_NAME", "llama3.2:3b")


# ---------------- PROMPTS ----------------
PROMPTS = {
    "invoice": """
Extract structured invoice data.

Return ONLY valid JSON.

{
  "vendor_name": "",
  "vendor_address": "",
  "invoice_number": "",
  "invoice_date": "",
  "due_date": "",
  "subtotal": "",
  "tax": "",
  "total": "",
  "currency": "",
  "line_items": []
}
""",

    "resume": """
Extract resume data.

Return ONLY valid JSON.

{
  "full_name": null,
  "email": null,
  "phone": null,
  "skills": [],
  "experience": [],
  "education": []
}
""",

    "research": """
Extract research paper data.

Return ONLY valid JSON.

{
  "title": "",
  "authors": [],
  "summary": "",
  "key_findings": []
}
"""
}


# ---------------- OCR ----------------
def extract_text(file_bytes: bytes, is_pdf: bool):
    if is_pdf:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        text = ""

        for page in doc:
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
            img_bytes = pix.tobytes("png")

            image = Image.open(io.BytesIO(img_bytes))
            text += pytesseract.image_to_string(image) + "\n"

        return text

    image = Image.open(io.BytesIO(file_bytes))
    return pytesseract.image_to_string(image)


# ---------------- JSON CLEANER ----------------
def extract_json(text: str):
    if not text:
        return {"error": "empty_response"}

    try:
        return json.loads(text)
    except:
        pass

    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        raw = match.group()
        raw = re.sub(r",\s*}", "}", raw)
        raw = re.sub(r",\s*]", "]", raw)

        try:
            return json.loads(raw)
        except Exception as e:
            return {
                "error": "invalid_json",
                "details": str(e),
                "raw": raw
            }

    return {
        "error": "no_json_found",
        "raw_output": text
    }


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

    # ---------------- OCR STEP ----------------
    try:
        extracted_text = extract_text(file_bytes, is_pdf)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"OCR error: {str(e)}")

    # ---------------- PROMPT ----------------
    if mode == "custom":
        prompt = (custom_prompt or "Extract structured data") + "\nReturn ONLY valid JSON."
    else:
        prompt = PROMPTS.get(mode)
        if not prompt:
            raise HTTPException(status_code=400, detail="Invalid mode")

    # ---------------- OLLAMA CALL ----------------
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "user",
                "content": f"{prompt}\n\nDOCUMENT TEXT:\n{extracted_text}",
            }
        ],
        "stream": False,
        "options": {
            "temperature": 0,
            "num_predict": 1024,
        },
    }

    async def call_ollama():
        async with httpx.AsyncClient(timeout=600.0) as client:
            return await client.post(
                f"{OLLAMA_HOST}/api/chat",
                json=payload,
            )

    try:
        response = await call_ollama()

        if response.status_code != 200:
            raise HTTPException(status_code=502, detail=response.text)

    except httpx.ReadTimeout:
        raise HTTPException(status_code=504, detail="Model took too long")

    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))

    # ---------------- PARSE ----------------
    try:
        data = response.json()
        raw = data.get("message", {}).get("content", "")
    except:
        raise HTTPException(status_code=500, detail="Invalid Ollama response")

    parsed = extract_json(raw)

    # ALWAYS SAFE OUTPUT
    if isinstance(parsed, dict) and parsed.get("error"):
        parsed = {"raw_text": raw}

    return {
        "success": True,
        "data": parsed,
        "raw": raw,
        "model": MODEL_NAME,
    }


# ---------------- HEALTH ----------------
@app.get("/api/health")
async def health():
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            r = await client.get(f"{OLLAMA_HOST}/api/tags")

        models = r.json().get("models", [])

        return {
            "status": "ok",
            "ollama": "connected",
            "model": MODEL_NAME,
            "model_ready": len(models) > 0,
            "available_models": [m["name"] for m in models],
        }

    except Exception as e:
        return {
            "status": "error",
            "ollama": "disconnected",
            "error": str(e),
            "model": MODEL_NAME
        }


# ---------------- FRONTEND ----------------
frontend_path = os.path.join(os.path.dirname(__file__), "..", "frontend")

if os.path.exists(frontend_path):
    app.mount("/", StaticFiles(directory=frontend_path, html=True), name="frontend")