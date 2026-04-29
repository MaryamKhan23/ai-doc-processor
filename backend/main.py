import os
import base64
import httpx
import json
import re
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from typing import Optional
import fitz

app = FastAPI(title="AI Document Processor (Local)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://ollama:11434")
MODEL_NAME = os.environ.get("MODEL_NAME", "llava:7b")


# ---------------- PROMPTS ----------------
PROMPTS = {
    "invoice": """Extract invoice data. Return ONLY valid JSON:
{"vendor_name":"","vendor_address":"","invoice_number":"","invoice_date":"","due_date":"","subtotal":"","tax":"","total":"","currency":"","line_items":[]}
No extra text.""",

    "resume": """You are a strict OCR system.

RULES:
- ONLY use visible text
- NEVER guess anything
- NEVER use placeholders like "John Doe"
- If missing → null

Return ONLY JSON:
{
  "full_name": null,
  "email": null,
  "phone": null,
  "location": null,
  "linkedin": null,
  "skills": [],
  "experience": [],
  "education": []
}
""",

    "research": """Extract research data. Return ONLY JSON:
{"title":"","authors":[],"summary":"","key_findings":[]}
"""
}


# ---------------- PDF → IMAGE ----------------
def pdf_to_base64_image(file_bytes: bytes):
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    page = doc[0]
    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
    return base64.b64encode(pix.tobytes("png")).decode("utf-8")


# ---------------- STRICT JSON FIXER ----------------
def clean_json_keys(text: str):
    """
    Fix common LLM mistakes like:
    "education[]" → "education"
    """
    text = text.replace('"education[]"', '"education"')
    text = text.replace("'education[]'", '"education"')
    return text

def extract_json(text: str):
    if not text:
        return {"error": "empty_response"}

    # FIX malformed keys BEFORE parsing
    text = text.replace('"education[]"', '"education"')
    text = text.replace("'education[]'", '"education"')

    # remove markdown fences if any
    text = text.replace("```json", "").replace("```", "").strip()

    try:
        return json.loads(text)
    except:
        pass

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        cleaned = match.group()

        # second safety cleanup
        cleaned = cleaned.replace('"education[]"', '"education"')

        try:
            return json.loads(cleaned)
        except:
            pass

    return {
        "error": "invalid_json",
        "raw_output": text
    }


# ---------------- API ----------------
@app.post("/api/process")
async def process_document(
    file: UploadFile = File(...),
    mode: str = Form(...),
    custom_prompt: Optional[str] = Form(None),
):

    file_bytes = await file.read()

    if len(file_bytes) > 20 * 1024 * 1024:
        return JSONResponse(status_code=400, content={"error": "file_too_large"})

    is_pdf = file.filename.lower().endswith(".pdf")

    try:
        b64_image = pdf_to_base64_image(file_bytes) if is_pdf else base64.b64encode(file_bytes).decode()
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": "image_error", "details": str(e)})

    prompt = PROMPTS.get(mode)
    if mode == "custom":
        prompt = (custom_prompt or "Extract data") + "\nReturn ONLY JSON."

    if not prompt:
        return JSONResponse(status_code=400, content={"error": "invalid_mode"})

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
            "temperature": 0
        },
    }

    async with httpx.AsyncClient(timeout=180.0) as client:
        r = await client.post(f"{OLLAMA_HOST}/api/chat", json=payload)

    if r.status_code != 200:
        return JSONResponse(status_code=502, content={"error": "ollama_failed", "details": r.text})

    try:
        raw = r.json()["message"]["content"]
    except:
        return JSONResponse(status_code=500, content={"error": "bad_ollama_response"})

    parsed = extract_json(raw)

    return {
        "success": True,
        "data": parsed,
        "model": MODEL_NAME
    }


@app.get("/api/health")
async def health():
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            await client.get(f"{OLLAMA_HOST}/api/tags")
        return {"status": "ok"}
    except:
        return {"status": "error"}


# ---------------- FRONTEND ----------------
frontend_path = os.path.join(os.path.dirname(__file__), "..", "frontend")

if os.path.exists(frontend_path):
    app.mount("/", StaticFiles(directory=frontend_path, html=True), name="frontend")