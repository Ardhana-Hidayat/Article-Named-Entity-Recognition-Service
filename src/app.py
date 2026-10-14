"""
FastAPI Backend for Article Named-Entity Recognition Service.
Provides POST /extract-entities endpoint and web UI at GET /.
"""
import os
import sys
from pathlib import Path

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, field_validator
import spacy
import uvicorn

# ============================================================
# Configuration
# ============================================================
PROJECT_ROOT = Path(__file__).parent.parent
MODEL_DIR = PROJECT_ROOT / "models" / "ner_finetuned"
MAX_COMBINED_LENGTH = 50000  # Maximum combined text length
MODEL_NAME = "spacy-ner-finetuned-maybank-articles"

# Supported labels
SUPPORTED_LABELS = ["PERSON", "ORG", "GPE", "DATE", "EVENT"]

# ============================================================
# Load NER Model
# ============================================================
print(f"Loading NER model from {MODEL_DIR}...")
try:
    nlp = spacy.load(str(MODEL_DIR))
    nlp.max_length = MAX_COMBINED_LENGTH + 1000
    print(f"Model loaded successfully. Pipeline: {nlp.pipe_names}")
except Exception as e:
    print(f"ERROR: Could not load fine-tuned model: {e}")
    print("Falling back to en_core_web_sm baseline...")
    nlp = spacy.load("en_core_web_sm")
    nlp.max_length = MAX_COMBINED_LENGTH + 1000

# ============================================================
# FastAPI Application
# ============================================================
app = FastAPI(
    title="Article NER Service",
    description="Named Entity Recognition service for news articles",
    version="1.0.0",
)


# ============================================================
# Request/Response Models
# ============================================================
class ExtractRequest(BaseModel):
    title: str
    body: str


class EntityItem(BaseModel):
    text: str
    label: str
    start: int
    end: int


class MetaInfo(BaseModel):
    entity_count: int
    model: str


class ExtractResponse(BaseModel):
    entities: list[EntityItem]
    meta: MetaInfo


# ============================================================
# Helper Functions
# ============================================================
def combine_text(title: str, body: str) -> str:
    """Combine title and body with consistent separator."""
    return f"{title.strip()}\n\n{body.strip()}"


def extract_entities(text: str) -> list[dict]:
    """Run NER inference on text and return entities."""
    doc = nlp(text)
    entities = []
    for ent in doc.ents:
        if ent.label_ in SUPPORTED_LABELS:
            entities.append({
                "text": ent.text,
                "label": ent.label_,
                "start": ent.start_char,
                "end": ent.end_char,
            })
    return entities


# ============================================================
# API Endpoints
# ============================================================
@app.post("/extract-entities", response_model=ExtractResponse)
async def extract_entities_endpoint(request: Request):
    """Extract named entities from article title and body."""
    # Parse JSON body
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(
            status_code=400,
            detail={"error": "Invalid JSON", "message": "Request body must be valid JSON."}
        )
    
    # Validate required fields exist
    if not isinstance(body, dict):
        raise HTTPException(
            status_code=400,
            detail={"error": "Invalid format", "message": "Request body must be a JSON object with 'title' and 'body' fields."}
        )
    
    title = body.get("title")
    body_text = body.get("body")
    
    # Validate title
    if title is None:
        raise HTTPException(
            status_code=400,
            detail={"error": "Missing field", "message": "'title' field is required."}
        )
    if not isinstance(title, str):
        raise HTTPException(
            status_code=400,
            detail={"error": "Invalid type", "message": "'title' must be a string."}
        )
    if not title.strip():
        raise HTTPException(
            status_code=400,
            detail={"error": "Empty field", "message": "'title' must not be empty or whitespace-only."}
        )
    
    # Validate body
    if body_text is None:
        raise HTTPException(
            status_code=400,
            detail={"error": "Missing field", "message": "'body' field is required."}
        )
    if not isinstance(body_text, str):
        raise HTTPException(
            status_code=400,
            detail={"error": "Invalid type", "message": "'body' must be a string."}
        )
    if not body_text.strip():
        raise HTTPException(
            status_code=400,
            detail={"error": "Empty field", "message": "'body' must not be empty or whitespace-only."}
        )
    
    # Validate combined length
    combined = combine_text(title, body_text)
    if len(combined) > MAX_COMBINED_LENGTH:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "Text too long",
                "message": f"Combined text length ({len(combined)} chars) exceeds maximum of {MAX_COMBINED_LENGTH} chars."
            }
        )
    
    # Run NER inference
    entities = extract_entities(combined)
    
    return {
        "entities": entities,
        "meta": {
            "entity_count": len(entities),
            "model": MODEL_NAME,
        }
    }


@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    """Serve the web UI panel."""
    html_path = Path(__file__).parent / "templates" / "index.html"
    if html_path.exists():
        return HTMLResponse(content=html_path.read_text(encoding='utf-8'))
    else:
        return HTMLResponse(content=get_fallback_html())


def get_fallback_html():
    """Return embedded HTML if template file doesn't exist."""
    return """<!DOCTYPE html>
<html>
<head><title>NER Service</title></head>
<body><h1>Article NER Service</h1><p>Template not found.</p></body>
</html>"""


# ============================================================
# Error Handlers
# ============================================================
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail},
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"detail": {"error": "Internal server error", "message": str(exc)}},
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
