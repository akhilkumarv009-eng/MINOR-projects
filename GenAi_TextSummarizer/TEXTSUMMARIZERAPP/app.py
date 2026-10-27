
import re
from pathlib import Path

import torch
from fastapi import FastAPI, Request, HTTPException
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from transformers import T5ForConditionalGeneration, T5Tokenizer


# ==================================================
# 1. PROJECT PATHS
# ==================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "saved_summary_model"
TEMPLATE_DIR = BASE_DIR

app = FastAPI(
    title="Text Summarizer App",
    description="Text Summarization using T5",
    version="1.0.0",
)

# index.html is in the same directory as app.py
templates = Jinja2Templates(directory=str(TEMPLATE_DIR))


# ==================================================
# 2. SELECT DEVICE
# ==================================================

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")


# ==================================================
# 3. LOAD SAVED T5 MODEL
# ==================================================

if not MODEL_DIR.is_dir():
    raise FileNotFoundError(
        f"Saved model directory not found: {MODEL_DIR}"
    )

tokenizer = T5Tokenizer.from_pretrained(str(MODEL_DIR))

model = T5ForConditionalGeneration.from_pretrained(
    str(MODEL_DIR)
)

model.to(device)
model.eval()

print("T5 model loaded successfully!")


# ==================================================
# 4. INPUT VALIDATION
# ==================================================

class DialogueInput(BaseModel):
    dialogue: str = Field(
        ...,
        min_length=1,
        max_length=50000,
        description="Text that needs summarization",
    )


# ==================================================
# 5. CLEAN INPUT TEXT
# ==================================================

def clean_data(text: str) -> str:
    # Remove HTML tags
    text = re.sub(r"<[^>]*>", " ", text)

    # Remove line breaks
    text = re.sub(r"[\r\n]+", " ", text)

    # Normalize multiple spaces
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# ==================================================
# 6. SUMMARIZATION FUNCTION
# ==================================================

def summarize_dialogue(dialogue: str) -> str:
    dialogue = clean_data(dialogue)

    if not dialogue:
        raise ValueError("Please enter text to summarize.")

    inputs = tokenizer(
        dialogue,
        max_length=512,
        truncation=True,
        padding=True,
        return_tensors="pt",
    )

    # Put input tensors on the same device as the model
    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=150,
            num_beams=4,
            early_stopping=True,
        )

    summary = tokenizer.decode(
        output_ids[0],
        skip_special_tokens=True,
    )

    return summary.strip()


# ==================================================
# 7. HOMEPAGE
# ==================================================

@app.get("/")
async def home(request: Request):
    html_file = BASE_DIR / "index.html"

    if not html_file.is_file():
        raise HTTPException(
            status_code=500,
            detail=f"index.html not found at {html_file}",
        )

    return templates.TemplateResponse(
        request=request,
        name="index.html",
    )


# ==================================================
# 8. SUMMARIZE API
# ==================================================

@app.post("/summarize/")
async def summarize(dialogue_input: DialogueInput):
    try:
        summary = summarize_dialogue(
            dialogue_input.dialogue
        )

        return {
            "summary": summary,
            "device": str(device),
        }

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        print(f"Summarization failed: {exc}")

        raise HTTPException(
            status_code=500,
            detail="Summarization failed. Check the server terminal.",
        ) from exc


# ==================================================
# 9. HEALTH CHECK
# ==================================================

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model_loaded": True,
        "device": str(device),
    }
##uvicorn app:app --reload