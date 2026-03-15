from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi import HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .inference import predict_sentiment
from .inference import ArtifactLoadError


PROJECT_ROOT = Path(__file__).resolve().parents[1]

app = FastAPI(title="Sentiment Analysis - Word2Vec")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class TextInput(BaseModel):
    text: str


@app.get("/", response_class=FileResponse)
def read_root():
    """
    Serve the simple HTML page with a textarea and button.
    """
    index_path = PROJECT_ROOT / "index.html"
    return FileResponse(index_path)


@app.post("/predict")
def predict(input_data: TextInput):
    """
    Return sentiment prediction for a single sentence.
    """
    try:
        result = predict_sentiment(input_data.text)
        return result
    except ArtifactLoadError as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
    except Exception as e:
        # For local/dev use: surface the underlying error to the client
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}") from e


