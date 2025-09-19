import os
import logging

from fastapi import FastAPI
import torch

from sentence_transformers import SentenceTransformer
from api.ranker import router as ranker_router
from api import ranker as ranker_module

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("main")

# Config via env
MODEL_NAME = os.environ.get("EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-0.6B")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = int(os.environ.get("EMBED_BATCH_SIZE", "32"))
MAX_RESUMES = int(os.environ.get("MAX_RESUMES", "2000"))

app = FastAPI(title="Resume Ranker Service")

# Inject config into ranker module
ranker_module.model = None
ranker_module.BATCH_SIZE = BATCH_SIZE
ranker_module.MAX_RESUMES = MAX_RESUMES

@app.on_event("startup")
def startup_event():
    global ranker_module
    logger.info("Loading model %s on %s", MODEL_NAME, DEVICE)
    ranker_module.model = SentenceTransformer(MODEL_NAME, device=DEVICE)
    logger.info("Model loaded into ranker module")

# Mount router
app.include_router(ranker_router, prefix="/api")

@app.get("/")
def root():
    return {"service": "resume-ranker", "ready": True}
