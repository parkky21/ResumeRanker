from typing import List, Optional
import logging
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import numpy as np
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from schemas.models import Resume

logger = logging.getLogger("api.ranker")
router = APIRouter()

# Request/response Pydantic models
class RankRequest(BaseModel):
    job_description: str
    resumes: List[Resume]
    top_k: Optional[int] = None

class RankedResume(BaseModel):
    candidate_id: str
    name: str
    score: float
    resume: Resume

class RankResponse(BaseModel):
    total_resumes: int
    returned: int
    results: List[RankedResume]

# Globals set by main.py
model = None
BATCH_SIZE = 32
MAX_RESUMES = 1000

# ------------------------------
# Helper: serialize resume -> text
# ------------------------------
def serialize_resume(r: Resume) -> str:
    parts = []
    if r.summary:
        parts.append(r.summary)

    for e in r.experience:
        exp_parts = [e.title.strip(), e.company.strip()]
        if e.description:
            exp_parts.append(e.description.strip())
        parts.append(" | ".join([p for p in exp_parts if p]))

    if r.skills:
        skill_strs = []
        for s in r.skills:
            if s.level:
                skill_strs.append(f"{s.name} ({s.level})")
            else:
                skill_strs.append(s.name)
        parts.append("Skills: " + ", ".join(skill_strs))

    if r.projects:
        proj_strs = []
        for p in r.projects:
            s = p.title
            if p.description:
                s = s + " - " + p.description
            if p.link:
                s = s + f" ({p.link})"
            proj_strs.append(s)
        parts.append("Projects: " + " || ".join(proj_strs))

    if r.certifications:
        parts.append("Certifications: " + ", ".join(r.certifications))

    if r.education:
        edu_strs = []
        for ed in r.education:
            s = ed.degree + ", " + ed.institution
            if ed.year:
                s = s + f" ({ed.year})"
            edu_strs.append(s)
        parts.append("Education: " + " ; ".join(edu_strs))

    parts.append(f"Candidate: {r.name}")

    return "\n".join(parts)


def _batch_encode(texts: List[str], batch_size: int = BATCH_SIZE):
    global model
    encs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i: i + batch_size]
        emb = model.encode(batch, convert_to_tensor=True, show_progress_bar=False)
        encs.append(emb)
    return torch.cat(encs, dim=0)

def cosine_similarity_matrix(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a_norm = F.normalize(a, p=2, dim=1)
    b_norm = F.normalize(b, p=2, dim=1)
    return a_norm @ b_norm.T

# ------------------------------
# Router endpoints
# ------------------------------
@router.get("/healthz")
async def healthz():
    return {"status": "ok"}

@router.get("/readyz")
async def readyz():
    if model is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    return {"ready": True}

@router.post("/rank", response_model=RankResponse)
async def rank(req: RankRequest):
    if not req.job_description.strip():
        raise HTTPException(status_code=400, detail="job_description is empty")

    n = len(req.resumes)
    if n == 0:
        raise HTTPException(status_code=400, detail="No resumes provided")
    if n > MAX_RESUMES:
        raise HTTPException(status_code=413, detail=f"Too many resumes. Max allowed is {MAX_RESUMES}")

    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Serialize
    resume_texts = [serialize_resume(r) for r in req.resumes]

    try:
        job_emb = model.encode([req.job_description], convert_to_tensor=True, show_progress_bar=False)
        resume_embs = _batch_encode(resume_texts)
    except Exception as e:
        logger.exception("Embedding failure")
        raise HTTPException(status_code=500, detail=f"Embedding failure: {e}")

    try:
        sims = cosine_similarity_matrix(job_emb, resume_embs).squeeze(0).cpu().numpy()
    except Exception as e:
        logger.exception("Similarity computation failed")
        raise HTTPException(status_code=500, detail=f"Similarity computation failure: {e}")

    idx_sorted = np.argsort(-sims)
    top_k = req.top_k if req.top_k is not None else n
    top_k = min(max(1, top_k), n)

    results = []
    for rank_idx in range(top_k):
        i = int(idx_sorted[rank_idx])
        cand = req.resumes[i]
        results.append(
            RankedResume(candidate_id=cand.candidate_id, name=cand.name, score=float(sims[i]), resume=cand)
        )

    return RankResponse(total_resumes=n, returned=len(results), results=results)
