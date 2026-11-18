"""DEEP SHIELD API — Modules 01–04 (Threat, Deepfake, Video Events, Forensics).

    uvicorn deep_shield.api.main:app --host 0.0.0.0 --port 8002
    open http://localhost:8002/docs
"""
from contextlib import asynccontextmanager
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from deep_shield.api.routes_deepfake import get_engine, router as m02_router
from deep_shield.api.routes_forensics import get_engine as get_m04_engine, router as m04_router
from deep_shield.api.routes_intelligence import router as intelligence_router
from deep_shield.api.routes_threat import get_engine as get_m01_engine, router as m01_router
from deep_shield.api.routes_video_events import get_engine as get_m03_engine, router as m03_router


@asynccontextmanager
async def lifespan(_app: FastAPI):
    try:
        get_m01_engine().health()
        get_engine().warmup()      # load M02 SigLIP2 + trained face head + YuNet
        get_m03_engine().health()  # load M03 PyTorch Temporal GRU weights
        get_m04_engine().health()  # load M04 calibrated forensics parameters
    except Exception as exc:       # noqa: BLE001 — /api/v1/intelligence/models reports details
        print(f"[DEEP SHIELD] warm-up warning: {exc}")
    yield


_cors_origins = [
    o.strip()
    for o in os.getenv("DEEPSHIELD_CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000").split(",")
    if o.strip()
]
app = FastAPI(title="DEEP SHIELD — Modules 01–04 API", version="0.4.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=_cors_origins, allow_methods=["*"], allow_headers=["*"])
app.include_router(m02_router)
app.include_router(m01_router)
app.include_router(m03_router)
app.include_router(m04_router)
app.include_router(intelligence_router)
