"""DEEP SHIELD API — Modules 01–04 (Threat, Deepfake, Video Events, Forensics).

    uvicorn deep_shield.api.main:app --host 0.0.0.0 --port 8002
    open http://localhost:8002/docs
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from deep_shield.api.routes_deepfake import get_engine, router as m02_router
from deep_shield.api.routes_forensics import router as m04_router
from deep_shield.api.routes_threat import router as m01_router
from deep_shield.api.routes_video_events import router as m03_router


@asynccontextmanager
async def lifespan(_app: FastAPI):
    try:
        get_engine().warmup()      # load the M02 face model once so the first request is fast
    except Exception as exc:       # noqa: BLE001 — /api/v1/m02/health reports the details
        print(f"[M02] warm-up failed: {exc}")
    yield


app = FastAPI(title="DEEP SHIELD — Modules 01–04 API", version="0.4.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:3000"], allow_methods=["*"], allow_headers=["*"])
app.include_router(m02_router)
app.include_router(m01_router)
app.include_router(m03_router)
app.include_router(m04_router)
