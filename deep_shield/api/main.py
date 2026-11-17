"""DEEP SHIELD API — Module 02 (Deepfake & Image Integrity) + Module 04 (Visual Forensics).

    uvicorn deep_shield.api.main:app --host 0.0.0.0 --port 8002
    open http://localhost:8002/docs
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from deep_shield.api.routes_deepfake import get_engine, router as m02_router
from deep_shield.api.routes_forensics import router as m04_router


@asynccontextmanager
async def lifespan(_app: FastAPI):
    try:
        get_engine().warmup()      # load the M02 face model once so the first request is fast
    except Exception as exc:       # noqa: BLE001 — /api/v1/m02/health reports the details
        print(f"[M02] warm-up failed: {exc}")
    yield


app = FastAPI(title="DEEP SHIELD — Module 02 + Module 04 API", version="0.3.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:3000"], allow_methods=["*"], allow_headers=["*"])
app.include_router(m02_router)
app.include_router(m04_router)
