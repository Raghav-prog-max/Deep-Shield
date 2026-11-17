from contextlib import asynccontextmanager
import logging
import uuid

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .api import build_router
from .config import settings
from .database import Database
from .jobs import JobService
from .media import MediaService

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
database = Database(settings.data_dir / "deep_shield.sqlite3")
media_service = MediaService(settings, database)
jobs = JobService(database, media_service.root)


@asynccontextmanager
async def lifespan(_: FastAPI):
    database.initialize()
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    yield
    jobs.shutdown()


app = FastAPI(
    title="DEEP SHIELD API",
    version="0.1.0",
    description="Local prototype API for media intake and explainable analysis jobs.",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.cors_origins),
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "X-Correlation-ID"],
    expose_headers=["X-Correlation-ID"],
)


@app.middleware("http")
async def correlation_id(request: Request, call_next):
    correlation = request.headers.get("X-Correlation-ID") or str(uuid.uuid4())
    request.state.correlation_id = correlation
    try:
        response = await call_next(request)
    except Exception:
        logging.getLogger(__name__).exception("Unexpected request error; correlation_id=%s", correlation)
        response = JSONResponse(status_code=500, content={
            "detail": "An unexpected server error occurred",
            "correlation_id": correlation,
        })
    response.headers["X-Correlation-ID"] = correlation
    return response


app.include_router(build_router(database, media_service, jobs), prefix="/api")
