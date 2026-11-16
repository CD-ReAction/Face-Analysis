import logging
import os
import threading
from contextlib import asynccontextmanager

from fastapi import BackgroundTasks, FastAPI, status
from pydantic import BaseModel, Field, HttpUrl

from app.services.face_analysis.analyzer import FaceVideoAnalyzer
from app.services.face_analysis.worker import _resolve_accelerator, run_analysis_job

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Opt-in (FACE_ANALYZER_PRELOAD=1, set by Dockerfile.cpu): load the models once
# at startup and reuse them for every request, like app/handler.py does for
# RunPod. Off by default so the API keeps building a fresh analyzer per job.
_analyzer: FaceVideoAnalyzer | None = None
# The analyzer is stateful (PersonTracker), so a shared instance must only
# run one job at a time.
_analyzer_lock = threading.Lock()


def _preload_enabled() -> bool:
    return os.getenv("FACE_ANALYZER_PRELOAD", "0").strip().lower() in {
        "1", "true", "yes", "on",
    }


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global _analyzer
    if _preload_enabled():
        device, onnx_providers = _resolve_accelerator()
        logger.info(
            "Preloading analyzer: device=%s onnx_providers=%s",
            device or "cpu",
            onnx_providers or ["CPUExecutionProvider"],
        )
        analyzer = FaceVideoAnalyzer(device=device, onnx_providers=onnx_providers)
        analyzer.warmup()
        _analyzer = analyzer
    yield


def _run_job(payload: dict) -> None:
    if _analyzer is None:
        run_analysis_job(payload)
        return
    with _analyzer_lock:
        run_analysis_job(payload, analyzer=_analyzer)


class KnownActorPayload(BaseModel):
    actor_id: int = Field(..., ge=1)
    # Multi-exemplar gallery (each inner list = one 512-d ArcFace embedding).
    # BE accumulates new angles per actor across rehearsals; the analyzer
    # matches max-over-N so frontal/profile/makeup variants all contribute.
    face_templates: list[list[float]] = Field(..., min_length=1)


class AnalyzeRequest(BaseModel):
    video_id: int = Field(..., ge=1)
    session_id: int = Field(..., ge=1)
    s3_key: str = Field(..., min_length=1)
    s3_url: HttpUrl
    callback_url: HttpUrl
    known_actors: list[KnownActorPayload] = Field(default_factory=list)
    # S3 key prefix BE wants thumbnails uploaded under, e.g. "42/27/" — the
    # analyzer appends "thumb-{idx}.jpg" so video + thumbnails end up in the
    # same per-session folder. Optional for back-compat with older callers
    # that have not yet been updated; in that case the legacy
    # "thumbnails/{video_id}/" layout is used instead.
    thumbnail_dir: str | None = None


class AnalyzeAcceptedResponse(BaseModel):
    video_id: int
    analysis_status: str


app = FastAPI(
    title="Face Analyzer API",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post(
    "/analyze",
    response_model=AnalyzeAcceptedResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
def analyze(
    request: AnalyzeRequest,
    background_tasks: BackgroundTasks,
) -> AnalyzeAcceptedResponse:
    background_tasks.add_task(_run_job, request.model_dump(mode="json"))
    return AnalyzeAcceptedResponse(
        video_id=request.video_id,
        analysis_status="accepted",
    )
