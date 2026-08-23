from __future__ import annotations

import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse

from app.composer import AIComposer
from app.config import Settings
from app.context_store import ContextStore
from app.llm.gemini import GeminiProvider
from app.models import (
    ContextRequest,
    ContextResponse,
    HealthResponse,
    MetadataResponse,
    ReplyRequest,
    ReplyResponse,
    TeardownResponse,
    TickRequest,
    TickResponse,
)
from app.reply_service import ReplyService
from app.tick_service import TickService


def create_app(settings: Settings | None = None) -> FastAPI:
    runtime_settings = settings or Settings()

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        application.state.started_at = time.monotonic()
        application.state.store = ContextStore(runtime_settings.database_path)

        gemini_provider = GeminiProvider(
            runtime_settings.gemini_api_key,
            runtime_settings.gemini_model,
        )

        application.state.composer = AIComposer(
            provider=gemini_provider if gemini_provider.available else None
        )

        yield

        application.state.store.close()

    application = FastAPI(
        title="Magicpin AI Challenge Bot",
        version=runtime_settings.version or "0.1.0",
        lifespan=lifespan,
    )

    @application.get("/v1/healthz", response_model=HealthResponse)
    async def healthz(request: Request) -> HealthResponse:
        return HealthResponse(
            status="ok",
            uptime_seconds=int(
                time.monotonic() - request.app.state.started_at
            ),
            contexts_loaded=request.app.state.store.counts(),
        )

    @application.get("/v1/metadata", response_model=MetadataResponse)
    async def metadata() -> MetadataResponse:
        return MetadataResponse(
            team_name=runtime_settings.team_name,
            team_members=runtime_settings.parsed_team_members,
            model=runtime_settings.reported_model,
            approach=runtime_settings.approach,
            contact_email=runtime_settings.contact_email,
            version=runtime_settings.version,
            submitted_at=runtime_settings.submitted_at,
        )

    @application.post("/v1/context", response_model=ContextResponse)
    async def push_context(
        request_body: ContextRequest,
        request: Request,
    ):
        result = request.app.state.store.put(
            request_body.scope,
            request_body.context_id,
            request_body.version,
            request_body.payload,
        )

        if not result.accepted:
            response = ContextResponse(
                accepted=False,
                reason="stale_version",
                current_version=result.current_version,
            )

            return JSONResponse(
                status_code=status.HTTP_409_CONFLICT,
                content=response.model_dump(mode="json"),
            )

        return ContextResponse(
            accepted=True,
            ack_id=f"ack_{request_body.context_id}_v{request_body.version}",
            stored_at=result.stored_at,
        )

    @application.post("/v1/tick", response_model=TickResponse)
    async def tick(
        request_body: TickRequest,
        request: Request,
    ) -> TickResponse:
        service = TickService(
            request.app.state.store,
            composer=request.app.state.composer,
        )

        return service.tick(request_body)

    @application.post("/v1/reply", response_model=ReplyResponse)
    async def reply(
        request_body: ReplyRequest,
        request: Request,
    ) -> ReplyResponse:
        service = ReplyService(request.app.state.store)
        return service.handle(request_body)

    @application.post("/v1/teardown", response_model=TeardownResponse)
    async def teardown(request: Request) -> TeardownResponse:
        request.app.state.store.clear_all()
        return TeardownResponse(status="ok")

    return application


app = create_app()