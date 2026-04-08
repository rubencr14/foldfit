"""FastAPI application factory."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from foldfit.api.v1.datasets import router as datasets_router
from foldfit.api.v1.finetune import router as finetune_router
from foldfit.api.v1.inference import router as inference_router
from foldfit.api.v1.msa import router as msa_router


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="Foldfit API",
        description=(
            "REST API for fine-tuning OpenFold on antibody structures using LoRA/QLoRA.\n\n"
            "## Sections\n\n"
            "- **Datasets** — Create and manage SAbDab antibody structure datasets\n"
            "- **Training** — Start, monitor, and manage LoRA fine-tuning jobs\n"
            "- **Prediction** — Run structure prediction with optional LoRA adapters\n"
            "- **MSA** — Compute Multiple Sequence Alignments\n"
        ),
        version="0.1.0",
    )

    # CORS for frontend dev server
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(datasets_router)
    app.include_router(finetune_router)
    app.include_router(inference_router)
    app.include_router(msa_router)

    @app.get("/health", tags=["System"], summary="Health check")
    async def health() -> dict[str, str]:
        """Returns OK if the API server is running."""
        return {"status": "ok"}

    @app.post("/v1/gpu/clear", tags=["System"], summary="Free GPU memory")
    async def gpu_clear() -> dict:
        """Force-free all GPU memory. Use before training if GPU is full."""
        import gc
        gc.collect()
        if torch.cuda.is_available():
            before = torch.cuda.memory_allocated() / 1024**3
            torch.cuda.empty_cache()
            gc.collect()
            torch.cuda.empty_cache()
            after = torch.cuda.memory_allocated() / 1024**3
            return {"freed_gb": round(before - after, 1), "remaining_gb": round(after, 1)}
        return {"freed_gb": 0, "remaining_gb": 0}

    @app.get("/v1/gpu", tags=["System"], summary="GPU information")
    async def gpu_info() -> dict:
        """Returns GPU availability, name, and VRAM."""
        import torch

        if not torch.cuda.is_available():
            return {"available": False, "device": "cpu", "name": None, "vram_gb": 0}

        props = torch.cuda.get_device_properties(0)
        vram_total = props.total_memory / 1024**3
        vram_used = torch.cuda.memory_allocated(0) / 1024**3
        vram_free = vram_total - vram_used

        return {
            "available": True,
            "device": "cuda",
            "name": torch.cuda.get_device_name(0),
            "vram_total_gb": round(vram_total, 1),
            "vram_used_gb": round(vram_used, 1),
            "vram_free_gb": round(vram_free, 1),
            "cuda_version": torch.version.cuda or "unknown",
        }

    return app
