"""Structure prediction endpoints.

Run OpenFold inference with optional LoRA adapter loading.
After each prediction the model is deleted to free GPU memory for training.
"""

from __future__ import annotations

import gc
import logging

import torch
from fastapi import APIRouter, HTTPException

from foldfit.api.schemas import PredictRequest, PredictResponse
from foldfit.application.inference_service import InferenceService
from foldfit.domain.value_objects import ModelConfig
from foldfit.infrastructure.checkpoint_store import FileCheckpointStore
from foldfit.infrastructure.openfold.adapter import OpenFoldAdapter
from foldfit.infrastructure.peft.injector import LoraInjector

router = APIRouter(prefix="/v1/predict", tags=["Prediction"])
logger = logging.getLogger(__name__)


@router.post(
    "",
    response_model=PredictResponse,
    summary="Predict protein structure",
    description=(
        "Predicts the 3D structure of a protein from its amino acid sequence using OpenFold. "
        "Optionally loads a LoRA adapter checkpoint for fine-tuned prediction. "
        "Returns predicted atom coordinates as a PDB string and per-residue pLDDT confidence. "
        "The model is unloaded after each request to free GPU memory for training."
    ),
)
async def predict(body: PredictRequest) -> PredictResponse:
    device = body.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    service = None
    try:
        service = InferenceService(
            model=OpenFoldAdapter(),
            peft=LoraInjector(),
            checkpoint=FileCheckpointStore(),
        )
        model_config = ModelConfig(weights_path=body.weights_path, device=device)
        service.load(model_config=model_config, adapter_path=body.adapter_path)

        result = service.predict_from_sequence(sequence=body.sequence, device=device)
        return PredictResponse(**result)

    except ImportError as e:
        raise HTTPException(
            status_code=503,
            detail=f"OpenFold not available: {e}. Install openfold to enable prediction.",
        )
    except Exception as e:
        logger.error(f"Prediction failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Always free GPU memory after prediction
        del service
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info(
            f"GPU memory after cleanup: "
            f"{torch.cuda.memory_allocated() / 1024**3:.1f} GB allocated"
            if torch.cuda.is_available() else "No CUDA"
        )
