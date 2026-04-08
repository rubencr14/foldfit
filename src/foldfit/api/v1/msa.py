"""MSA computation endpoints.

Compute Multiple Sequence Alignments using various backends.
"""

from __future__ import annotations

from fastapi import APIRouter

from foldfit.api.schemas import MsaRequest, MsaResponse
from foldfit.application.msa_service import MsaService
from foldfit.domain.value_objects import MsaConfig
from foldfit.infrastructure.data.msa_provider import MsaProvider

router = APIRouter(prefix="/v1/msa", tags=["MSA"])


@router.post(
    "",
    response_model=MsaResponse,
    summary="Compute MSA for a sequence",
    description=(
        "Computes a Multiple Sequence Alignment for the given amino acid sequence. "
        "Backends: 'single' (dummy, fast), 'precomputed' (local .a3m files), 'colabfold' (remote API)."
    ),
)
async def compute_msa(body: MsaRequest) -> MsaResponse:
    config = MsaConfig(backend=body.backend, msa_dir=body.msa_dir)
    provider = MsaProvider(config)
    service = MsaService(provider)

    result = service.compute(body.sequence, body.pdb_id)
    msa_tensor = result["msa"]

    return MsaResponse(
        num_sequences=msa_tensor.shape[0],
        sequence_length=msa_tensor.shape[1],
        backend=body.backend,
    )
