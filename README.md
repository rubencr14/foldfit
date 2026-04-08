# Foldfit

Parameter-efficient fine-tuning of OpenFold on antibody structures using LoRA and QLoRA, built in pure PyTorch.

Foldfit provides a clean, self-contained framework to adapt OpenFold's protein structure prediction model specifically for antibodies. It downloads curated antibody structures from [SAbDab](https://opig.stats.ox.ac.uk/webapps/sabdab-sabpred/sabdab/), injects low-rank adapters (LoRA) into the Evoformer attention layers, and fine-tunes with a full training loop that supports mixed precision, gradient accumulation, early stopping, and more.

---

## How It Works

### 1. LoRA Injection (Parameter-Efficient Fine-Tuning)

Instead of updating all ~93M parameters of OpenFold, Foldfit freezes the entire base model and injects small trainable adapters into specific linear layers. This is the LoRA (Low-Rank Adaptation) technique:

- For each target layer (by default `linear_q` and `linear_v` in the Evoformer attention), the original `nn.Linear` is replaced with a `LoRALinear` module.
- `LoRALinear` keeps the original frozen weight `W` and adds two small trainable matrices:
  - `lora_A` with shape `[rank, in_features]` (initialized with Kaiming uniform)
  - `lora_B` with shape `[out_features, rank]` (initialized to zero)
- The forward pass computes: **y = W @ x + (alpha / rank) * B @ A @ x**
- Since `B` starts at zero, the model initially behaves identically to the original.
- Only `lora_A` and `lora_B` are trainable (~600K parameters for rank=8), making fine-tuning feasible on a single GPU with 12GB VRAM.

For **QLoRA**, the frozen base weights are additionally quantized to 4-bit (NF4) using bitsandbytes before injecting LoRA adapters, reducing memory usage even further.

After training, LoRA weights can be **merged** into the base weights (`W_new = W + scaling * B @ A`) for zero-overhead inference, or kept separate for easy adapter swapping.

### 2. The Injector

The `LoraInjector` walks through the model's module tree and replaces every `nn.Linear` whose name matches a target substring. For example, with `target_modules: ["linear_q", "linear_v"]`, it will find and replace layers like:

```
evoformer.blocks.0.msa_att.linear_q  -> LoRALinear
evoformer.blocks.0.msa_att.linear_v  -> LoRALinear
evoformer.blocks.1.msa_att.linear_q  -> LoRALinear
...
```

All other layers remain frozen and untouched.

### 3. Training Loop

The `Trainer` handles the full training lifecycle:

- **Separate learning rates**: LoRA parameters and the prediction head get independent learning rates (typically `lr_lora=5e-5`, `lr_head=5e-4`).
- **Optimizer**: AdamW with configurable weight decay.
- **Scheduler**: Linear warmup followed by cosine, linear, or constant decay.
- **Mixed precision (AMP)**: Uses `torch.autocast` and `GradScaler` for faster training and lower memory.
- **Gradient accumulation**: Accumulates gradients over N batches before stepping, enabling effective larger batch sizes.
- **Gradient clipping**: Clips gradient norms to prevent training instability.
- **EMA (Exponential Moving Average)**: Optionally maintains a shadow copy of parameters for smoother evaluation.
- **Early stopping**: Monitors validation loss and stops training if it doesn't improve for `patience` epochs.
- **Checkpointing**: Saves the best model (LoRA weights + head + optimizer state) for resumption.

**OpenFold constraint**: The model must remain in `eval()` mode even during training because EvoformerStack's chunked operations require it. Gradients still flow through `requires_grad=True` parameters.

### 4. Data Pipeline

**SAbDab Repository**: Queries the Structural Antibody Database for curated antibody PDB IDs, filters by resolution, and downloads PDB files from RCSB. Files are cached locally for reuse.

**OpenFold Featurizer**: Converts PDB files into OpenFold-compatible feature dictionaries containing:
- Amino acid types, residue indices
- All-atom positions and masks (37 atoms per residue)
- MSA features (multiple sequence alignment)
- Template placeholders
- Ground truth labels for structure loss (FAPE, chi angles, distogram, etc.)

**MSA Providers** (three backends):
- `single`: Dummy MSA with just the query sequence (fast prototyping, no external dependencies).
- `precomputed`: Loads `.a3m` files from a local directory.
- `colabfold`: Queries the ColabFold MMseqs2 API for live MSA computation.

**StructureDataset**: A PyTorch `Dataset` that loads and featurizes PDB files on-the-fly, with a custom collate function that pads variable-length sequences.

### 5. Inference

The `InferenceService` loads the base OpenFold model and optionally applies a saved LoRA adapter:

1. Load base model weights.
2. Inject LoRA structure (same rank/alpha/targets as training).
3. Load saved `lora_A`/`lora_B` weights.
4. Optionally merge adapters into base weights for faster inference.
5. Run forward pass and return structure predictions with pLDDT confidence scores.

### 6. API and CLI

**FastAPI** server with versioned endpoints:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/finetune` | POST | Start a fine-tuning job (runs in background, returns job ID) |
| `/v1/finetune/{id}` | GET | Get job status and metrics |
| `/v1/predict` | POST | Run structure prediction |
| `/v1/msa` | POST | Compute MSA for a sequence |
| `/health` | GET | Health check |

**Typer CLI** with three subcommands:

```bash
python scripts/finetune.py finetune --config config.yaml
python scripts/finetune.py predict MKWVTFISLLLLFSSAYS --adapter-path ./checkpoints/final/peft
python scripts/finetune.py msa MKWVTFISLLLLFSSAYS --backend colabfold
```

---

## Architecture

The project follows a clean Domain-Driven Design with three layers:

```
src/foldfit/
├── domain/                          # Pure business logic, no dependencies
│   ├── value_objects.py             # Immutable configs: LoraConfig, TrainingConfig, etc.
│   ├── interfaces.py                # ABC ports: ModelPort, PeftPort, DatasetPort, MsaPort
│   └── entities.py                  # TrunkOutput, FinetuneJob, TrainedModel
├── application/                     # Use case orchestration
│   ├── finetune_service.py          # Coordinates: load model -> inject LoRA -> train -> save
│   ├── inference_service.py         # Load model + adapter -> predict
│   └── msa_service.py              # MSA computation
├── infrastructure/                  # Concrete implementations
│   ├── peft/
│   │   ├── lora_linear.py           # LoRALinear nn.Module (pure PyTorch)
│   │   └── injector.py             # Walks model tree, replaces target layers
│   ├── training/
│   │   ├── trainer.py               # Full training loop
│   │   └── scheduler.py            # Warmup + cosine/linear/constant
│   ├── openfold/
│   │   ├── adapter.py               # Wraps AlphaFold behind ModelPort
│   │   ├── featurizer.py           # PDB -> feature dict
│   │   └── loss.py                 # Structure loss (FAPE, pLDDT, etc.)
│   ├── data/
│   │   ├── sabdab_repository.py     # SAbDab query + PDB download
│   │   ├── structure_dataset.py     # PyTorch Dataset + collate
│   │   └── msa_provider.py         # single / precomputed / colabfold
│   └── checkpoint_store.py          # Save/load LoRA + head + training state
└── api/
    ├── app.py                       # FastAPI factory
    ├── schemas.py                   # Request/response models
    └── v1/                          # Versioned endpoints
```

All external dependencies (OpenFold, SAbDab API, filesystem) are behind abstract interfaces in `domain/interfaces.py`. The domain layer has zero infrastructure dependencies. This makes every component independently testable with mocks.

---

## Quick Start

```bash
# Install
pip install -e ".[dev]"

# Run fine-tuning (requires OpenFold + GPU)
python scripts/finetune.py finetune --config config.yaml

# Compute MSA (works without GPU)
python scripts/finetune.py msa MKWVTFISLLLLFSSAYS --backend single

# Start API server
make run
# Then open http://localhost:8000/docs for Swagger UI
```

## Configuration

All settings live in `config.yaml` and are validated with Pydantic:

```yaml
model:
  weights_path: null         # Path to pretrained OpenFold weights
  head: structure            # "structure" or "affinity"
  device: cuda

data:
  sabdab_dir: ./data/sabdab  # Local cache for downloaded PDBs
  max_structures: 200        # Max antibodies to use
  max_seq_len: 256           # Crop sequences longer than this
  resolution_max: 3.0        # Only use structures with resolution <= 3A
  val_frac: 0.1
  split_seed: 42

training:
  epochs: 20
  learning_rate: 5.0e-5
  lr_lora: 5.0e-5            # LoRA-specific learning rate
  lr_head: 5.0e-4            # Head-specific learning rate (10x default)
  scheduler: cosine           # cosine / linear / constant
  warmup_steps: 100
  accumulation_steps: 4       # Effective batch size = batch_size * 4
  amp: true                   # Mixed precision
  early_stopping_patience: 5
  gradient_checkpointing: true

lora:
  rank: 8                     # Low-rank dimension (2-64)
  alpha: 16.0                 # Scaling factor (scaling = alpha / rank = 2.0)
  dropout: 0.0
  target_modules:             # Which layers to adapt
    - linear_q
    - linear_v

msa:
  backend: single             # single / precomputed / colabfold

output:
  checkpoint_dir: ./checkpoints
```

## Testing

```bash
make test              # All tests with coverage (44 tests)
make test-unit         # Unit tests only
make test-integration  # API integration tests
make lint              # Ruff linting
make typecheck         # MyPy type checking
```

## Docker

```bash
docker compose up --build
# API available at http://localhost:8000
```
