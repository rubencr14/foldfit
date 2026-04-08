# Foldfit

Parameter-efficient fine-tuning of OpenFold on antibody structures using LoRA, built with PyTorch and OpenFold's native modules.

Foldfit adapts OpenFold's protein structure prediction model specifically for antibodies. It downloads curated antibody structures from [SAbDab](https://opig.stats.ox.ac.uk/webapps/sabdab-sabpred/sabdab/), injects low-rank adapters (LoRA) into the Evoformer attention layers, and fine-tunes with the full AlphaFold2 loss formulation (FAPE + distogram + masked MSA + supervised chi + pLDDT). MSA computation supports local tools with custom databases like [OAS](http://opig.stats.ox.ac.uk/webapps/oas/) for deep antibody-specific alignments.

---

## How It Works

### 1. LoRA Injection (Parameter-Efficient Fine-Tuning)

Instead of updating all ~93M parameters of OpenFold, Foldfit freezes the entire base model and injects small trainable adapters into specific linear layers:

- For each target layer (by default `linear_q` and `linear_v` in the Evoformer attention), the original `nn.Linear` is replaced with a `LoRALinear` module.
- `LoRALinear` keeps the original frozen weight `W` and adds two small trainable matrices:
  - `lora_A` with shape `[rank, in_features]` (initialized with Kaiming uniform)
  - `lora_B` with shape `[out_features, rank]` (initialized to zero)
- The forward pass computes: **y = W @ x + (alpha / rank) * B @ A @ x**
- Only `lora_A` and `lora_B` are trainable (~600K parameters for rank=8), making fine-tuning feasible on a single GPU.

After training, LoRA weights can be **merged** into the base weights for zero-overhead inference, or kept separate for adapter swapping.

### 2. Loss Function

Uses OpenFold's native `AlphaFoldLoss` with the full AlphaFold2 loss formulation:

| Loss Term | Weight | Description |
|-----------|--------|-------------|
| FAPE (backbone + sidechain) | 1.0 | Frame Aligned Point Error — primary structure loss |
| Distogram | 0.3 | Pairwise CB distance distribution (64 bins) |
| Masked MSA | 2.0 | BERT-style sequence recovery from MSA |
| Supervised chi | 1.0 | Torsion angle prediction |
| pLDDT | 0.01 | Confidence prediction via lDDT binning |
| Violations | 0.0 | Bond/angle/clash violations (disabled by default) |

Falls back to partial loss computation if model outputs are incomplete (e.g., missing logits).

### 3. Evaluation Metrics

Computed during validation using OpenFold's native utilities:

- **CA-RMSD**: After SVD superimposition (`openfold.utils.superimposition`)
- **GDT-TS**: Global Distance Test at 1, 2, 4, 8 A thresholds
- **pLDDT**: Predicted confidence via `openfold.utils.loss.compute_plddt`

Metrics are logged each epoch alongside loss terms.

### 4. Training Loop

The `Trainer` handles the full training lifecycle:

- **Separate learning rates** for LoRA parameters and prediction head
- **AdamW optimizer** with configurable weight decay
- **Scheduler**: Linear warmup + cosine/linear/constant decay
- **Mixed precision (AMP)** with `torch.autocast` and `GradScaler`
- **Gradient accumulation** for effective larger batch sizes
- **Gradient clipping** to prevent instability
- **EMA (Exponential Moving Average)** for smoother evaluation
- **Early stopping** on validation loss
- **Gradient checkpointing** (40-60% VRAM savings)
- **Checkpointing** of best model (LoRA weights + optimizer state)

**OpenFold constraint**: The model remains in `eval()` mode even during training because EvoformerStack's chunked operations require it. Gradients still flow through `requires_grad=True` parameters.

### 5. MSA Pipeline

MSA (Multiple Sequence Alignment) provides coevolutionary signal that OpenFold uses to predict 3D contacts. Four backends:

| Backend | Use Case | Dependencies |
|---------|----------|-------------|
| `single` | Fast prototyping, no alignment | None |
| `precomputed` | Load cached `.a3m` / `.msa.pt` files | None |
| `colabfold` | Query ColabFold MMseqs2 server | Network (public or self-hosted) |
| `local` | Run MMseqs2/HHblits/JackHMMER locally | Tool binary + sequence database |

**For antibodies**, we recommend using the `local` backend with the [OAS (Observed Antibody Space)](http://opig.stats.ox.ac.uk/webapps/oas/) database. Generic databases (UniRef, BFD) have poor coverage of CDR regions — OAS provides thousands of antibody-specific hits with real variability in CDR1/CDR2/CDR3.

```yaml
msa:
  backend: local
  tool: mmseqs2
  database_paths:
    - /data/oas/oas_db           # antibody-specific
    - /data/uniref30/uniref30    # general proteins
  n_cpu: 8
```

Multiple databases are searched in order and results merged into a single MSA.

MSAs can be pre-generated via the CLI:

```bash
# With ColabFold public API
python scripts/finetune.py generate-msa --pdb-dir data/sabdab --output-dir data/msa

# With local MMseqs2 + OAS
python scripts/finetune.py generate-msa --pdb-dir data/sabdab --output-dir data/msa \
    --backend local --database /data/oas/oas_db --database /data/uniref30/uniref30

# With self-hosted ColabFold server
python scripts/finetune.py generate-msa --pdb-dir data/sabdab --output-dir data/msa \
    --backend colabfold --colabfold-server http://my-server:8080
```

### 6. Data Pipeline

**SAbDab Repository**: Queries the Structural Antibody Database for curated antibody PDB IDs, filters by resolution, and downloads PDB files from RCSB.

**OpenFold Featurizer**: Converts PDB files into feature dictionaries using OpenFold's native modules:
- PDB parsing via `openfold.np.protein.from_pdb_string()`
- Ground truth transforms via `openfold.data.data_transforms` (atom14 masks, backbone frames, chi angles, pseudo-beta)
- MSA features via `data_transforms.make_msa_feat()` (23-class one-hot + deletion features)
- Auto-detects antibody chains (H, L) with fallback

**StructureDataset**: PyTorch `Dataset` that loads and featurizes PDB files with MSA integration. Custom collate pads variable-length sequences.

### 7. Inference

The `InferenceService` loads the base OpenFold model and optionally applies a saved LoRA adapter:

1. Load base model weights.
2. Inject LoRA structure (same rank/alpha/targets as training).
3. Load saved `lora_A`/`lora_B` weights.
4. Optionally merge adapters into base weights for faster inference.
5. Output PDB string via `openfold.np.protein.to_pdb()` with pLDDT as B-factors.

### 8. API and CLI

**FastAPI** server with versioned endpoints:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/finetune` | POST | Start a fine-tuning job |
| `/v1/finetune/{id}` | GET | Get job status and metrics |
| `/v1/predict` | POST | Run structure prediction |
| `/v1/msa` | POST | Compute MSA for a sequence |
| `/health` | GET | Health check |

**Typer CLI** with six commands:

```bash
# Download antibody structures from SAbDab/RCSB
python scripts/finetune.py download -n 200 -r 2.5 -t nanobody

# Generate MSAs for downloaded structures
python scripts/finetune.py generate-msa --pdb-dir data/sabdab --output-dir data/msa \
    --backend local --database /data/oas/oas_db

# Run fine-tuning
python scripts/finetune.py finetune --config config.yaml

# Predict structure (outputs PDB to stdout or file)
python scripts/finetune.py predict EVQLVESGG... --adapter-path ./checkpoints/final/peft -o pred.pdb

# Evaluate on ground-truth structures (CA-RMSD, GDT-TS, pLDDT)
python scripts/finetune.py evaluate data/sabdab --adapter-path ./checkpoints/final/peft

# Compute MSA for a single sequence
python scripts/finetune.py msa EVQLVESGG... --backend colabfold -o query.msa.pt
```

---

## Architecture

Clean Domain-Driven Design with three layers:

```
src/foldfit/
├── domain/                          # Pure business logic, no dependencies
│   ├── value_objects.py             # Immutable configs: LoraConfig, TrainingConfig, MsaConfig...
│   ├── interfaces.py                # ABC ports: ModelPort, PeftPort, DatasetPort, MsaPort
│   └── entities.py                  # TrunkOutput, FinetuneJob, TrainedModel
├── application/                     # Use case orchestration
│   ├── finetune_service.py          # Load model -> inject LoRA -> train -> save
│   ├── inference_service.py         # Load model + adapter -> predict
│   └── msa_service.py              # MSA computation
├── infrastructure/                  # Concrete implementations (OpenFold-backed)
│   ├── peft/
│   │   ├── lora_linear.py           # LoRALinear nn.Module
│   │   └── injector.py             # Walks model tree, replaces target layers
│   ├── training/
│   │   ├── trainer.py               # Full training loop with metrics logging
│   │   ├── scheduler.py            # Warmup + cosine/linear/constant
│   │   └── checkpointing.py        # Gradient checkpointing
│   ├── openfold/
│   │   ├── adapter.py               # Wraps AlphaFold behind ModelPort
│   │   ├── featurizer.py           # PDB -> features (delegates to OpenFold transforms)
│   │   ├── loss.py                 # Wraps AlphaFoldLoss with partial fallback
│   │   ├── metrics.py              # RMSD, GDT-TS, pLDDT (uses OpenFold superimposition)
│   │   └── pdb_writer.py           # Delegates to openfold.np.protein.to_pdb()
│   ├── data/
│   │   ├── sabdab_repository.py     # SAbDab query + PDB download
│   │   ├── structure_dataset.py     # PyTorch Dataset + MSA integration + collate
│   │   └── msa_provider.py         # single / precomputed / colabfold / local
│   └── checkpoint_store.py          # Save/load LoRA + head + training state
├── api/
│   ├── app.py                       # FastAPI factory
│   ├── schemas.py                   # Request/response models
│   └── v1/                          # Versioned endpoints
└── config.py                        # YAML config loader
```

---

## Quick Start

### Requirements

- Python >= 3.10
- PyTorch >= 2.0
- [OpenFold](https://github.com/aqlaboratory/openfold) (required)
- GPU with >= 12GB VRAM (for training with LoRA + gradient checkpointing)

### Install

```bash
pip install -e ".[dev]"
# OpenFold must be installed separately — see their repo for instructions
```

### Fine-tune

```bash
# 1. Download antibody structures
python scripts/finetune.py download -n 200 -r 3.0

# 2. Generate MSAs (optional but recommended)
python scripts/finetune.py generate-msa --pdb-dir data/sabdab --output-dir data/msa

# 3. Update config.yaml to point MSA to precomputed
#    msa:
#      backend: precomputed
#      msa_dir: ./data/msa

# 4. Run fine-tuning
python scripts/finetune.py finetune --config config.yaml
```

### Predict

```bash
python scripts/finetune.py predict EVQLVESGG... --adapter-path ./checkpoints/final/peft
```

### API Server

```bash
make run
# Swagger UI at http://localhost:8000/docs
```

## Configuration

All settings in `config.yaml`, validated with Pydantic:

```yaml
model:
  weights_path: null               # Path to pretrained OpenFold weights
  head: structure
  device: cuda

data:
  sabdab_dir: ./data/sabdab
  max_structures: 200
  max_seq_len: 256
  resolution_max: 3.0
  val_frac: 0.1

training:
  epochs: 20
  learning_rate: 5.0e-5
  lr_lora: 5.0e-5
  lr_head: 5.0e-4
  scheduler: cosine
  warmup_steps: 100
  accumulation_steps: 4
  amp: true
  early_stopping_patience: 5
  gradient_checkpointing: true

lora:
  rank: 8
  alpha: 16.0
  target_modules: [linear_q, linear_v]

msa:
  backend: single                  # single / precomputed / colabfold / local
  msa_dir: null
  max_msa_depth: 512
  colabfold_server: "https://api.colabfold.com"
  tool: mmseqs2                    # for local backend
  database_paths: []               # e.g. [/data/oas/oas_db, /data/uniref30/uniref30]
  n_cpu: 4

output:
  checkpoint_dir: ./checkpoints
```

## Testing

```bash
make test              # All tests with coverage
make test-unit         # Unit tests only
make test-integration  # API integration tests
make lint              # Ruff linting
make typecheck         # MyPy type checking
```

## Docker

```bash
docker compose up --build
# API at http://localhost:8000
```
