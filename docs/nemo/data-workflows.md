# Repo-Aware NVIDIA Data Workflows

> Use NeMoCode to analyze a repository, scaffold seed artifacts, and feed NVIDIA data services.

## Why This Exists

NeMoCode already understands repository structure, code, tests, and documentation. NVIDIA now has a credible data stack around that:

- **NeMo Data Designer** for synthetic data generation from schemas, prompts, and seed data
- **NeMo Evaluator** for scoring generated tasks and model behavior
- **NeMo Safe Synthesizer** for privacy-preserving tabular data workflows
- **NeMo Curator** for large-scale curation and filtering

That makes a strong repo-to-data flywheel possible:

1. Analyze a repo
2. Build grounded seed artifacts
3. Generate repo-specific tasks and examples
4. Evaluate quality
5. Promote the best data into training, eval, or regression suites

## Commands

| Command | Description |
|---|---|
| `nemo data analyze` | Profile a repo and recommend NVIDIA services |
| `nemo data export-seeds` | Export repo-aware seed artifacts for Data Designer |
| `nemo data preview` | Preview synthetic data via a local Data Designer |
| `nemo data export-sft` | Export JSONL dataset for SFT / instruction tuning |
| `nemo data job create` | Create an async Data Designer generation job |
| `nemo data job status <id>` | Check job status |
| `nemo data job logs <id>` | Fetch job logs |
| `nemo data job results <id>` | List or download job results |
| `nemo setup data` | Check prerequisites and show setup instructions |

## End-to-End Usage

### 1. Local Setup

Install NeMoCode:

```bash
pip install -e ".[dev]"
```

Set up NVIDIA data services (Docker required):

```bash
nemo setup data
```

This checks your environment and prints the official Docker Compose setup paths for Data Designer, Evaluator, and Safe Synthesizer.

### 2. Repo Analysis

```bash
nemo data analyze
nemo data analyze --output .nemocode/data/repo-data-plan.yaml
nemo data analyze --format json --show-plan
```

This profiles the repo, detects languages/frameworks/tests/docs, recommends which NVIDIA services fit, and generates a starter Data Designer config.

### 3. Seed Export

```bash
nemo data export-seeds
nemo data export-seeds --output-dir ./my-seeds
```

Scans source, tests, and docs to produce:

- `repo_profile.yaml` — machine-readable repo summary
- `file_manifest.jsonl` — paths and metadata for candidate files
- `task_taxonomy.yaml` — repo-specific task families and difficulty levels
- `context_packs.jsonl` — sampled file snippets for prompt grounding

Automatically excludes secrets (`.env`, keys), lockfiles, binary files, vendor folders, and caches.

### 4. Data Designer Preview

Start a local Data Designer instance (see `nemo setup data` for Docker instructions), then:

```bash
nemo data preview
nemo data preview --num-records 10
nemo data preview --base-url http://localhost:9090
nemo data preview --output preview-results.jsonl
```

The preview sends repo-derived seed context to `POST /v1/data-designer/preview` and prints the generated records.

**Environment variable override:** `NEMOCODE_DATA_BASE_URL`

### 5. Data Designer Jobs (Async)

For larger batches, use async jobs:

```bash
# Create a job
nemo data job create --name "my-batch" --num-records 500

# Check status
nemo data job status <job-id>

# View logs
nemo data job logs <job-id>

# List or download results
nemo data job results <job-id>
nemo data job results <job-id> --download
nemo data job results <job-id> --download --output ./output.jsonl
```

### 6. Dataset Export for Fine-Tuning

```bash
nemo data export-sft
nemo data export-sft --max-records 100
nemo data export-sft --output custom-dataset.jsonl
```

Exports JSONL suitable for SFT / instruction tuning. Each record contains:

- **system**: grounding context (repo summary + optional file snippet)
- **user**: a realistic repo-specific request
- **assistant**: acceptance criteria and approach
- **metadata**: repo area, language, task type, difficulty

If seeds have been exported first (`nemo data export-seeds`), file snippets from `context_packs.jsonl` are woven into the prompts for grounding.

This is for **repo-specific assistant tuning**, not raw code dumping.

### 7. Optional: Safe Synthesizer (Private Tabular Data)

Use NeMo Safe Synthesizer when private CSV, CRM, ticketing, support, or tabular product data must be anonymized before mixing with repo-derived data.

```bash
# Set the base URL
export NEMOCODE_SAFE_SYNTH_BASE_URL=http://localhost:8080

# The Safe Synthesizer client is available via Python:
from nemocode.core.nvidia_client import SafeSynthesizerClient
client = SafeSynthesizerClient()
```

API endpoints follow `/v1beta1/safe-synthesizer/jobs/...`.

### 8. Optional: Evaluator (Quality Scoring)

Use NeMo Evaluator to score generated tasks before promoting them into training or benchmark sets.

```bash
# Set the base URL
export NEMOCODE_EVAL_BASE_URL=http://localhost:8080

# The Evaluator client is available via Python:
from nemocode.core.nvidia_client import EvaluatorClient
client = EvaluatorClient()
jobs = client.list_jobs()
```

API endpoints follow `/v2/evaluation/jobs/...`.

## Environment Variables

| Variable | Default | Purpose |
|---|---|---|
| `NEMOCODE_DATA_BASE_URL` | `http://localhost:8080` | Data Designer service URL |
| `NEMOCODE_EVAL_BASE_URL` | `http://localhost:8080` | Evaluator service URL |
| `NEMOCODE_SAFE_SYNTH_BASE_URL` | `http://localhost:8080` | Safe Synthesizer service URL |
| `NIM_API_KEY` | — | API key for NVIDIA API endpoints |
| `NGC_CLI_API_KEY` | — | NGC API key (fallback) |

## Recommended Build Order

1. **Data Designer first**
   - best path for repo-grounded synthetic coding tasks
   - lowest-friction way to preview and iterate on task schemas
2. **Evaluator second**
   - score the generated tasks before they become training or benchmark data
3. **Safe Synthesizer only when needed**
   - use when private tabular product, support, CRM, or telemetry data must be protected
4. **Curator later**
   - use when the corpus extends beyond repo-local assets into larger document or data collections

## Official References

- Data Designer: `https://docs.nvidia.com/nemo/microservices/latest/design-synthetic-data-from-scratch-or-seeds/docker-compose.html`
- Evaluator: `https://docs.nvidia.com/nemo/microservices/latest/evaluate/docker-compose.html`
- Safe Synthesizer: `https://docs.nvidia.com/nemo/microservices/latest/generate-private-synthetic-data/docker-compose.html`
- Curator: `https://docs.nvidia.com/nemo/curator/latest/index.html`
