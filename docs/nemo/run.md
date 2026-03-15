# NeMo Run

> Experiment configuration, execution, and management.

## Canonical URLs

| Resource | URL |
|---|---|
| GitHub | `https://github.com/NVIDIA/NeMo-Run` |
| Docs (in Framework Guide) | `https://docs.nvidia.com/nemo-framework/user-guide/latest/index.html` |

## Overview

NeMo Run simplifies the lifecycle of NeMo experiments — from configuration through execution to result tracking. It provides a unified interface for running experiments locally, on clusters, or in the cloud.

## Key Features

- **Declarative Configuration**: Define experiments as composable config objects
- **Multi-Backend Execution**: Run on local machines, SLURM clusters, or cloud
- **Experiment Tracking**: Automatic logging of configs, metrics, and artifacts
- **Reproducibility**: Full experiment state captured for reproduction
- **CLI and Python API**: Use from command line or programmatically

## Execution Backends

| Backend | Description |
|---|---|
| Local | Run on the current machine |
| SLURM | Submit to HPC cluster via SLURM scheduler |
| Cloud | Run on cloud instances (AWS, GCP, Azure) |

## Installation

```bash
pip install nemo-run
```

## Usage Pattern

```python
import nemo_run as run

# Define experiment
experiment = run.Experiment(
    name="my-training-run",
    executor=run.LocalExecutor(),
)

# Configure and launch
experiment.add(
    run.Script("train.py"),
    config=run.Config(model="llama-3", batch_size=32),
)
experiment.run()
```
