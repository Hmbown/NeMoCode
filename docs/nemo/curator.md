# NeMo Curator

> GPU-accelerated data preprocessing for text, image, video, and audio datasets.

## Canonical URLs

| Resource | URL |
|---|---|
| GitHub | `https://github.com/NVIDIA/NeMo-Curator` |
| Docs (in Framework Guide) | `https://docs.nvidia.com/nemo-framework/user-guide/latest/index.html` |

## Overview

NeMo Curator provides scalable, GPU-accelerated data curation pipelines for preparing training data. It handles deduplication, filtering, quality scoring, and domain classification across multiple modalities.

## Key Features

- **Text Curation**: Deduplication (exact, fuzzy, semantic), quality filtering, language ID, PII removal
- **Image Curation**: NSFW filtering, aesthetic scoring, deduplication
- **Video Curation**: Frame extraction, quality assessment, deduplication
- **Audio Curation**: Quality filtering, transcript alignment
- **Scalability**: Distributed processing via Dask and RAPIDS on GPU clusters
- **Modular Pipeline**: Composable stages that can be mixed and matched

## Pipeline Stages (Text)

1. **Document Extraction** — Parse Common Crawl, WARC, PDF, etc.
2. **Language Identification** — fastText-based language detection
3. **Unicode Reformatting** — Normalize text encoding
4. **Heuristic Filtering** — Rule-based quality filters (length, repetition, etc.)
5. **Classifier Filtering** — ML-based quality scoring
6. **Exact Deduplication** — Hash-based exact match removal
7. **Fuzzy Deduplication** — MinHash + LSH near-duplicate detection
8. **Semantic Deduplication** — Embedding-based semantic duplicate removal
9. **PII Removal** — Detect and redact personally identifiable information
10. **Domain Classification** — Categorize documents by topic/domain

## Installation

```bash
pip install nemo-curator
# or with GPU support:
pip install nemo-curator[cuda12x]
```

## Usage Pattern

```python
from nemo_curator import Pipeline
from nemo_curator.modules import ExactDeduplication, QualityFilter

pipeline = Pipeline([
    QualityFilter(score_threshold=0.7),
    ExactDeduplication(),
])
result = pipeline(dataset)
```
