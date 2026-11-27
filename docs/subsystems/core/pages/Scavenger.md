# Scavenger

## Overview

Scavenger is HyperNix's intelligent dataset discovery engine. It searches the HuggingFace Hub for datasets matching your training criteria, filtering by storage constraints, quality signals, content relevance, and recency.

## Features

- **Keyword search**: Find datasets by content keywords
- **Storage budgeting**: Per-dataset and combined storage limits
- **Quality filtering**: Minimum likes, downloads, entries
- **Recency filtering**: Maximum age since last update
- **Token budgeting**: Estimate total token count across datasets
- **Relevance scoring**: Automatic ranking by quality signals

## Usage

### Python API

```python
from hypernix.scavenger import Scavenger, ScavengerCriteria

sc = Scavenger()

criteria = ScavengerCriteria(
    keywords=["code", "python", "instruction"],
    max_storage_per_dataset_gb=10.0,
    max_combined_storage_gb=50.0,
    min_entries=1000,
    data_types=["jsonl", "parquet"],
    min_likes=10,
    max_age_days=365,
)

datasets = sc.hunt(criteria)
sc.display_results(datasets)

# Download a dataset
ds = sc.download_dataset("codeparrot/github-code", streaming=True)
```

### CLI

```bash
# Search for datasets
hnx scavenger --keywords code,python --max-storage 10 --min-likes 50

# Search with age limit
hnx scavenger --keywords medical --max-age 180 --min-entries 5000

# Download a specific dataset
hnx scavenger --download codeparrot/github-code
```

## Search Criteria

| Parameter | Description | Example |
|-----------|-------------|---------|
| keywords | Content keywords | ["code", "python"] |
| max_storage_per_dataset_gb | Max size per dataset | 10.0 |
| max_combined_storage_gb | Max total size | 50.0 |
| min_entries | Minimum rows | 1000 |
| max_entries | Maximum rows | 1000000 |
| min_likes | Minimum Hub likes | 10 |
| min_downloads | Monthly downloads | 100 |
| max_age_days | Max age in days | 365 |
| data_types | Preferred formats | ["jsonl", "parquet"] |
| tokens_per_entry | Estimated tokens/row | 200 |
| max_total_tokens | Max total tokens | 1e9 |

## Relevance Scoring

Datasets are scored on:
- Downloads (up to 20 points)
- Likes (up to 15 points)
- Recency (up to 10 points)
- Size appropriateness (up to 10 points)
- Keyword match density (5 points per keyword)
