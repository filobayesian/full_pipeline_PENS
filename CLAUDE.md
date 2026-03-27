# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Stress-testing framework that evaluates how two complementary news personalization models degrade under data scarcity. The two models under test are a **Reranker** (neural news ranking with BPR/listwise loss) and a **Style Profiler** (lexicon-based headline feature extraction and user profiling). Both operate on the Microsoft PENS dataset.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# One-time artifact setup (parses news, generates embeddings, builds profiles)
./setup_experiments.sh

# Run stress tests
python -m stress_test run --pens_root data/PENS --models profiler
python -m stress_test run --pens_root data/PENS --models reranker \
    --train_impressions "refactor_PENS copy/artifacts/train_impressions.jsonl" \
    --valid_impressions "refactor_PENS copy/artifacts/valid_impressions.jsonl" \
    --embeddings "refactor_PENS copy/artifacts/news_embeddings.pt"

# Dry run (print experiment plan without executing)
python -m stress_test run --pens_root data/PENS --dry_run

# Analyze results
python -m stress_test analyze --run_dir stress_test_results/run_TIMESTAMP/

# Show experiment grid info
python -m stress_test info

# Validate framework (quick smoke test)
python -m stress_test.validate
python -m stress_test.validate --pens_root data/PENS --quick_test

# Tests
pytest "style_steering_PENS copy/tests/" -v
pytest "refactor_PENS copy/tests/" -v
pytest "style_steering_PENS copy/tests/test_metrics.py::TestFeatureExtraction::test_quote_features" -v
```

## Architecture

Three modules with the stress test framework orchestrating the other two:

```
stress_test/               Orchestrator: CLI, experiment grid, data sampling, adapters
  ├── runner.py            Runs experiment grid (user_count x history_length x seed x model)
  ├── config.py            Experiment grid defaults and ExperimentConfig dataclass
  ├── data_utils.py        User-level sampling and history truncation
  ├── reranker_adapter.py  Wraps reranker training/eval for stress test use
  ├── profiler_adapter.py  Wraps profiler pipeline for stress test use
  └── analysis.py          Results aggregation and sensitivity plots

refactor_PENS copy/       Reranker: neural news ranking
  └── src/
      ├── model/reranker.py   NewsReranker (history → mean pool → MLP → score)
      ├── data/dataset.py     ImpressionDataset (BPR pairs, listwise softmax)
      ├── data/parse_*.py     TSV → parquet/JSONL parsers
      ├── embeddings/         Jina v2 768-dim news embeddings
      ├── train.py, eval.py   Training and evaluation entry points
      └── utils/metrics.py    AUC, MRR, nDCG@K, Hit@K

style_steering_PENS copy/ Style Profiler: lexicon-based feature extraction
  └── headline_style/
      ├── config_v2.py        Lexicons, patterns, 22 features across 8 families
      ├── metrics_v2.py       Feature extraction (no LLM, uses regex/lexicon/VADER)
      ├── profiling_v2.py     Contrastive profiling, z-scores, alignment scoring
      ├── pens_adapter.py     PENS dataset loading
      └── run_profile_v2.py   CLI runner
```

## Key Design Decisions

- **Data sampling is user-level, not impression-level** — preserves each user's history structure when reducing dataset size. History truncation is applied after user sampling.
- **Reranker requires pre-computed artifacts** — embeddings (768-dim Jina), parsed impressions (JSONL). Run `setup_experiments.sh` first.
- **Profiler runs without GPU** — lexicon/regex-based extraction with optional VADER sentiment and RoBERTa formality.
- **Contrastive profiling** — compares top 25% vs bottom 25% dwell-time interactions; falls back to weighted mean for users with <8 interactions.
- **11 of 22 features** used for alignment scoring (defined in `ALIGNMENT_FEATURES` in `config_v2.py`).
- **Experiment grid** defaults: user_counts=[1000, 5000, 10000, 50000], history_lengths=[5, 10, 20, 30, 50], 1 seed.

## Data Layout

- PENS dataset: `data/PENS/` (news.tsv, train.tsv, valid.tsv) — tab-delimited, space-separated within list fields
- Brysbaert lexicon: `data/lexicons/brysbaert_concreteness.csv`
- Reranker artifacts: `refactor_PENS copy/artifacts/` (news.parquet, news_embeddings.pt, train/valid_impressions.jsonl)
- Profiler outputs: `style_steering_PENS copy/outputs/` (user_profiles.parquet, headline_features.parquet)
- Stress test results: `stress_test_results/run_TIMESTAMP/` (JSON metrics per experiment)

## PENS Data Format

- **IDs**: News = `N{number}`, User = `U{number}`
- **Delimiter**: TAB between columns, SPACE within list fields (history, pos, neg)
- **Timestamps**: `M/D/YYYY H:MM:SS AM/PM`