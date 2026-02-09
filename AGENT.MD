# AGENT.MD

This repository merges two complementary PENS models plus a shared stress-testing
framework. The two models are independent and focus on different tasks:

- `refactor_PENS copy/`: content selection via a neural reranker.
- `style_steering_PENS copy/`: headline style profiling and alignment scoring.
- `stress_test/`: a unified experiment runner that evaluates both models across
  user-fraction and history-length regimes.

The core dataset is the Microsoft PENS dataset located at `data/PENS/`.

---

## Repository Layout

Top-level:

- `refactor_PENS copy/`: neural news reranker (content selector).
- `style_steering_PENS copy/`: headline style profiler (style preferences).
- `stress_test/`: experiment grid runner and adapters for both models.
- `setup_experiments.sh`: prepares artifacts for both models.
- `data/`: local datasets and lexicons; `data/PENS/` is expected input.
- `stress_test_results/`: experiment outputs (JSON, summaries, plots).

---

## Model 1: Reranker (Content Selector)

Location: `refactor_PENS copy/`

Goal: learn user preferences from click history and rank candidate news
articles.

Key entrypoints:

- `src/data/parse_news.py`: parse `news.tsv` to `artifacts/news.parquet`.
- `src/embeddings/encode_news.py`: compute Jina embeddings into
  `artifacts/news_embeddings.pt`.
- `src/data/parse_impressions.py`: parse `train.tsv`/`valid.tsv` to
  `artifacts/*_impressions.jsonl`.
- `src/train.py`: training pipeline (BPR or listwise loss).
- `src/eval.py`: evaluation (AUC, MRR, nDCG@5/10, Hit@5/10).
- `src/continual.py`: time-windowed continual learning.

Core model:

- `src/model/reranker.py`: `NewsReranker` (user encoder + MLP scorer).
- `src/model/adaptor.py`: optional trainable embedding adaptor.

Model flow (high-level):

1. Encode news into frozen embeddings.
2. Build a user vector by masked mean pooling of click-history embeddings.
3. Score candidate items using an MLP on interaction features:
   `[user, candidate, user * candidate, cosine(user, candidate)]`.
4. Train with BPR (pairwise) or listwise softmax (impression-level).

Artifacts:

- `refactor_PENS copy/artifacts/news.parquet`
- `refactor_PENS copy/artifacts/news_embeddings.pt`
- `refactor_PENS copy/artifacts/train_impressions.jsonl`
- `refactor_PENS copy/artifacts/valid_impressions.jsonl`
- `refactor_PENS copy/artifacts/checkpoints/*`

Primary documentation:

- `refactor_PENS copy/README.md`
- `refactor_PENS copy/CLAUDE.md`

---

## Model 2: Style Profiler (Title Rewriter Support)

Location: `style_steering_PENS copy/`

Goal: extract interpretable linguistic features from headlines, build per-user
style profiles, and compute alignment scores that can guide rewriting.

Key entrypoints:

- `headline_style/run_profile_v2.py`: main CLI for profiling.
- `headline_style/metrics_v2.py`: feature extraction (22 features, 8 families).
- `headline_style/profiling_v2.py`: user profile building and alignment scoring.
- `headline_style/pens_adapter.py`: loads and normalizes PENS data.
- `headline_style/analysis.py`: reports and visualization outputs.

Pipeline (high-level):

1. Load PENS interactions with `pens_adapter`.
2. Extract features from each headline (quotes, sentiment, concreteness,
   entities, formality, etc.).
3. Build per-user profiles with dwell-time weighting and contrastive profiling.
4. Compute alignment scores between user profiles and headline features.
5. Generate summary reports and visualizations.

Outputs (default `outputs/`):

- `headline_features.parquet`
- `user_profiles.parquet`
- `style_alignment_examples.parquet`
- `population_statistics.parquet`
- `summary.md` and plots

Primary documentation:

- `style_steering_PENS copy/README.md`
- `style_steering_PENS copy/QUICKSTART.md`
- `style_steering_PENS copy/docs/METHODOLOGY.md`

---

## Stress Testing Framework

Location: `stress_test/`

Goal: evaluate both models under controlled data scarcity by varying:

- fraction of users (default: 0.1, 0.25, 0.5, 0.75, 1.0)
- history length per user (default: 5, 10, 20, 30, 50)
- random seeds (default: 3)

Key files:

- `stress_test/__main__.py`: CLI entry point (`run`, `analyze`, `info`).
- `stress_test/config.py`: experiment grid + per-model configs.
- `stress_test/runner.py`: grid orchestration and result saving.
- `stress_test/reranker_adapter.py`: wraps reranker train + eval.
- `stress_test/profiler_adapter.py`: wraps profiler pipeline and metrics.
- `stress_test/data_utils.py`: shared loading, user sampling, truncation.

How it works:

1. Build an experiment grid in `config.py`.
2. For each config:
   - sample users (keeps all impressions for sampled users),
   - truncate history to `history_length`,
   - run either reranker or profiler pipeline through adapters.
3. Save per-experiment JSON and a summary in `stress_test_results/`.

---

## Data Expectations

PENS data is expected at `data/PENS/`:

- `news.tsv`
- `train.tsv`
- `valid.tsv`
- (optional) `personalized_test.tsv`

Lexicon files used by style profiling live under `data/lexicons/`.

---

## Common Workflows

Prepare artifacts for both models:

- `./setup_experiments.sh`

Run stress tests:

- Profiler-only (fast):
  `python -m stress_test run --pens_root data/PENS --models profiler`
- Reranker-only (needs embeddings + impressions):
  `python -m stress_test run --pens_root data/PENS \
   --train_impressions refactor_PENS copy/artifacts/train_impressions.jsonl \
   --valid_impressions refactor_PENS copy/artifacts/valid_impressions.jsonl \
   --embeddings refactor_PENS copy/artifacts/news_embeddings.pt \
   --models reranker`
- Both models:
  `python -m stress_test run --pens_root data/PENS \
   --train_impressions refactor_PENS copy/artifacts/train_impressions.jsonl \
   --valid_impressions refactor_PENS copy/artifacts/valid_impressions.jsonl \
   --embeddings refactor_PENS copy/artifacts/news_embeddings.pt`

---

## Notes for Future Agents

- The two models do not call each other directly; `stress_test/` is the common
  layer that compares their behavior under varying data availability.
- The reranker expects precomputed embeddings; missing this file is the most
  common failure.
- The profiler is CPU-friendly and has optional dependencies (spaCy, RoBERTa).
- Outputs are large and are intentionally gitignored; do not commit artifacts.

