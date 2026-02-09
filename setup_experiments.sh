#!/bin/bash
# Setup script for PENS stress testing experiments
# Run this to prepare all necessary artifacts before running experiments

set -e  # Exit on error

echo "============================================================"
echo "PENS EXPERIMENT SETUP"
echo "============================================================"

# Configuration
PENS_ROOT="data/PENS"
RERANKER_DIR="refactor_PENS copy"
PROFILER_DIR="style_steering_PENS copy"
ARTIFACTS_DIR="${RERANKER_DIR}/artifacts"

# Check PENS data exists
if [ ! -f "${PENS_ROOT}/news.tsv" ] || [ ! -f "${PENS_ROOT}/train.tsv" ]; then
    echo "ERROR: PENS data not found at ${PENS_ROOT}/"
    echo "Expected files: news.tsv, train.tsv, valid.tsv"
    exit 1
fi

echo "PENS data found at ${PENS_ROOT}/"

# Create artifacts directory
mkdir -p "${ARTIFACTS_DIR}"
mkdir -p "${PROFILER_DIR}/outputs"

# ============================================================
# Step 1: Prepare Reranker Artifacts
# ============================================================
echo ""
echo "============================================================"
echo "STEP 1: Preparing Reranker Artifacts"
echo "============================================================"

# 1a. Parse news to parquet
if [ ! -f "${ARTIFACTS_DIR}/news.parquet" ]; then
    echo "Parsing news.tsv to parquet..."
    cd "${RERANKER_DIR}"
    python -m src.data.parse_news \
        --news_tsv "../${PENS_ROOT}/news.tsv" \
        --out "artifacts/news.parquet"
    cd ..
else
    echo "news.parquet already exists, skipping..."
fi

# 1b. Generate embeddings (this takes 10-15 min with GPU, 30-40 min CPU)
if [ ! -f "${ARTIFACTS_DIR}/news_embeddings.pt" ]; then
    echo "Generating news embeddings (this may take 10-40 minutes)..."
    cd "${RERANKER_DIR}"
    python -m src.embeddings.encode_news \
        --news_table "artifacts/news.parquet" \
        --out "artifacts/news_embeddings.pt" \
        --model "jinaai/jina-embeddings-v2-base-en" \
        --batch_size 64 \
        --normalize
    cd ..
else
    echo "news_embeddings.pt already exists, skipping..."
fi

# 1c. Parse train impressions
if [ ! -f "${ARTIFACTS_DIR}/train_impressions.jsonl" ]; then
    echo "Parsing train impressions..."
    cd "${RERANKER_DIR}"
    python -m src.data.parse_impressions \
        --tsv "../${PENS_ROOT}/train.tsv" \
        --out "artifacts/train_impressions.jsonl"
    cd ..
else
    echo "train_impressions.jsonl already exists, skipping..."
fi

# 1d. Parse valid impressions
if [ ! -f "${ARTIFACTS_DIR}/valid_impressions.jsonl" ]; then
    echo "Parsing valid impressions..."
    cd "${RERANKER_DIR}"
    python -m src.data.parse_impressions \
        --tsv "../${PENS_ROOT}/valid.tsv" \
        --out "artifacts/valid_impressions.jsonl"
    cd ..
else
    echo "valid_impressions.jsonl already exists, skipping..."
fi

echo "Reranker artifacts ready!"

# ============================================================
# Step 2: Prepare Profiler Artifacts
# ============================================================
echo ""
echo "============================================================"
echo "STEP 2: Preparing Profiler Artifacts"
echo "============================================================"

# Check if profiler outputs exist
if [ ! -f "${PROFILER_DIR}/outputs/user_profiles.parquet" ]; then
    echo "Running profiling pipeline (this may take 5-15 minutes)..."
    cd "${PROFILER_DIR}"
    python -m headline_style.run_profile_v2 \
        --pens_root "../${PENS_ROOT}" \
        --out "outputs/" \
        --split train \
        --use_contrastive
    cd ..
else
    echo "Profiler outputs already exist, skipping..."
fi

echo "Profiler artifacts ready!"

# ============================================================
# Summary
# ============================================================
echo ""
echo "============================================================"
echo "SETUP COMPLETE"
echo "============================================================"
echo ""
echo "Reranker artifacts (${ARTIFACTS_DIR}/):"
ls -la "${ARTIFACTS_DIR}/" 2>/dev/null || echo "  (directory empty or not found)"
echo ""
echo "Profiler outputs (${PROFILER_DIR}/outputs/):"
ls -la "${PROFILER_DIR}/outputs/" 2>/dev/null || echo "  (directory empty or not found)"
echo ""
echo "You can now run stress tests:"
echo ""
echo "  # Profiler only (faster, no GPU required):"
echo "  python -m stress_test run --pens_root data/PENS --models profiler"
echo ""
echo "  # Reranker only (requires GPU for faster training):"
echo "  python -m stress_test run --pens_root data/PENS \\"
echo "      --train_impressions '${ARTIFACTS_DIR}/train_impressions.jsonl' \\"
echo "      --valid_impressions '${ARTIFACTS_DIR}/valid_impressions.jsonl' \\"
echo "      --embeddings '${ARTIFACTS_DIR}/news_embeddings.pt' \\"
echo "      --models reranker"
echo ""
echo "  # Both models:"
echo "  python -m stress_test run --pens_root data/PENS \\"
echo "      --train_impressions '${ARTIFACTS_DIR}/train_impressions.jsonl' \\"
echo "      --valid_impressions '${ARTIFACTS_DIR}/valid_impressions.jsonl' \\"
echo "      --embeddings '${ARTIFACTS_DIR}/news_embeddings.pt'"
