"""
Stress Testing Framework for PENS Models

This package provides tools to systematically evaluate how the content selector
(reranker) and style profiler models degrade as a function of:
1. Number of users in training data
2. Length of user history

Key components:
- config: Experiment grid configuration
- data_utils: Data loading, user sampling, history truncation
- reranker_adapter: Wrapper for reranker training/evaluation
- profiler_adapter: Wrapper for style profiler
- runner: Experiment orchestration
- analysis: Results aggregation and visualization

Usage:
    python -m stress_test run --pens_root data/PENS --models profiler
    python -m stress_test analyze --run_dir stress_test_results/run_xxx/
    python -m stress_test info
"""

from .config import (
    EXPERIMENT_GRID,
    RERANKER_CONFIG,
    PROFILER_CONFIG,
    OUTPUT_CONFIG,
    ExperimentConfig,
    generate_experiment_grid,
    count_experiments,
)
from .data_utils import (
    load_reranker_data,
    load_profiler_data,
    sample_users_reranker,
    sample_users_profiler,
    truncate_history_reranker,
    truncate_history_profiler,
    get_reranker_stats,
    get_profiler_stats,
    prepare_experiment_data,
)

__all__ = [
    # Config
    'EXPERIMENT_GRID',
    'RERANKER_CONFIG',
    'PROFILER_CONFIG',
    'OUTPUT_CONFIG',
    'ExperimentConfig',
    'generate_experiment_grid',
    'count_experiments',
    # Data utilities
    'load_reranker_data',
    'load_profiler_data',
    'sample_users_reranker',
    'sample_users_profiler',
    'truncate_history_reranker',
    'truncate_history_profiler',
    'get_reranker_stats',
    'get_profiler_stats',
    'prepare_experiment_data',
]
