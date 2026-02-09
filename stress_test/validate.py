#!/usr/bin/env python3
"""
Validation Script for Stress Testing Framework

This script validates that the stress testing framework is correctly installed
and can be used when PENS data is available.

Usage:
    # Validate imports and structure
    python -m stress_test.validate

    # Validate with actual data (if available)
    python -m stress_test.validate --pens_root data/PENS --quick_test
"""

import argparse
import sys
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def validate_imports():
    """Validate that all modules can be imported."""
    logger.info("Validating imports...")
    
    try:
        from stress_test import (
            EXPERIMENT_GRID,
            RERANKER_CONFIG,
            PROFILER_CONFIG,
            ExperimentConfig,
            generate_experiment_grid,
            count_experiments,
        )
        logger.info("  Config imports: OK")
    except ImportError as e:
        logger.error(f"  Config imports: FAILED - {e}")
        return False
    
    try:
        from stress_test import (
            load_reranker_data,
            load_profiler_data,
            sample_users_reranker,
            sample_users_profiler,
            truncate_history_reranker,
            truncate_history_profiler,
        )
        logger.info("  Data utils imports: OK")
    except ImportError as e:
        logger.error(f"  Data utils imports: FAILED - {e}")
        return False
    
    try:
        from stress_test.reranker_adapter import (
            run_reranker_experiment,
            load_embeddings_cache,
        )
        logger.info("  Reranker adapter imports: OK")
    except ImportError as e:
        logger.error(f"  Reranker adapter imports: FAILED - {e}")
        return False
    
    try:
        from stress_test.profiler_adapter import (
            run_profiler_experiment,
            compute_profile_metrics,
        )
        logger.info("  Profiler adapter imports: OK")
    except ImportError as e:
        logger.error(f"  Profiler adapter imports: FAILED - {e}")
        return False
    
    try:
        from stress_test.runner import (
            run_experiment_grid,
            run_single_experiment,
        )
        logger.info("  Runner imports: OK")
    except ImportError as e:
        logger.error(f"  Runner imports: FAILED - {e}")
        return False
    
    try:
        from stress_test.analysis import (
            load_experiment_results,
            results_to_dataframe,
            aggregate_results,
            generate_report,
        )
        logger.info("  Analysis imports: OK")
    except ImportError as e:
        logger.error(f"  Analysis imports: FAILED - {e}")
        return False
    
    return True


def validate_config():
    """Validate configuration structure."""
    logger.info("Validating configuration...")
    
    from stress_test.config import (
        EXPERIMENT_GRID,
        RERANKER_CONFIG,
        PROFILER_CONFIG,
        generate_experiment_grid,
        count_experiments,
    )
    
    # Check experiment grid
    assert 'user_fractions' in EXPERIMENT_GRID
    assert 'history_lengths' in EXPERIMENT_GRID
    assert 'n_seeds' in EXPERIMENT_GRID
    logger.info(f"  Experiment grid: {count_experiments()} total experiments")
    
    # Check reranker config
    assert 'epochs' in RERANKER_CONFIG
    assert 'loss_type' in RERANKER_CONFIG
    logger.info(f"  Reranker config: {len(RERANKER_CONFIG)} parameters")
    
    # Check profiler config
    assert 'use_contrastive' in PROFILER_CONFIG
    logger.info(f"  Profiler config: {len(PROFILER_CONFIG)} parameters")
    
    # Test experiment generation
    experiments = generate_experiment_grid(
        models=['profiler'],
        user_fractions=[0.5, 1.0],
        history_lengths=[10],
        n_seeds=1,
    )
    assert len(experiments) == 2
    logger.info(f"  Experiment generation: OK")
    
    return True


def validate_data_utils():
    """Validate data utility functions."""
    logger.info("Validating data utilities...")
    
    from stress_test.data_utils import (
        sample_users_reranker,
        truncate_history_reranker,
        get_reranker_stats,
    )
    
    # Create mock impressions
    mock_impressions = [
        {'user_id': 'U1', 'history': ['N1', 'N2', 'N3'], 'pos': ['N4'], 'neg': ['N5', 'N6']},
        {'user_id': 'U1', 'history': ['N1', 'N2', 'N3', 'N4'], 'pos': ['N7'], 'neg': ['N8']},
        {'user_id': 'U2', 'history': ['N10', 'N11'], 'pos': ['N12'], 'neg': ['N13']},
        {'user_id': 'U3', 'history': ['N20'], 'pos': ['N21'], 'neg': ['N22']},
    ]
    
    # Test sampling
    sampled = sample_users_reranker(mock_impressions, fraction=0.5, seed=42)
    assert len(sampled) < len(mock_impressions)
    logger.info(f"  User sampling: {len(mock_impressions)} -> {len(sampled)} impressions")
    
    # Test truncation
    truncated = truncate_history_reranker(mock_impressions, max_len=2)
    for imp in truncated:
        assert len(imp['history']) <= 2
    logger.info(f"  History truncation: OK")
    
    # Test stats
    stats = get_reranker_stats(mock_impressions)
    assert 'n_impressions' in stats
    assert 'n_users' in stats
    logger.info(f"  Stats computation: {stats['n_users']} users, {stats['n_impressions']} impressions")
    
    return True


def validate_with_data(pens_root: str):
    """Validate with actual PENS data."""
    logger.info(f"Validating with PENS data at {pens_root}...")
    
    pens_path = Path(pens_root)
    
    # Check required files
    required_files = ['news.tsv', 'train.tsv']
    for f in required_files:
        if not (pens_path / f).exists():
            logger.error(f"  Missing required file: {f}")
            return False
    logger.info("  Required files found")
    
    # Try loading profiler data
    from stress_test.data_utils import load_profiler_data, get_profiler_stats
    
    df = load_profiler_data(pens_root, split='train')
    stats = get_profiler_stats(df)
    logger.info(f"  Loaded {stats['n_users']} users, {stats['n_interactions']} interactions")
    
    return True


def run_quick_test(pens_root: str):
    """Run a quick test with minimal configuration."""
    logger.info("Running quick profiler test...")
    
    from stress_test.data_utils import (
        load_profiler_data,
        sample_users_profiler,
        truncate_history_profiler,
    )
    from stress_test.profiler_adapter import run_profiler_experiment
    from stress_test.config import PROFILER_CONFIG
    
    # Load and subsample data
    df = load_profiler_data(pens_root, split='train')
    df = sample_users_profiler(df, fraction=0.05, seed=42)  # 5% of users
    df = truncate_history_profiler(df, max_len=20)
    
    logger.info(f"  Test data: {df['user_id'].nunique()} users, {len(df)} interactions")
    
    # Run profiler
    result = run_profiler_experiment(df, PROFILER_CONFIG)
    
    if result['success']:
        metrics = result['metrics']
        logger.info(f"  Profiler test: OK")
        logger.info(f"    - Mean |z|: {metrics.get('mean_abs_z', 0):.4f}")
        logger.info(f"    - Coverage: {metrics.get('coverage', 0):.4f}")
        logger.info(f"    - Strong prefs: {metrics.get('n_strong_prefs', 0):.2f}")
        return True
    else:
        logger.error(f"  Profiler test: FAILED - {result['error']}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Validate stress testing framework'
    )
    
    parser.add_argument(
        '--pens_root',
        type=str,
        default=None,
        help='Path to PENS data for data validation'
    )
    
    parser.add_argument(
        '--quick_test',
        action='store_true',
        help='Run quick profiler test with real data'
    )
    
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print("STRESS TESTING FRAMEWORK VALIDATION")
    print("=" * 60 + "\n")
    
    all_passed = True
    
    # 1. Validate imports
    if not validate_imports():
        all_passed = False
    print()
    
    # 2. Validate config
    if not validate_config():
        all_passed = False
    print()
    
    # 3. Validate data utilities
    if not validate_data_utils():
        all_passed = False
    print()
    
    # 4. Validate with real data (if provided)
    if args.pens_root:
        if not validate_with_data(args.pens_root):
            all_passed = False
        print()
        
        if args.quick_test:
            if not run_quick_test(args.pens_root):
                all_passed = False
            print()
    
    # Summary
    print("=" * 60)
    if all_passed:
        print("VALIDATION: ALL TESTS PASSED")
    else:
        print("VALIDATION: SOME TESTS FAILED")
    print("=" * 60 + "\n")
    
    return 0 if all_passed else 1


if __name__ == '__main__':
    sys.exit(main())
