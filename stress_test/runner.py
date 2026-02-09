"""
Experiment Runner for Stress Testing

This module orchestrates the execution of the stress testing experiment grid,
running both reranker and profiler experiments across different user fractions
and history lengths.

Key functions:
- run_experiment: Run a single experiment
- run_experiment_grid: Run all experiments in grid
- main: CLI entry point
"""

import argparse
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
import torch

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
)
from .reranker_adapter import (
    run_reranker_experiment,
    load_embeddings_cache,
)
from .profiler_adapter import run_profiler_experiment

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_device() -> str:
    """Get best available device."""
    if torch.cuda.is_available():
        return 'cuda'
    elif torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


def run_single_experiment(
    experiment: ExperimentConfig,
    pens_root: str,
    train_impressions_path: Optional[str] = None,
    valid_impressions_path: Optional[str] = None,
    embeddings_path: Optional[str] = None,
    device: str = 'cpu',
) -> Dict[str, Any]:
    """
    Run a single stress test experiment.
    
    Args:
        experiment: Experiment configuration
        pens_root: Path to PENS dataset root
        train_impressions_path: Path to train impressions JSONL (reranker)
        valid_impressions_path: Path to valid impressions JSONL (reranker)
        embeddings_path: Path to embeddings cache (reranker)
        device: Device to use
        
    Returns:
        Dictionary with experiment results
    """
    start_time = time.time()
    
    result = {
        'experiment': experiment.to_dict(),
        'success': False,
        'error': None,
        'metrics': None,
        'data_stats': None,
        'runtime_seconds': None,
    }
    
    try:
        if experiment.model == 'reranker':
            # Load data
            if not train_impressions_path or not embeddings_path:
                raise ValueError("train_impressions_path and embeddings_path required for reranker")
            
            train_data = load_reranker_data(train_impressions_path)
            valid_data = load_reranker_data(valid_impressions_path) if valid_impressions_path else []
            embeddings_cache = load_embeddings_cache(embeddings_path)
            
            # Sample users
            train_data = sample_users_reranker(
                train_data,
                experiment.user_fraction,
                experiment.seed
            )
            
            # Truncate history
            train_data = truncate_history_reranker(
                train_data,
                experiment.history_length
            )
            if valid_data:
                valid_data = truncate_history_reranker(
                    valid_data,
                    experiment.history_length
                )
            
            # Get stats
            result['data_stats'] = {
                'train': get_reranker_stats(train_data),
                'valid': get_reranker_stats(valid_data) if valid_data else None,
            }
            
            # Run experiment
            exp_result = run_reranker_experiment(
                train_data,
                valid_data,
                embeddings_cache,
                experiment.model_config,
                device
            )
            
            result['metrics'] = exp_result['eval_metrics']
            result['training_metrics'] = {
                'final_val_loss': exp_result['final_val_loss'],
                'final_val_acc': exp_result['final_val_acc'],
            }
            result['success'] = True
            
        elif experiment.model == 'profiler':
            # Load data
            df = load_profiler_data(pens_root, split='train')
            
            # Sample users
            df = sample_users_profiler(
                df,
                experiment.user_fraction,
                experiment.seed
            )
            
            # Truncate history
            df = truncate_history_profiler(
                df,
                experiment.history_length
            )
            
            # Get stats
            result['data_stats'] = get_profiler_stats(df)
            
            # Run experiment
            exp_result = run_profiler_experiment(
                df,
                experiment.model_config
            )
            
            if exp_result['success']:
                result['metrics'] = exp_result['metrics']
                result['success'] = True
            else:
                result['error'] = exp_result['error']
        
        else:
            raise ValueError(f"Unknown model: {experiment.model}")
            
    except Exception as e:
        logger.error(f"Experiment failed: {e}")
        result['error'] = str(e)
    
    result['runtime_seconds'] = time.time() - start_time
    
    return result


def save_experiment_result(
    result: Dict[str, Any],
    output_dir: Path,
    experiment_name: str,
) -> str:
    """
    Save experiment result to JSON file.
    
    Args:
        result: Experiment result dictionary
        output_dir: Output directory
        experiment_name: Experiment name for file naming
        
    Returns:
        Path to saved file
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{experiment_name}.json"
    
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2, default=str)
    
    return str(output_path)


def run_experiment_grid(
    pens_root: str,
    output_dir: str,
    train_impressions_path: Optional[str] = None,
    valid_impressions_path: Optional[str] = None,
    embeddings_path: Optional[str] = None,
    models: List[str] = ['reranker', 'profiler'],
    user_fractions: Optional[List[float]] = None,
    history_lengths: Optional[List[int]] = None,
    n_seeds: int = 3,
    device: str = None,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """
    Run full experiment grid.
    
    Args:
        pens_root: Path to PENS dataset root
        output_dir: Output directory for results
        train_impressions_path: Path to train impressions JSONL
        valid_impressions_path: Path to valid impressions JSONL
        embeddings_path: Path to embeddings cache
        models: List of models to test
        user_fractions: List of user fractions (default: from config)
        history_lengths: List of history lengths (default: from config)
        n_seeds: Number of random seeds
        device: Device to use (default: auto-detect)
        dry_run: If True, print plan without running
        
    Returns:
        Dictionary with all results and summary
    """
    if device is None:
        device = get_device()
    
    logger.info(f"Using device: {device}")
    
    # Generate experiment grid
    experiments = generate_experiment_grid(
        models=models,
        user_fractions=user_fractions,
        history_lengths=history_lengths,
        n_seeds=n_seeds,
    )
    
    n_experiments = len(experiments)
    logger.info(f"Generated {n_experiments} experiments")
    
    # Print experiment summary
    logger.info("\nExperiment Grid Summary:")
    logger.info(f"  Models: {models}")
    logger.info(f"  User fractions: {user_fractions or EXPERIMENT_GRID['user_fractions']}")
    logger.info(f"  History lengths: {history_lengths or EXPERIMENT_GRID['history_lengths']}")
    logger.info(f"  Seeds: {n_seeds}")
    logger.info(f"  Total experiments: {n_experiments}")
    
    if dry_run:
        logger.info("\n[DRY RUN] Would run the following experiments:")
        for i, exp in enumerate(experiments[:10]):
            logger.info(f"  {i+1}. {exp.experiment_name} ({exp.model})")
        if n_experiments > 10:
            logger.info(f"  ... and {n_experiments - 10} more")
        return {'dry_run': True, 'n_experiments': n_experiments}
    
    # Setup output directory
    output_path = Path(output_dir)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = output_path / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Save run config
    run_config = {
        'pens_root': pens_root,
        'train_impressions_path': train_impressions_path,
        'valid_impressions_path': valid_impressions_path,
        'embeddings_path': embeddings_path,
        'models': models,
        'user_fractions': user_fractions or EXPERIMENT_GRID['user_fractions'],
        'history_lengths': history_lengths or EXPERIMENT_GRID['history_lengths'],
        'n_seeds': n_seeds,
        'device': device,
        'n_experiments': n_experiments,
        'timestamp': timestamp,
    }
    
    with open(run_dir / 'config.json', 'w') as f:
        json.dump(run_config, f, indent=2)
    
    # Run experiments
    results = []
    
    for i, exp in enumerate(experiments):
        logger.info(f"\n{'='*60}")
        logger.info(f"Running experiment {i+1}/{n_experiments}: {exp.experiment_name}")
        logger.info(f"Model: {exp.model}, Users: {exp.user_fraction}, History: {exp.history_length}, Seed: {exp.seed}")
        logger.info(f"{'='*60}")
        
        result = run_single_experiment(
            exp,
            pens_root=pens_root,
            train_impressions_path=train_impressions_path,
            valid_impressions_path=valid_impressions_path,
            embeddings_path=embeddings_path,
            device=device,
        )
        
        # Save individual result
        model_dir = run_dir / 'experiments' / exp.model
        save_experiment_result(result, model_dir, exp.experiment_name)
        
        results.append(result)
        
        # Log summary
        if result['success']:
            metrics_str = ', '.join(
                f"{k}={v:.4f}" for k, v in (result.get('metrics') or {}).items()
                if isinstance(v, (int, float))
            )
            logger.info(f"Success! Metrics: {metrics_str}")
        else:
            logger.error(f"Failed: {result['error']}")
        
        logger.info(f"Runtime: {result['runtime_seconds']:.1f}s")
    
    # Save all results
    all_results_path = run_dir / 'all_results.json'
    with open(all_results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Generate summary
    summary = generate_run_summary(results)
    summary_path = run_dir / 'summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"\n{'='*60}")
    logger.info("EXPERIMENT GRID COMPLETE")
    logger.info(f"{'='*60}")
    logger.info(f"Results saved to: {run_dir}")
    logger.info(f"Total experiments: {n_experiments}")
    logger.info(f"Successful: {summary['n_successful']}")
    logger.info(f"Failed: {summary['n_failed']}")
    logger.info(f"Total runtime: {summary['total_runtime_seconds']:.1f}s")
    
    return {
        'run_dir': str(run_dir),
        'results': results,
        'summary': summary,
    }


def generate_run_summary(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Generate summary statistics for experiment run.
    
    Args:
        results: List of experiment results
        
    Returns:
        Summary dictionary
    """
    n_total = len(results)
    n_successful = sum(1 for r in results if r['success'])
    n_failed = n_total - n_successful
    
    total_runtime = sum(r.get('runtime_seconds', 0) or 0 for r in results)
    
    # Aggregate by model
    model_summaries = {}
    for model in ['reranker', 'profiler']:
        model_results = [r for r in results if r['experiment']['model'] == model]
        if model_results:
            successful = [r for r in model_results if r['success']]
            model_summaries[model] = {
                'n_total': len(model_results),
                'n_successful': len(successful),
                'avg_runtime': sum(r.get('runtime_seconds', 0) or 0 for r in model_results) / len(model_results),
            }
    
    return {
        'n_total': n_total,
        'n_successful': n_successful,
        'n_failed': n_failed,
        'total_runtime_seconds': total_runtime,
        'model_summaries': model_summaries,
    }


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Run stress testing experiment grid for PENS models'
    )
    
    parser.add_argument(
        '--pens_root',
        type=str,
        required=True,
        help='Path to PENS dataset root directory'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default='stress_test_results',
        help='Output directory for results'
    )
    
    parser.add_argument(
        '--train_impressions',
        type=str,
        default=None,
        help='Path to train impressions JSONL (for reranker)'
    )
    
    parser.add_argument(
        '--valid_impressions',
        type=str,
        default=None,
        help='Path to valid impressions JSONL (for reranker)'
    )
    
    parser.add_argument(
        '--embeddings',
        type=str,
        default=None,
        help='Path to embeddings cache .pt file (for reranker)'
    )
    
    parser.add_argument(
        '--models',
        type=str,
        nargs='+',
        default=['reranker', 'profiler'],
        choices=['reranker', 'profiler'],
        help='Models to test'
    )
    
    parser.add_argument(
        '--user_fractions',
        type=float,
        nargs='+',
        default=None,
        help='User fractions to test (default: 0.1 0.25 0.5 0.75 1.0)'
    )
    
    parser.add_argument(
        '--history_lengths',
        type=int,
        nargs='+',
        default=None,
        help='History lengths to test (default: 5 10 20 30 50)'
    )
    
    parser.add_argument(
        '--n_seeds',
        type=int,
        default=3,
        help='Number of random seeds for variance estimation'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        choices=['cpu', 'cuda', 'mps'],
        help='Device to use (default: auto-detect)'
    )
    
    parser.add_argument(
        '--dry_run',
        action='store_true',
        help='Print experiment plan without running'
    )
    
    parser.add_argument(
        '--log_level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logging.getLogger().setLevel(args.log_level)
    
    # Validate paths
    if 'reranker' in args.models:
        if not args.train_impressions:
            parser.error("--train_impressions required when testing reranker")
        if not args.embeddings:
            parser.error("--embeddings required when testing reranker")
    
    # Run experiment grid
    run_experiment_grid(
        pens_root=args.pens_root,
        output_dir=args.output_dir,
        train_impressions_path=args.train_impressions,
        valid_impressions_path=args.valid_impressions,
        embeddings_path=args.embeddings,
        models=args.models,
        user_fractions=args.user_fractions,
        history_lengths=args.history_lengths,
        n_seeds=args.n_seeds,
        device=args.device,
        dry_run=args.dry_run,
    )


if __name__ == '__main__':
    main()
