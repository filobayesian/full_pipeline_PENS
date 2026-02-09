"""
CLI Entry Point for Stress Testing Framework

Usage:
    # Run full experiment grid
    python -m stress_test run \
        --pens_root data/PENS \
        --train_impressions artifacts/train_impressions.jsonl \
        --valid_impressions artifacts/valid_impressions.jsonl \
        --embeddings artifacts/news_embeddings.pt \
        --output_dir stress_test_results/

    # Run with custom grid
    python -m stress_test run \
        --pens_root data/PENS \
        --models profiler \
        --user_fractions 0.1 0.5 1.0 \
        --history_lengths 10 30 \
        --n_seeds 1

    # Dry run to see experiment plan
    python -m stress_test run --pens_root data/PENS --dry_run

    # Analyze existing results
    python -m stress_test analyze --run_dir stress_test_results/run_20240101_120000/

    # Show help
    python -m stress_test --help
"""

import argparse
import sys
import logging


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog='stress_test',
        description='Stress Testing Framework for PENS Models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run profiler-only experiments (faster, no GPU needed)
  python -m stress_test run --pens_root data/PENS --models profiler

  # Run quick test (small grid)
  python -m stress_test run --pens_root data/PENS \\
      --models profiler --user_fractions 0.25 1.0 \\
      --history_lengths 10 50 --n_seeds 1

  # Run full reranker experiments
  python -m stress_test run --pens_root data/PENS \\
      --train_impressions artifacts/train_impressions.jsonl \\
      --valid_impressions artifacts/valid_impressions.jsonl \\
      --embeddings artifacts/news_embeddings.pt

  # Analyze completed run
  python -m stress_test analyze --run_dir stress_test_results/run_20240101_120000/
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # =========================================================================
    # Run command
    # =========================================================================
    run_parser = subparsers.add_parser(
        'run',
        help='Run stress testing experiments'
    )
    
    run_parser.add_argument(
        '--pens_root',
        type=str,
        required=True,
        help='Path to PENS dataset root directory'
    )
    
    run_parser.add_argument(
        '--output_dir',
        type=str,
        default='stress_test_results',
        help='Output directory for results (default: stress_test_results)'
    )
    
    run_parser.add_argument(
        '--train_impressions',
        type=str,
        default=None,
        help='Path to train impressions JSONL (required for reranker)'
    )
    
    run_parser.add_argument(
        '--valid_impressions',
        type=str,
        default=None,
        help='Path to valid impressions JSONL (required for reranker)'
    )
    
    run_parser.add_argument(
        '--embeddings',
        type=str,
        default=None,
        help='Path to embeddings cache .pt file (required for reranker)'
    )
    
    run_parser.add_argument(
        '--models',
        type=str,
        nargs='+',
        default=['reranker', 'profiler'],
        choices=['reranker', 'profiler'],
        help='Models to test (default: both)'
    )
    
    run_parser.add_argument(
        '--user_fractions',
        type=float,
        nargs='+',
        default=None,
        help='User fractions to test (default: 0.1 0.25 0.5 0.75 1.0)'
    )
    
    run_parser.add_argument(
        '--history_lengths',
        type=int,
        nargs='+',
        default=None,
        help='History lengths to test (default: 5 10 20 30 50)'
    )
    
    run_parser.add_argument(
        '--n_seeds',
        type=int,
        default=3,
        help='Number of random seeds (default: 3)'
    )
    
    run_parser.add_argument(
        '--device',
        type=str,
        default=None,
        choices=['cpu', 'cuda', 'mps'],
        help='Device to use (default: auto-detect)'
    )
    
    run_parser.add_argument(
        '--dry_run',
        action='store_true',
        help='Print experiment plan without running'
    )
    
    run_parser.add_argument(
        '--log_level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level'
    )
    
    # =========================================================================
    # Analyze command
    # =========================================================================
    analyze_parser = subparsers.add_parser(
        'analyze',
        help='Analyze completed experiment results'
    )
    
    analyze_parser.add_argument(
        '--run_dir',
        type=str,
        required=True,
        help='Path to experiment run directory'
    )
    
    analyze_parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Output directory for analysis (default: same as run_dir)'
    )
    
    analyze_parser.add_argument(
        '--log_level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level'
    )
    
    # =========================================================================
    # Info command
    # =========================================================================
    info_parser = subparsers.add_parser(
        'info',
        help='Show experiment grid information'
    )
    
    info_parser.add_argument(
        '--models',
        type=str,
        nargs='+',
        default=['reranker', 'profiler'],
        choices=['reranker', 'profiler'],
        help='Models to include'
    )
    
    info_parser.add_argument(
        '--user_fractions',
        type=float,
        nargs='+',
        default=None,
        help='User fractions'
    )
    
    info_parser.add_argument(
        '--history_lengths',
        type=int,
        nargs='+',
        default=None,
        help='History lengths'
    )
    
    info_parser.add_argument(
        '--n_seeds',
        type=int,
        default=3,
        help='Number of seeds'
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return 1
    
    # Setup logging
    log_level = getattr(args, 'log_level', 'INFO')
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Execute command
    if args.command == 'run':
        from .runner import run_experiment_grid
        
        # Validate paths for reranker
        if 'reranker' in args.models:
            if not args.train_impressions:
                print("Error: --train_impressions required when testing reranker")
                return 1
            if not args.embeddings:
                print("Error: --embeddings required when testing reranker")
                return 1
        
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
        
    elif args.command == 'analyze':
        from .analysis import analyze_run
        
        result = analyze_run(
            run_dir=args.run_dir,
            output_dir=args.output_dir,
        )
        
        print(f"\nAnalysis complete!")
        print(f"Report: {result['report_path']}")
        print(f"Plots: {len(result['plots'])} generated")
        
    elif args.command == 'info':
        from .config import (
            EXPERIMENT_GRID,
            RERANKER_CONFIG,
            PROFILER_CONFIG,
            count_experiments,
        )
        
        user_fractions = args.user_fractions or EXPERIMENT_GRID['user_fractions']
        history_lengths = args.history_lengths or EXPERIMENT_GRID['history_lengths']
        n_seeds = args.n_seeds
        
        n_experiments = count_experiments(
            models=args.models,
            user_fractions=user_fractions,
            history_lengths=history_lengths,
            n_seeds=n_seeds,
        )
        
        print("\n" + "=" * 60)
        print("STRESS TESTING EXPERIMENT GRID")
        print("=" * 60)
        print(f"\nModels: {args.models}")
        print(f"User fractions: {user_fractions}")
        print(f"History lengths: {history_lengths}")
        print(f"Seeds: {n_seeds}")
        print(f"\nTotal experiments: {n_experiments}")
        
        # Estimate time
        if 'reranker' in args.models:
            reranker_exp = len(user_fractions) * len(history_lengths) * n_seeds
            print(f"\nReranker: {reranker_exp} experiments")
            print(f"  Estimated time: {reranker_exp * 5:.0f} - {reranker_exp * 10:.0f} minutes")
        
        if 'profiler' in args.models:
            profiler_exp = len(user_fractions) * len(history_lengths) * n_seeds
            print(f"\nProfiler: {profiler_exp} experiments")
            print(f"  Estimated time: {profiler_exp * 1:.0f} - {profiler_exp * 2:.0f} minutes")
        
        print("\n" + "=" * 60)
        
        # Print configs
        if 'reranker' in args.models:
            print("\nReranker Config:")
            for k, v in RERANKER_CONFIG.items():
                print(f"  {k}: {v}")
        
        if 'profiler' in args.models:
            print("\nProfiler Config:")
            for k, v in PROFILER_CONFIG.items():
                print(f"  {k}: {v}")
        
        print()
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
