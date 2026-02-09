"""
Results Analysis and Visualization for Stress Testing

This module provides tools for aggregating experiment results and generating
visualizations to understand model sensitivity to data availability.

Key functions:
- load_experiment_results: Load results from experiment run
- aggregate_results: Aggregate metrics across seeds
- generate_sensitivity_plots: Create visualization plots
- generate_report: Create markdown summary report
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Try to import visualization libraries
try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    logger.warning("matplotlib not available - visualization disabled")

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False


def load_experiment_results(run_dir: str) -> Dict[str, Any]:
    """
    Load all experiment results from a run directory.
    
    Args:
        run_dir: Path to experiment run directory
        
    Returns:
        Dictionary with config and results
    """
    run_path = Path(run_dir)
    
    # Load config
    config_path = run_path / 'config.json'
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Load all results
    results_path = run_path / 'all_results.json'
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    return {
        'config': config,
        'results': results,
        'run_dir': str(run_path),
    }


def results_to_dataframe(results: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Convert results list to DataFrame for analysis.
    
    Args:
        results: List of experiment result dictionaries
        
    Returns:
        DataFrame with flattened results
    """
    rows = []
    
    for result in results:
        exp = result.get('experiment', {})
        metrics = result.get('metrics', {}) or {}
        rewriter_metrics = result.get('rewriter_metrics', {}) or {}
        rewriter_degradation = result.get('rewriter_degradation', {}) or {}
        data_stats = result.get('data_stats', {}) or {}
        
        row = {
            'model': exp.get('model'),
            'user_count': exp.get('user_count'),
            'history_length': exp.get('history_length'),
            'seed': exp.get('seed'),
            'experiment_name': exp.get('experiment_name'),
            'success': result.get('success', False),
            'runtime_seconds': result.get('runtime_seconds'),
        }
        
        # Add metrics
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                row[f'metric_{key}'] = value

        for key, value in rewriter_metrics.items():
            if isinstance(value, (int, float)):
                row[f'rewriter_{key}'] = value

        for key, value in rewriter_degradation.items():
            if isinstance(value, (int, float)):
                row[f'degradation_{key}'] = value
        
        # Add data stats (handle nested for reranker)
        if isinstance(data_stats, dict):
            if 'train' in data_stats:
                for key, value in (data_stats.get('train') or {}).items():
                    if isinstance(value, (int, float)):
                        row[f'train_{key}'] = value
            else:
                for key, value in data_stats.items():
                    if isinstance(value, (int, float)):
                        row[f'data_{key}'] = value
        
        rows.append(row)
    
    return pd.DataFrame(rows)


def aggregate_results(
    df: pd.DataFrame,
    group_cols: List[str] = ['model', 'user_count', 'history_length'],
) -> pd.DataFrame:
    """
    Aggregate results across seeds, computing mean and std.
    
    Args:
        df: Results DataFrame
        group_cols: Columns to group by
        
    Returns:
        Aggregated DataFrame
    """
    # Get metric columns
    metric_cols = [col for col in df.columns if col.startswith('metric_')]
    
    if not metric_cols:
        logger.warning("No metric columns found")
        return df
    
    # Aggregate
    agg_funcs = {}
    for col in metric_cols:
        agg_funcs[col] = ['mean', 'std', 'count']
    
    agg_funcs['success'] = 'sum'
    agg_funcs['runtime_seconds'] = 'mean'
    
    aggregated = df.groupby(group_cols).agg(agg_funcs)
    
    # Flatten column names
    aggregated.columns = [
        f"{col[0]}_{col[1]}" if col[1] else col[0]
        for col in aggregated.columns
    ]
    
    return aggregated.reset_index()


def compute_sensitivity_scores(
    df: pd.DataFrame,
    metric_col: str = 'metric_auc_mean',
) -> Dict[str, float]:
    """
    Compute sensitivity scores for user count and history length.
    
    Sensitivity = relative performance drop from max to min setting.
    
    Args:
        df: Aggregated results DataFrame
        metric_col: Column to compute sensitivity for
        
    Returns:
        Dictionary with sensitivity scores
    """
    if metric_col not in df.columns:
        return {}
    
    # Sensitivity to user count (averaged over history lengths)
    user_effect = df.groupby('user_count')[metric_col].mean()
    user_sensitivity = (user_effect.max() - user_effect.min()) / user_effect.max()
    
    # Sensitivity to history length (averaged over user counts)
    history_effect = df.groupby('history_length')[metric_col].mean()
    history_sensitivity = (history_effect.max() - history_effect.min()) / history_effect.max()
    
    return {
        'user_count_sensitivity': user_sensitivity,
        'history_length_sensitivity': history_sensitivity,
        'user_count_max': user_effect.idxmax(),
        'history_length_max': history_effect.idxmax(),
    }


def find_minimum_requirements(
    df: pd.DataFrame,
    metric_col: str = 'metric_auc_mean',
    threshold_fraction: float = 0.9,
) -> Dict[str, Any]:
    """
    Find minimum data requirements to achieve threshold of max performance.
    
    Args:
        df: Aggregated results DataFrame
        metric_col: Metric to analyze
        threshold_fraction: Fraction of max performance to achieve (e.g., 0.9 = 90%)
        
    Returns:
        Dictionary with minimum requirements
    """
    if metric_col not in df.columns:
        return {}
    
    max_metric = df[metric_col].max()
    threshold = max_metric * threshold_fraction
    
    # Find minimum user count for threshold
    user_means = df.groupby('user_count')[metric_col].mean()
    min_user_count = None
    for uc in sorted(user_means.index):
        if user_means[uc] >= threshold:
            min_user_count = uc
            break
    
    # Find minimum history length for threshold
    history_means = df.groupby('history_length')[metric_col].mean()
    min_history_length = None
    for hl in sorted(history_means.index):
        if history_means[hl] >= threshold:
            min_history_length = hl
            break
    
    return {
        'max_metric': max_metric,
        'threshold': threshold,
        'threshold_fraction': threshold_fraction,
        'min_user_count': min_user_count,
        'min_history_length': min_history_length,
    }


def generate_sensitivity_plots(
    df: pd.DataFrame,
    output_dir: str,
    model: str,
    metric_col: str = 'metric_auc_mean',
    metric_std_col: str = 'metric_auc_std',
    metric_name: str = 'AUC',
) -> List[str]:
    """
    Generate sensitivity plots for a model.
    
    Args:
        df: Aggregated results DataFrame
        output_dir: Output directory for plots
        model: Model name
        metric_col: Mean metric column
        metric_std_col: Std metric column
        metric_name: Display name for metric
        
    Returns:
        List of saved plot paths
    """
    if not HAS_MATPLOTLIB:
        logger.warning("matplotlib not available - skipping plots")
        return []
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Filter to model
    model_df = df[df['model'] == model].copy()
    
    if len(model_df) == 0:
        logger.warning(f"No data for model {model}")
        return []
    
    if metric_col not in model_df.columns:
        logger.warning(f"Metric {metric_col} not found")
        return []
    
    plot_paths = []
    
    # Plot 1: Metric vs User Count (lines for history lengths)
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for hl in sorted(model_df['history_length'].unique()):
        subset = model_df[model_df['history_length'] == hl]
        subset = subset.sort_values('user_count')
        
        ax.plot(
            subset['user_count'],
            subset[metric_col],
            marker='o',
            label=f'History={hl}'
        )
        
        if metric_std_col in subset.columns:
            ax.fill_between(
                subset['user_count'],
                subset[metric_col] - subset[metric_std_col],
                subset[metric_col] + subset[metric_std_col],
                alpha=0.2
            )
    
    ax.set_xlabel('User Count')
    ax.set_ylabel(metric_name)
    ax.set_title(f'{model.title()}: {metric_name} vs User Count')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plot_path = output_path / f'{model}_sensitivity_users.png'
    fig.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    plot_paths.append(str(plot_path))
    
    # Plot 2: Metric vs History Length (lines for user counts)
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for uc in sorted(model_df['user_count'].unique()):
        subset = model_df[model_df['user_count'] == uc]
        subset = subset.sort_values('history_length')
        
        ax.plot(
            subset['history_length'],
            subset[metric_col],
            marker='o',
            label=f'Users={uc}'
        )
        
        if metric_std_col in subset.columns:
            ax.fill_between(
                subset['history_length'],
                subset[metric_col] - subset[metric_std_col],
                subset[metric_col] + subset[metric_std_col],
                alpha=0.2
            )
    
    ax.set_xlabel('History Length')
    ax.set_ylabel(metric_name)
    ax.set_title(f'{model.title()}: {metric_name} vs History Length')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plot_path = output_path / f'{model}_sensitivity_history.png'
    fig.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    plot_paths.append(str(plot_path))
    
    # Plot 3: Heatmap
    if HAS_SEABORN:
        fig, ax = plt.subplots(figsize=(10, 8))
        
        pivot = model_df.pivot_table(
            index='history_length',
            columns='user_count',
            values=metric_col,
            aggfunc='mean'
        )
        
        sns.heatmap(
            pivot,
            annot=True,
            fmt='.3f',
            cmap='RdYlGn',
            ax=ax
        )
        
        ax.set_title(f'{model.title()}: {metric_name} Heatmap')
        ax.set_xlabel('User Count')
        ax.set_ylabel('History Length')
        
        plot_path = output_path / f'{model}_heatmap.png'
        fig.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        plot_paths.append(str(plot_path))
    
    logger.info(f"Generated {len(plot_paths)} plots for {model}")
    
    return plot_paths


def generate_all_plots(
    results: Dict[str, Any],
    output_dir: str,
) -> Dict[str, List[str]]:
    """
    Generate all visualization plots for experiment results.
    
    Args:
        results: Results dictionary from load_experiment_results
        output_dir: Output directory for plots
        
    Returns:
        Dictionary mapping model to plot paths
    """
    df = results_to_dataframe(results['results'])
    agg_df = aggregate_results(df)
    
    all_plots = {}
    
    # Reranker plots
    reranker_df = agg_df[agg_df['model'] == 'reranker']
    if len(reranker_df) > 0:
        plots = generate_sensitivity_plots(
            agg_df, output_dir, 'reranker',
            metric_col='metric_auc_mean',
            metric_std_col='metric_auc_std',
            metric_name='AUC'
        )
        all_plots['reranker'] = plots
        
        # Also plot MRR
        plots_mrr = generate_sensitivity_plots(
            agg_df, output_dir, 'reranker',
            metric_col='metric_mrr_mean',
            metric_std_col='metric_mrr_std',
            metric_name='MRR'
        )
        all_plots['reranker'].extend(plots_mrr)
    
    # Profiler plots
    profiler_df = agg_df[agg_df['model'] == 'profiler']
    if len(profiler_df) > 0:
        plots = generate_sensitivity_plots(
            agg_df, output_dir, 'profiler',
            metric_col='metric_mean_abs_z_mean',
            metric_std_col='metric_mean_abs_z_std',
            metric_name='Mean |z-score|'
        )
        all_plots['profiler'] = plots
        
        # Also plot coverage
        plots_coverage = generate_sensitivity_plots(
            agg_df, output_dir, 'profiler',
            metric_col='metric_coverage_mean',
            metric_std_col='metric_coverage_std',
            metric_name='Coverage'
        )
        all_plots['profiler'].extend(plots_coverage)
    
    return all_plots


def generate_report(
    results: Dict[str, Any],
    output_path: str,
) -> str:
    """
    Generate markdown summary report.
    
    Args:
        results: Results dictionary from load_experiment_results
        output_path: Output path for report
        
    Returns:
        Path to generated report
    """
    df = results_to_dataframe(results['results'])
    agg_df = aggregate_results(df)
    
    lines = [
        "# Stress Testing Results Report",
        "",
        f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Experiment Configuration",
        "",
        f"- User counts: {results['config'].get('user_counts', [])}",
        f"- History lengths: {results['config'].get('history_lengths', [])}",
        f"- Number of seeds: {results['config'].get('n_seeds', 0)}",
        f"- Total experiments: {results['config'].get('n_experiments', 0)}",
        "",
    ]
    
    # Reranker results
    reranker_df = agg_df[agg_df['model'] == 'reranker']
    if len(reranker_df) > 0:
        lines.extend([
            "## Reranker Results",
            "",
        ])
        
        if 'metric_auc_mean' in reranker_df.columns:
            sens = compute_sensitivity_scores(reranker_df, 'metric_auc_mean')
            min_req = find_minimum_requirements(reranker_df, 'metric_auc_mean')
            
            lines.extend([
                "### AUC Sensitivity",
                "",
                f"- User count sensitivity: {sens.get('user_count_sensitivity', 0):.3f}",
                f"- History length sensitivity: {sens.get('history_length_sensitivity', 0):.3f}",
                f"- Max AUC: {min_req.get('max_metric', 0):.4f}",
                f"- Min user count for 90% max: {min_req.get('min_user_count')}",
                f"- Min history length for 90% max: {min_req.get('min_history_length')}",
                "",
            ])
        
        # Summary table
        lines.extend([
            "### Summary by Configuration",
            "",
            "| User Count | History | AUC | MRR | nDCG@10 |",
            "|-----------|---------|-----|-----|---------|",
        ])
        
        for _, row in reranker_df.iterrows():
            auc = row.get('metric_auc_mean', 0) or 0
            mrr = row.get('metric_mrr_mean', 0) or 0
            ndcg = row.get('metric_ndcg@10_mean', 0) or 0
            lines.append(
                f"| {row['user_count']} | {row['history_length']} | "
                f"{auc:.4f} | {mrr:.4f} | {ndcg:.4f} |"
            )
        
        lines.append("")
    
    # Profiler results
    profiler_df = agg_df[agg_df['model'] == 'profiler']
    if len(profiler_df) > 0:
        lines.extend([
            "## Profiler Results",
            "",
        ])
        
        if 'metric_mean_abs_z_mean' in profiler_df.columns:
            sens = compute_sensitivity_scores(profiler_df, 'metric_mean_abs_z_mean')
            min_req = find_minimum_requirements(profiler_df, 'metric_mean_abs_z_mean')
            
            lines.extend([
                "### Mean |z-score| Sensitivity",
                "",
                f"- User count sensitivity: {sens.get('user_count_sensitivity', 0):.3f}",
                f"- History length sensitivity: {sens.get('history_length_sensitivity', 0):.3f}",
                f"- Max mean |z|: {min_req.get('max_metric', 0):.4f}",
                f"- Min user count for 90% max: {min_req.get('min_user_count')}",
                f"- Min history length for 90% max: {min_req.get('min_history_length')}",
                "",
            ])
        
        # Summary table
        lines.extend([
            "### Summary by Configuration",
            "",
            "| User Count | History | Mean |z| | Strong Prefs | Coverage |",
            "|-----------|---------|---------|--------------|----------|",
        ])
        
        for _, row in profiler_df.iterrows():
            mean_z = row.get('metric_mean_abs_z_mean', 0) or 0
            strong = row.get('metric_n_strong_prefs_mean', 0) or 0
            coverage = row.get('metric_coverage_mean', 0) or 0
            lines.append(
                f"| {row['user_count']} | {row['history_length']} | "
                f"{mean_z:.4f} | {strong:.2f} | {coverage:.4f} |"
            )
        
        lines.append("")

        # Rewriter degradation summary (if available)
        if 'degradation_style_lift_ratio_mean' in profiler_df.columns:
            lines.extend([
                "### Rewriter Degradation vs Best-Case",
                "",
                "| User Count | History | Lift Ratio | Factual Ratio | Consistency Ratio |",
                "|-----------|---------|-----------|---------------|-------------------|",
            ])
            for _, row in profiler_df.iterrows():
                lift_ratio = row.get('degradation_style_lift_ratio_mean', 0) or 0
                factual_ratio = row.get('degradation_factual_ratio_mean', 0) or 0
                consistency_ratio = row.get('degradation_content_consistent_ratio_mean', 0) or 0
                lines.append(
                    f"| {row['user_count']} | {row['history_length']} | "
                    f"{lift_ratio:.3f} | {factual_ratio:.3f} | {consistency_ratio:.3f} |"
                )
            lines.append("")
    
    # Write report
    report_content = '\n'.join(lines)
    
    with open(output_path, 'w') as f:
        f.write(report_content)
    
    logger.info(f"Generated report: {output_path}")
    
    return output_path


def analyze_run(run_dir: str, output_dir: Optional[str] = None) -> Dict[str, Any]:
    """
    Complete analysis of an experiment run.
    
    Args:
        run_dir: Path to experiment run directory
        output_dir: Output directory for plots/report (default: run_dir)
        
    Returns:
        Dictionary with analysis results
    """
    if output_dir is None:
        output_dir = run_dir
    
    # Load results
    results = load_experiment_results(run_dir)
    
    # Convert to DataFrame
    df = results_to_dataframe(results['results'])
    
    # Aggregate
    agg_df = aggregate_results(df)
    
    # Save aggregated results
    agg_path = Path(output_dir) / 'aggregated'
    agg_path.mkdir(parents=True, exist_ok=True)
    
    reranker_df = agg_df[agg_df['model'] == 'reranker']
    if len(reranker_df) > 0:
        reranker_df.to_csv(agg_path / 'reranker_results.csv', index=False)
    
    profiler_df = agg_df[agg_df['model'] == 'profiler']
    if len(profiler_df) > 0:
        profiler_df.to_csv(agg_path / 'profiler_results.csv', index=False)
    
    # Generate plots
    plots_dir = Path(output_dir) / 'plots'
    plots = generate_all_plots(results, str(plots_dir))
    
    # Generate report
    report_path = Path(output_dir) / 'report.md'
    generate_report(results, str(report_path))
    
    return {
        'aggregated_df': agg_df,
        'plots': plots,
        'report_path': str(report_path),
    }
