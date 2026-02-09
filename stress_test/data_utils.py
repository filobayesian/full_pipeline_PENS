"""
Data Utilities for Stress Testing

This module provides unified data loading, user sampling, and history truncation
for both the reranker and style profiler models.

Key functions:
- load_reranker_data: Load impressions JSONL for reranker
- load_profiler_data: Load canonical DataFrame for profiler
- sample_users_*: Sample a fixed number of users
- truncate_history_*: Truncate user history to max length
"""

import json
import logging
import random
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import pandas as pd
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# Reranker Data Loading and Manipulation
# =============================================================================

def load_reranker_data(
    impressions_path: str,
) -> List[Dict[str, Any]]:
    """
    Load impressions JSONL file for reranker.
    
    Args:
        impressions_path: Path to impressions JSONL file
        
    Returns:
        List of impression dictionaries
    """
    logger.info(f"Loading reranker data from {impressions_path}")
    
    impressions = []
    with open(impressions_path, 'r') as f:
        for line in f:
            impressions.append(json.loads(line))
    
    logger.info(f"Loaded {len(impressions)} impressions")
    
    # Get unique users
    user_ids = set(imp.get('user_id') for imp in impressions)
    logger.info(f"Unique users: {len(user_ids)}")
    
    return impressions


def save_reranker_data(
    impressions: List[Dict[str, Any]],
    output_path: str,
) -> None:
    """
    Save impressions to JSONL file.
    
    Args:
        impressions: List of impression dictionaries
        output_path: Output file path
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        for imp in impressions:
            f.write(json.dumps(imp) + '\n')
    
    logger.info(f"Saved {len(impressions)} impressions to {output_path}")


def sample_users_reranker(
    impressions: List[Dict[str, Any]],
    n_users: int,
    seed: int,
) -> List[Dict[str, Any]]:
    """
    Sample an absolute number of users, keeping ALL their impressions.
    
    Important: We sample users, not impressions, to preserve user-level structure.
    
    Args:
        impressions: List of impression dictionaries
        n_users: Number of users to keep (>= 1)
        seed: Random seed for reproducibility
        
    Returns:
        Filtered list of impressions
    """
    # Get unique user IDs
    user_ids = list(set(imp.get('user_id') for imp in impressions))
    total_users = len(user_ids)
    n_sample = max(1, min(int(n_users), total_users))
    
    # Sample users
    random.seed(seed)
    sampled_users = set(random.sample(user_ids, n_sample))
    
    # Filter impressions
    filtered = [imp for imp in impressions if imp.get('user_id') in sampled_users]
    
    logger.info(
        f"Sampled {len(sampled_users)}/{total_users} users, "
        f"{len(filtered)}/{len(impressions)} impressions"
    )
    
    return filtered


def truncate_history_reranker(
    impressions: List[Dict[str, Any]],
    max_len: int,
) -> List[Dict[str, Any]]:
    """
    Truncate user history in each impression to max_len most recent items.
    
    Args:
        impressions: List of impression dictionaries
        max_len: Maximum history length
        
    Returns:
        List of impressions with truncated history
    """
    truncated = []
    total_items_before = 0
    total_items_after = 0
    
    for imp in impressions:
        new_imp = imp.copy()
        history = imp.get('history', [])
        
        total_items_before += len(history)
        
        # Keep most recent items (last max_len)
        new_imp['history'] = history[-max_len:] if history else []
        
        # Also truncate history_dwelltime if present
        if 'history_dwelltime' in imp:
            dwelltime = imp['history_dwelltime']
            new_imp['history_dwelltime'] = dwelltime[-max_len:] if dwelltime else []
        
        total_items_after += len(new_imp['history'])
        truncated.append(new_imp)
    
    logger.info(
        f"Truncated history to max {max_len}: "
        f"{total_items_before} -> {total_items_after} items "
        f"({total_items_after/max(total_items_before,1)*100:.1f}% retained)"
    )
    
    return truncated


def filter_impressions_by_users(
    impressions: List[Dict[str, Any]],
    user_ids: set,
) -> List[Dict[str, Any]]:
    """
    Filter impressions to a set of user IDs.

    Args:
        impressions: List of impression dictionaries
        user_ids: Set of allowed user IDs

    Returns:
        Filtered list of impressions
    """
    if not user_ids:
        return []

    filtered = [imp for imp in impressions if imp.get('user_id') in user_ids]
    logger.info(
        f"Filtered impressions by users: {len(filtered)}/{len(impressions)} retained"
    )
    return filtered


def get_reranker_stats(impressions: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compute statistics for reranker impressions.
    
    Args:
        impressions: List of impression dictionaries
        
    Returns:
        Dictionary of statistics
    """
    n_impressions = len(impressions)
    user_ids = [imp.get('user_id') for imp in impressions]
    n_users = len(set(user_ids))
    
    history_lengths = [len(imp.get('history', [])) for imp in impressions]
    pos_counts = [len(imp.get('pos', [])) for imp in impressions]
    neg_counts = [len(imp.get('neg', [])) for imp in impressions]
    
    return {
        'n_impressions': n_impressions,
        'n_users': n_users,
        'impressions_per_user': n_impressions / max(n_users, 1),
        'history_length_mean': np.mean(history_lengths) if history_lengths else 0,
        'history_length_median': np.median(history_lengths) if history_lengths else 0,
        'history_length_max': max(history_lengths) if history_lengths else 0,
        'pos_per_impression_mean': np.mean(pos_counts) if pos_counts else 0,
        'neg_per_impression_mean': np.mean(neg_counts) if neg_counts else 0,
    }


# =============================================================================
# Profiler Data Loading and Manipulation
# =============================================================================

def load_profiler_data(
    pens_root: str,
    split: str = 'train',
) -> pd.DataFrame:
    """
    Load PENS dataset for style profiler using pens_adapter.
    
    Args:
        pens_root: Root directory containing PENS data
        split: Dataset split ('train', 'valid', 'test')
        
    Returns:
        DataFrame with canonical schema
    """
    # Import pens_adapter from style_steering_PENS
    import sys
    style_steering_path = Path(__file__).parent.parent / 'style_steering_PENS copy'
    if str(style_steering_path) not in sys.path:
        sys.path.insert(0, str(style_steering_path))
    
    from headline_style.pens_adapter import load_pens_dataset
    
    logger.info(f"Loading profiler data from {pens_root}, split={split}")
    df = load_pens_dataset(pens_root, split)
    
    return df


def sample_users_profiler(
    df: pd.DataFrame,
    n_users: int,
    seed: int,
) -> pd.DataFrame:
    """
    Sample an absolute number of users, keeping ALL their interactions.
    
    Args:
        df: DataFrame with canonical schema
        n_users: Number of users to keep (>= 1)
        seed: Random seed for reproducibility
        
    Returns:
        Filtered DataFrame
    """
    # Get unique user IDs
    user_ids = df['user_id'].unique().tolist()
    total_users = len(user_ids)
    n_sample = max(1, min(int(n_users), total_users))
    
    # Sample users
    random.seed(seed)
    sampled_users = set(random.sample(user_ids, n_sample))
    
    # Filter DataFrame
    filtered = df[df['user_id'].isin(sampled_users)].copy()
    
    logger.info(
        f"Sampled {len(sampled_users)}/{total_users} users, "
        f"{len(filtered)}/{len(df)} interactions"
    )
    
    return filtered


def truncate_history_profiler(
    df: pd.DataFrame,
    max_len: int,
) -> pd.DataFrame:
    """
    Truncate to keep only last max_len interactions per user.
    
    For the profiler, history length = number of interactions per user.
    We keep the most recent interactions based on timestamp or row order.
    
    Args:
        df: DataFrame with canonical schema
        max_len: Maximum number of interactions per user
        
    Returns:
        Filtered DataFrame
    """
    total_before = len(df)
    
    # Sort by timestamp if available, otherwise use row order
    if 'timestamp' in df.columns and df['timestamp'].notna().any():
        df = df.sort_values(['user_id', 'timestamp'])
    
    # Keep last max_len per user
    truncated = df.groupby('user_id').tail(max_len).reset_index(drop=True)
    
    logger.info(
        f"Truncated history to max {max_len} per user: "
        f"{total_before} -> {len(truncated)} interactions "
        f"({len(truncated)/max(total_before,1)*100:.1f}% retained)"
    )
    
    return truncated


def get_profiler_stats(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute statistics for profiler DataFrame.
    
    Args:
        df: DataFrame with canonical schema
        
    Returns:
        Dictionary of statistics
    """
    n_interactions = len(df)
    n_users = df['user_id'].nunique()
    
    interactions_per_user = df.groupby('user_id').size()
    
    stats = {
        'n_interactions': n_interactions,
        'n_users': n_users,
        'interactions_per_user_mean': interactions_per_user.mean(),
        'interactions_per_user_median': interactions_per_user.median(),
        'interactions_per_user_min': interactions_per_user.min(),
        'interactions_per_user_max': interactions_per_user.max(),
        'n_unique_headlines': df['headline'].nunique(),
    }
    
    if 'dwell_time' in df.columns:
        stats['dwell_time_mean'] = df['dwell_time'].mean()
        stats['dwell_time_median'] = df['dwell_time'].median()
    
    return stats


# =============================================================================
# Unified Data Processing
# =============================================================================

def prepare_experiment_data(
    pens_root: str,
    user_count: int,
    history_length: int,
    seed: int,
    model: str,
    train_impressions_path: Optional[str] = None,
    valid_impressions_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Prepare data for a single experiment.
    
    Args:
        pens_root: Root directory containing PENS data
        user_count: Number of users to sample
        history_length: Maximum history length
        seed: Random seed
        model: 'reranker' or 'profiler'
        train_impressions_path: Path to train impressions JSONL (for reranker)
        valid_impressions_path: Path to valid impressions JSONL (for reranker)
        
    Returns:
        Dictionary with processed data and statistics
    """
    if model == 'reranker':
        # Load impressions
        if train_impressions_path is None:
            raise ValueError("train_impressions_path required for reranker")
        
        train_data = load_reranker_data(train_impressions_path)
        valid_data = load_reranker_data(valid_impressions_path) if valid_impressions_path else None
        
        # Sample users (only from training data)
        train_data = sample_users_reranker(train_data, user_count, seed)
        
        # Truncate history
        train_data = truncate_history_reranker(train_data, history_length)
        if valid_data:
            valid_data = truncate_history_reranker(valid_data, history_length)
        
        return {
            'train_data': train_data,
            'valid_data': valid_data,
            'train_stats': get_reranker_stats(train_data),
            'valid_stats': get_reranker_stats(valid_data) if valid_data else None,
        }
    
    elif model == 'profiler':
        # Load canonical DataFrame
        df = load_profiler_data(pens_root, split='train')
        
        # Sample users
        df = sample_users_profiler(df, user_count, seed)
        
        # Truncate history
        df = truncate_history_profiler(df, history_length)
        
        return {
            'data': df,
            'stats': get_profiler_stats(df),
        }
    
    else:
        raise ValueError(f"Unknown model: {model}")


def get_user_overlap(
    reranker_impressions: List[Dict[str, Any]],
    profiler_df: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Compute user overlap between reranker and profiler datasets.
    
    Args:
        reranker_impressions: Reranker impression data
        profiler_df: Profiler DataFrame
        
    Returns:
        Dictionary with overlap statistics
    """
    reranker_users = set(imp.get('user_id') for imp in reranker_impressions)
    profiler_users = set(profiler_df['user_id'].unique())
    
    overlap = reranker_users & profiler_users
    
    return {
        'reranker_users': len(reranker_users),
        'profiler_users': len(profiler_users),
        'overlap_users': len(overlap),
        'overlap_fraction': len(overlap) / max(len(reranker_users | profiler_users), 1),
    }
