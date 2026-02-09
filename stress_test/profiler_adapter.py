"""
Style Profiler Adapter for Stress Testing

This module wraps the headline style profiler for programmatic use
in the stress testing framework.

Key functions:
- build_profiles: Build user profiles from interaction data
- compute_profile_metrics: Compute profile quality metrics
- run_profiler_experiment: Complete profiling pipeline
"""

import logging
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np

# Add style_steering source to path
style_steering_path = Path(__file__).parent.parent / 'style_steering_PENS copy'
if str(style_steering_path) not in sys.path:
    sys.path.insert(0, str(style_steering_path))

from headline_style import config_v2 as config
from headline_style.profiling_v2 import (
    build_user_profiles,
    compute_alignment_scores,
)
from headline_style.metrics_v2 import get_feature_names

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def build_profiles(
    df: pd.DataFrame,
    profiler_config: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Build user profiles from interaction data.
    
    Args:
        df: DataFrame with canonical schema (user_id, headline, dwell_time, clicked)
        profiler_config: Profiler configuration
        
    Returns:
        Dictionary with profiles and features DataFrames
    """
    use_external_sentiment = profiler_config.get('use_external_sentiment', True)
    use_transformer_formality = profiler_config.get('use_transformer_formality', False)
    use_contrastive = profiler_config.get('use_contrastive', True)
    use_spacy_ner = profiler_config.get('use_spacy_ner', False)
    
    logger.info(f"Building profiles for {df['user_id'].nunique()} users, {len(df)} interactions")
    logger.info(f"Config: contrastive={use_contrastive}, vader={use_external_sentiment}")
    
    try:
        headline_features_df, user_profiles_df = build_user_profiles(
            df,
            use_external_sentiment=use_external_sentiment,
            use_transformer_formality=use_transformer_formality,
            use_contrastive=use_contrastive,
            use_spacy_ner=use_spacy_ner
        )
        
        return {
            'headline_features': headline_features_df,
            'user_profiles': user_profiles_df,
            'success': True,
            'error': None,
        }
    except Exception as e:
        logger.error(f"Failed to build profiles: {e}")
        return {
            'headline_features': None,
            'user_profiles': None,
            'success': False,
            'error': str(e),
        }


def compute_alignment_metrics(
    headline_features_df: pd.DataFrame,
    user_profiles_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute alignment scores for all interactions.
    
    Args:
        headline_features_df: Features for each (user, headline) interaction
        user_profiles_df: User profiles
        
    Returns:
        DataFrame with alignment scores added
    """
    feature_cols = get_feature_names()
    population_mean = headline_features_df[feature_cols].mean()
    population_std = headline_features_df[feature_cols].std()
    population_std = population_std.replace(0, 1e-6)
    
    headline_features_df = compute_alignment_scores(
        headline_features_df,
        user_profiles_df,
        population_mean,
        population_std
    )
    
    return headline_features_df


def compute_profile_metrics(
    user_profiles_df: pd.DataFrame,
    headline_features_df: Optional[pd.DataFrame] = None,
    profiler_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, float]:
    """
    Compute profile quality metrics for stress testing.
    
    Args:
        user_profiles_df: User profiles DataFrame
        headline_features_df: Optional features DataFrame with alignment scores
        profiler_config: Optional profiler configuration
        
    Returns:
        Dictionary of metrics
    """
    if profiler_config is None:
        profiler_config = {}
    
    min_interactions = profiler_config.get('min_interactions_for_confidence', 10)
    
    n_users = len(user_profiles_df)
    
    if n_users == 0:
        return {
            'n_users': 0,
            'coverage': 0.0,
            'mean_abs_z': 0.0,
            'n_strong_prefs': 0.0,
            'n_very_strong_prefs': 0.0,
            'mean_history_size': 0.0,
            'alignment_mean': 0.0,
            'alignment_std': 0.0,
        }
    
    # Get z-score columns
    z_cols = [col for col in user_profiles_df.columns if col.endswith('_z')]
    
    # Mean absolute z-score across all users and features
    if z_cols:
        all_z_values = user_profiles_df[z_cols].values.flatten()
        mean_abs_z = np.abs(all_z_values[~np.isnan(all_z_values)]).mean()
        
        # Count strong preferences per user
        strong_prefs_per_user = (user_profiles_df[z_cols].abs() > 0.5).sum(axis=1).mean()
        very_strong_prefs_per_user = (user_profiles_df[z_cols].abs() > 1.0).sum(axis=1).mean()
    else:
        mean_abs_z = 0.0
        strong_prefs_per_user = 0.0
        very_strong_prefs_per_user = 0.0
    
    # Coverage: fraction of users with enough interactions
    if 'history_size' in user_profiles_df.columns:
        n_confident = (user_profiles_df['history_size'] >= min_interactions).sum()
        coverage = n_confident / n_users
        mean_history_size = user_profiles_df['history_size'].mean()
    else:
        coverage = 1.0
        mean_history_size = 0.0
    
    # Alignment metrics (if available)
    alignment_mean = 0.0
    alignment_std = 0.0
    if headline_features_df is not None and 'alignment_score' in headline_features_df.columns:
        alignment_scores = headline_features_df['alignment_score'].dropna()
        if len(alignment_scores) > 0:
            alignment_mean = alignment_scores.mean()
            alignment_std = alignment_scores.std()
    
    return {
        'n_users': n_users,
        'coverage': coverage,
        'mean_abs_z': mean_abs_z,
        'n_strong_prefs': strong_prefs_per_user,
        'n_very_strong_prefs': very_strong_prefs_per_user,
        'mean_history_size': mean_history_size,
        'alignment_mean': alignment_mean,
        'alignment_std': alignment_std,
    }


def analyze_profile_distribution(
    user_profiles_df: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Analyze distribution of profile characteristics.
    
    Args:
        user_profiles_df: User profiles DataFrame
        
    Returns:
        Dictionary with distribution statistics
    """
    z_cols = [col for col in user_profiles_df.columns if col.endswith('_z')]
    
    if not z_cols or len(user_profiles_df) == 0:
        return {'error': 'No z-score columns or empty profiles'}
    
    # Per-feature statistics
    feature_stats = {}
    for col in z_cols:
        feature_name = col.replace('_z', '')
        values = user_profiles_df[col].dropna()
        if len(values) > 0:
            feature_stats[feature_name] = {
                'mean': values.mean(),
                'std': values.std(),
                'min': values.min(),
                'max': values.max(),
                'n_strong_pos': (values > 0.5).sum(),
                'n_strong_neg': (values < -0.5).sum(),
            }
    
    # History size distribution
    history_stats = {}
    if 'history_size' in user_profiles_df.columns:
        hs = user_profiles_df['history_size']
        history_stats = {
            'mean': hs.mean(),
            'median': hs.median(),
            'min': hs.min(),
            'max': hs.max(),
            'std': hs.std(),
        }
    
    return {
        'feature_stats': feature_stats,
        'history_stats': history_stats,
        'n_features': len(z_cols),
        'n_users': len(user_profiles_df),
    }


def run_profiler_experiment(
    df: pd.DataFrame,
    profiler_config: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Run complete profiler experiment.
    
    Args:
        df: DataFrame with canonical schema
        profiler_config: Profiler configuration
        
    Returns:
        Dictionary with all results
    """
    # Build profiles
    build_result = build_profiles(df, profiler_config)
    
    if not build_result['success']:
        return {
            'success': False,
            'error': build_result['error'],
            'metrics': None,
        }
    
    headline_features_df = build_result['headline_features']
    user_profiles_df = build_result['user_profiles']
    
    # Compute alignment scores
    try:
        headline_features_df = compute_alignment_metrics(
            headline_features_df,
            user_profiles_df
        )
    except Exception as e:
        logger.warning(f"Failed to compute alignment: {e}")
    
    # Compute metrics
    metrics = compute_profile_metrics(
        user_profiles_df,
        headline_features_df,
        profiler_config
    )
    
    # Additional analysis
    distribution = analyze_profile_distribution(user_profiles_df)
    
    return {
        'success': True,
        'error': None,
        'metrics': metrics,
        'distribution': distribution,
        'n_interactions': len(df),
        'n_users': df['user_id'].nunique(),
    }


def get_top_features_for_user(
    user_id: str,
    user_profiles_df: pd.DataFrame,
    n_top: int = 5,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Get top positive and negative features for a user.
    
    Args:
        user_id: User ID
        user_profiles_df: User profiles DataFrame
        n_top: Number of top features to return
        
    Returns:
        Dictionary with top positive and negative features
    """
    user_row = user_profiles_df[user_profiles_df['user_id'] == user_id]
    
    if len(user_row) == 0:
        return {'error': f'User {user_id} not found'}
    
    user_row = user_row.iloc[0]
    z_cols = [col for col in user_profiles_df.columns if col.endswith('_z')]
    
    z_values = [(col.replace('_z', ''), user_row[col]) for col in z_cols]
    z_values = [(f, z) for f, z in z_values if not np.isnan(z)]
    z_values.sort(key=lambda x: abs(x[1]), reverse=True)
    
    top_positive = [
        {'feature': f, 'z_score': z}
        for f, z in z_values if z > 0
    ][:n_top]
    
    top_negative = [
        {'feature': f, 'z_score': z}
        for f, z in z_values if z < 0
    ][:n_top]
    
    return {
        'top_positive': top_positive,
        'top_negative': top_negative,
        'user_id': user_id,
    }
