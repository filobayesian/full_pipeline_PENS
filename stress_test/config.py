"""
Experiment Configuration for Stress Testing

This module defines the experimental grid and model-specific configurations
for stress testing both the reranker and style profiler models.
"""

from typing import List, Dict, Any
from dataclasses import dataclass, field


# =============================================================================
# Experiment Grid Configuration
# =============================================================================

EXPERIMENT_GRID = {
    # Absolute number of users to include in training data
    'user_counts': [1000, 5000, 10000, 50000],
    
    # Maximum history length (number of past interactions)
    'history_lengths': [5, 10, 20, 30, 50],
    
    # Number of random seeds for variance estimation
    'n_seeds': 1,
}


# =============================================================================
# Reranker Configuration
# =============================================================================

RERANKER_CONFIG = {
    # Training parameters (reduced for faster iteration)
    'epochs': 3,
    'batch_size': 256,
    'lr': 1e-3,
    'loss_type': 'listwise',  # 'bpr' or 'listwise'
    'temperature': 1.0,
    
    # Model architecture
    'hidden_dim': 256,
    'dropout': 0.2,
    'use_adaptor': False,
    
    # Data parameters
    'neg_per_pos': 4,  # For BPR loss
    'num_workers': 4,
    
    # Evaluation
    'eval_metrics': ['auc', 'mrr', 'ndcg@5', 'ndcg@10', 'hit@5', 'hit@10'],
}


# =============================================================================
# Profiler Configuration
# =============================================================================

PROFILER_CONFIG = {
    # Feature extraction
    'use_external_sentiment': True,  # Use VADER
    'use_transformer_formality': True,  # RoBERTa formality (higher quality)
    'use_spacy_ner': True,  # spaCy NER for better entity detection
    'use_contrastive': True,  # Contrastive profiling
    
    # Profile quality thresholds
    'min_interactions_for_confidence': 10,
    'min_interactions_for_contrastive': 8,
    
    # Evaluation metrics
    'eval_metrics': [
        'mean_abs_z',           # Mean |z-score| across features
        'n_strong_prefs',       # Features with |z| > 0.5
        'n_very_strong_prefs',  # Features with |z| > 1.0
        'coverage',             # % users with valid profiles
        'alignment_mean',       # Mean alignment score
        'alignment_std',        # Std of alignment scores
    ],
}


# =============================================================================
# Output Configuration
# =============================================================================

OUTPUT_CONFIG = {
    'save_checkpoints': False,  # Save model checkpoints (uses disk space)
    'save_profiles': False,      # Save user profiles parquet
    'save_metrics_json': True,   # Save metrics as JSON
    'log_to_wandb': False,       # Disable wandb for stress tests
}


# =============================================================================
# Helper Classes
# =============================================================================

@dataclass
class ExperimentConfig:
    """Configuration for a single experiment run."""
    user_count: int
    history_length: int
    seed: int
    model: str  # 'reranker' or 'profiler'
    
    # Paths (to be filled in by runner)
    pens_root: str = ''
    output_dir: str = ''
    
    # Model-specific config
    model_config: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def experiment_name(self) -> str:
        """Generate unique experiment name."""
        return f"users_{self.user_count}_history_{self.history_length}_seed_{self.seed}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'user_count': self.user_count,
            'history_length': self.history_length,
            'seed': self.seed,
            'model': self.model,
            'experiment_name': self.experiment_name,
            'pens_root': self.pens_root,
            'output_dir': self.output_dir,
            'model_config': self.model_config,
        }


def generate_experiment_grid(
    models: List[str] = ['reranker', 'profiler'],
    user_counts: List[int] = None,
    history_lengths: List[int] = None,
    n_seeds: int = None,
) -> List[ExperimentConfig]:
    """
    Generate list of experiment configurations from grid.
    
    Args:
        models: List of models to test ('reranker', 'profiler', or both)
        user_counts: Override default user counts
        history_lengths: Override default history lengths
        n_seeds: Override default number of seeds
        
    Returns:
        List of ExperimentConfig objects
    """
    if user_counts is None:
        user_counts = EXPERIMENT_GRID['user_counts']
    if history_lengths is None:
        history_lengths = EXPERIMENT_GRID['history_lengths']
    if n_seeds is None:
        n_seeds = EXPERIMENT_GRID['n_seeds']
    
    experiments = []
    
    for model in models:
        model_config = RERANKER_CONFIG if model == 'reranker' else PROFILER_CONFIG
        
        for user_count in user_counts:
            for hist_len in history_lengths:
                for seed in range(n_seeds):
                    exp = ExperimentConfig(
                        user_count=user_count,
                        history_length=hist_len,
                        seed=seed,
                        model=model,
                        model_config=model_config.copy(),
                    )
                    experiments.append(exp)
    
    return experiments


def count_experiments(
    models: List[str] = ['reranker', 'profiler'],
    user_counts: List[int] = None,
    history_lengths: List[int] = None,
    n_seeds: int = None,
) -> int:
    """Count total number of experiments in grid."""
    if user_counts is None:
        user_counts = EXPERIMENT_GRID['user_counts']
    if history_lengths is None:
        history_lengths = EXPERIMENT_GRID['history_lengths']
    if n_seeds is None:
        n_seeds = EXPERIMENT_GRID['n_seeds']
    
    return len(models) * len(user_counts) * len(history_lengths) * n_seeds
