"""
Reranker Adapter for Stress Testing

This module wraps the reranker training and evaluation for programmatic use
in the stress testing framework.

Key functions:
- train_reranker: Train model with given configuration
- evaluate_reranker: Evaluate model on validation data
- run_reranker_experiment: Complete train + eval pipeline
"""

import json
import logging
import sys
import tempfile
from pathlib import Path
from typing import Dict, Any, List, Optional
import torch
import numpy as np

# Add reranker source to path
reranker_path = Path(__file__).parent.parent / 'refactor_PENS copy'
if str(reranker_path) not in sys.path:
    sys.path.insert(0, str(reranker_path))

from src.model.reranker import NewsReranker
from src.data.dataset import (
    ImpressionPairDataset,
    ImpressionDataset,
    create_collate_fn,
    create_collate_fn_listwise
)
from src.utils.metrics import evaluate_impression
from src.utils.io import read_jsonl

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def train_reranker(
    train_impressions: List[Dict[str, Any]],
    valid_impressions: List[Dict[str, Any]],
    embeddings_cache: Dict[str, Any],
    config: Dict[str, Any],
    device: str = 'cpu',
) -> Dict[str, Any]:
    """
    Train reranker model with given data and configuration.
    
    Args:
        train_impressions: Training impression data
        valid_impressions: Validation impression data
        embeddings_cache: Pre-computed embeddings
        config: Training configuration
        device: Device to use ('cpu', 'cuda', 'mps')
        
    Returns:
        Dictionary with trained model and training metrics
    """
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    from tqdm import tqdm
    
    # Extract config
    epochs = config.get('epochs', 3)
    batch_size = config.get('batch_size', 256)
    lr = config.get('lr', 1e-3)
    loss_type = config.get('loss_type', 'listwise')
    temperature = config.get('temperature', 1.0)
    hidden_dim = config.get('hidden_dim', 256)
    dropout = config.get('dropout', 0.2)
    use_adaptor = config.get('use_adaptor', False)
    neg_per_pos = config.get('neg_per_pos', 4)
    max_history_len = config.get('max_history_len', 50)
    num_workers = config.get('num_workers', 0)  # Default 0 for safety
    
    embed_dim = embeddings_cache['metadata']['embed_dim']
    
    logger.info(f"Training reranker: {len(train_impressions)} train, {len(valid_impressions)} valid")
    logger.info(f"Config: epochs={epochs}, batch_size={batch_size}, loss_type={loss_type}")
    
    # Create datasets
    if loss_type == 'bpr':
        train_dataset = ImpressionPairDataset(
            train_impressions,
            embeddings_cache,
            neg_per_pos=neg_per_pos,
            max_history_len=max_history_len
        )
        valid_dataset = ImpressionPairDataset(
            valid_impressions,
            embeddings_cache,
            neg_per_pos=neg_per_pos,
            max_history_len=max_history_len
        )
        collate = create_collate_fn(embeddings_cache)
    else:  # listwise
        train_dataset = ImpressionDataset(
            train_impressions,
            embeddings_cache,
            max_history_len=max_history_len
        )
        valid_dataset = ImpressionDataset(
            valid_impressions,
            embeddings_cache,
            max_history_len=max_history_len
        )
        collate = create_collate_fn_listwise(embeddings_cache, max_history_len)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate,
    )
    
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate,
    )
    
    # Initialize model
    model = NewsReranker(
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        dropout=dropout,
        use_adaptor=use_adaptor,
    )
    model = model.to(device)
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Training loop
    train_losses = []
    val_losses = []
    val_accs = []
    
    for epoch in range(1, epochs + 1):
        # Train epoch
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}", leave=False):
            if loss_type == 'bpr':
                history_embs = batch['history_embs'].to(device)
                history_mask = batch['history_mask'].to(device)
                pos_emb = batch['pos_emb'].to(device)
                neg_emb = batch['neg_emb'].to(device)
                
                pos_scores = model(history_embs, history_mask, pos_emb)
                neg_scores = model(history_embs, history_mask, neg_emb)
                
                loss = F.softplus(-(pos_scores - neg_scores)).mean()
            else:  # listwise
                history_embs = batch['history_embs'].to(device)
                history_mask = batch['history_mask'].to(device)
                cand_embs = batch['cand_embs'].to(device)
                cand_mask = batch['cand_mask'].to(device)
                labels = batch['labels'].to(device)
                
                # Filter valid impressions
                valid_counts = cand_mask.sum(dim=1)
                pos_counts = (labels.bool() & cand_mask).sum(dim=1)
                valid_impressions_mask = (pos_counts > 0) & (valid_counts > 1)
                
                if not valid_impressions_mask.any():
                    continue
                
                history_embs = history_embs[valid_impressions_mask]
                history_mask = history_mask[valid_impressions_mask]
                cand_embs = cand_embs[valid_impressions_mask]
                cand_mask = cand_mask[valid_impressions_mask]
                labels = labels[valid_impressions_mask]
                
                user_vecs = model.user_encoder(history_embs, history_mask)
                scores = model.score_user_candidates(user_vecs, cand_embs)
                
                logits = scores / temperature
                logits = logits.masked_fill(~cand_mask, -1e9)
                
                den = torch.logsumexp(logits, dim=1)
                num = torch.logsumexp(logits.masked_fill(labels == 0, -1e9), dim=1)
                loss = (den - num).mean()
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            n_batches += 1
        
        avg_train_loss = epoch_loss / max(n_batches, 1)
        train_losses.append(avg_train_loss)
        
        # Validate
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        n_val_batches = 0
        
        with torch.no_grad():
            for batch in valid_loader:
                if loss_type == 'bpr':
                    history_embs = batch['history_embs'].to(device)
                    history_mask = batch['history_mask'].to(device)
                    pos_emb = batch['pos_emb'].to(device)
                    neg_emb = batch['neg_emb'].to(device)
                    
                    pos_scores = model(history_embs, history_mask, pos_emb)
                    neg_scores = model(history_embs, history_mask, neg_emb)
                    
                    loss = F.softplus(-(pos_scores - neg_scores)).mean()
                    val_correct += (pos_scores > neg_scores).sum().item()
                    val_total += pos_scores.size(0)
                else:
                    history_embs = batch['history_embs'].to(device)
                    history_mask = batch['history_mask'].to(device)
                    cand_embs = batch['cand_embs'].to(device)
                    cand_mask = batch['cand_mask'].to(device)
                    labels = batch['labels'].to(device)
                    
                    valid_counts = cand_mask.sum(dim=1)
                    pos_counts = (labels.bool() & cand_mask).sum(dim=1)
                    valid_impressions_mask = (pos_counts > 0) & (valid_counts > 1)
                    
                    if not valid_impressions_mask.any():
                        continue
                    
                    history_embs = history_embs[valid_impressions_mask]
                    history_mask = history_mask[valid_impressions_mask]
                    cand_embs = cand_embs[valid_impressions_mask]
                    cand_mask = cand_mask[valid_impressions_mask]
                    labels = labels[valid_impressions_mask]
                    
                    user_vecs = model.user_encoder(history_embs, history_mask)
                    scores = model.score_user_candidates(user_vecs, cand_embs)
                    
                    logits = scores / temperature
                    logits = logits.masked_fill(~cand_mask, -1e9)
                    
                    den = torch.logsumexp(logits, dim=1)
                    num = torch.logsumexp(logits.masked_fill(labels == 0, -1e9), dim=1)
                    loss = (den - num).mean()
                    
                    # Top-1 accuracy
                    scores_masked = scores.masked_fill(~cand_mask, -1e9)
                    pred_idx = scores_masked.argmax(dim=1)
                    pred_labels = labels.gather(1, pred_idx.unsqueeze(1)).squeeze(1)
                    val_correct += (pred_labels == 1).sum().item()
                    val_total += labels.size(0)
                
                val_loss += loss.item()
                n_val_batches += 1
        
        avg_val_loss = val_loss / max(n_val_batches, 1)
        val_acc = val_correct / max(val_total, 1)
        val_losses.append(avg_val_loss)
        val_accs.append(val_acc)
        
        logger.info(
            f"Epoch {epoch}: train_loss={avg_train_loss:.4f}, "
            f"val_loss={avg_val_loss:.4f}, val_acc={val_acc:.4f}"
        )
    
    return {
        'model': model,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_accs': val_accs,
        'final_val_loss': val_losses[-1] if val_losses else None,
        'final_val_acc': val_accs[-1] if val_accs else None,
    }


def evaluate_reranker(
    model: NewsReranker,
    impressions: List[Dict[str, Any]],
    embeddings_cache: Dict[str, Any],
    device: str = 'cpu',
) -> Dict[str, float]:
    """
    Evaluate reranker model with full metrics.
    
    Args:
        model: Trained model
        impressions: Evaluation impressions
        embeddings_cache: Pre-computed embeddings
        device: Device to use
        
    Returns:
        Dictionary with evaluation metrics
    """
    from tqdm import tqdm
    
    model.eval()
    embeddings = embeddings_cache['embeddings']
    id_to_idx = embeddings_cache['id_to_idx']
    
    all_metrics = []
    skipped = 0
    
    with torch.no_grad():
        for impression in tqdm(impressions, desc="Evaluating", leave=False):
            history_ids = impression.get('history', [])
            pos_ids = impression.get('pos', [])
            neg_ids = impression.get('neg', [])
            
            candidate_ids = pos_ids + neg_ids
            if len(candidate_ids) == 0:
                skipped += 1
                continue
            
            valid_candidates = [cid for cid in candidate_ids if cid in id_to_idx]
            if len(valid_candidates) == 0:
                skipped += 1
                continue
            
            labels = np.array(
                [1] * len(pos_ids) + [0] * len(neg_ids)
            )[:len(valid_candidates)]
            
            # Get history embeddings
            history_embs = []
            for hid in history_ids:
                if hid in id_to_idx:
                    idx = id_to_idx[hid]
                    history_embs.append(embeddings[idx])
            
            if len(history_embs) == 0:
                history_embs = torch.zeros(1, embeddings.shape[1])
                history_mask = torch.zeros(1)
            else:
                history_embs = torch.stack(history_embs).unsqueeze(0)
                history_mask = torch.ones(1, len(history_embs[0]))
            
            # Get candidate embeddings
            candidate_embs = []
            for cid in valid_candidates:
                idx = id_to_idx[cid]
                candidate_embs.append(embeddings[idx])
            
            candidate_embs = torch.stack(candidate_embs).unsqueeze(0)
            
            # Move to device
            history_embs = history_embs.to(device)
            history_mask = history_mask.to(device)
            candidate_embs = candidate_embs.to(device)
            
            # Score
            scores = model.score_candidates(history_embs, history_mask, candidate_embs)
            scores = scores.squeeze(0).cpu().numpy()
            
            # Evaluate
            metrics = evaluate_impression(labels, scores)
            all_metrics.append(metrics)
    
    if skipped > 0:
        logger.warning(f"Skipped {skipped} impressions due to missing data")
    
    # Aggregate
    aggregated = {}
    if all_metrics:
        for key in all_metrics[0].keys():
            values = [m[key] for m in all_metrics if m[key] is not None]
            aggregated[key] = np.mean(values) if values else 0.0
    
    return aggregated


def run_reranker_experiment(
    train_impressions: List[Dict[str, Any]],
    valid_impressions: List[Dict[str, Any]],
    embeddings_cache: Dict[str, Any],
    config: Dict[str, Any],
    device: str = 'cpu',
) -> Dict[str, Any]:
    """
    Run complete reranker experiment (train + evaluate).
    
    Args:
        train_impressions: Training data
        valid_impressions: Validation data
        embeddings_cache: Pre-computed embeddings
        config: Training configuration
        device: Device to use
        
    Returns:
        Dictionary with all results
    """
    # Train
    train_result = train_reranker(
        train_impressions,
        valid_impressions,
        embeddings_cache,
        config,
        device
    )
    
    # Evaluate
    eval_metrics = evaluate_reranker(
        train_result['model'],
        valid_impressions,
        embeddings_cache,
        device
    )
    
    return {
        'train_losses': train_result['train_losses'],
        'val_losses': train_result['val_losses'],
        'val_accs': train_result['val_accs'],
        'final_val_loss': train_result['final_val_loss'],
        'final_val_acc': train_result['final_val_acc'],
        'eval_metrics': eval_metrics,
    }


def load_embeddings_cache(embeddings_path: str) -> Dict[str, Any]:
    """
    Load pre-computed embeddings cache.
    
    Args:
        embeddings_path: Path to embeddings .pt file
        
    Returns:
        Embeddings cache dictionary
    """
    logger.info(f"Loading embeddings from {embeddings_path}")
    cache = torch.load(embeddings_path, map_location='cpu')
    logger.info(f"Loaded embeddings: {cache['metadata']}")
    return cache
