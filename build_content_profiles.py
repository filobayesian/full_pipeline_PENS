#!/usr/bin/env python3
"""
Build per-user content preference profiles from PENS click history.

Joins user interactions with news metadata (category, topic, entities)
to produce interpretable content profiles for each user.

Usage:
    python build_content_profiles.py --pens_root data/PENS --out outputs/
    python build_content_profiles.py --pens_root data/PENS --out outputs/ --split train
"""

import argparse
import json
import logging
import ast
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_news_metadata(pens_root: Path) -> pd.DataFrame:
    """Load news.tsv with category, topic, and entity metadata."""
    news_df = pd.read_csv(
        pens_root / "news.tsv",
        sep='\t',
        names=['news_id', 'category', 'topic', 'headline', 'body',
               'title_entity', 'entity_content'],
        dtype={'news_id': str, 'category': str, 'topic': str, 'headline': str}
    )
    # Drop body to save memory
    news_df = news_df.drop(columns=['body'])
    logger.info(f"Loaded {len(news_df):,} news articles")
    logger.info(f"  Categories: {news_df['category'].nunique()} unique")
    logger.info(f"  Topics: {news_df['topic'].nunique()} unique")
    return news_df


def parse_entities(title_entity_str: str) -> list[str]:
    """Extract entity names from the title_entity JSON-like field."""
    if pd.isna(title_entity_str) or not title_entity_str.strip():
        return []
    try:
        parsed = ast.literal_eval(title_entity_str)
        if isinstance(parsed, dict):
            return list(parsed.values())
        return []
    except (ValueError, SyntaxError):
        return []


def load_interactions(pens_root: Path, split: str) -> pd.DataFrame:
    """Load user interactions and expand to one row per (user, news_id)."""
    split_file = pens_root / f"{split}.tsv"
    logger.info(f"Loading {split} split from {split_file}")

    df = pd.read_csv(split_file, sep='\t', header=0)
    logger.info(f"Loaded {len(df):,} user records")

    rows = []
    for _, row in df.iterrows():
        user_id = row['UserID']
        news_ids = str(row['ClicknewsID']).split()
        dwell_times_str = str(row['dwelltime']).split()

        dwell_times = []
        for dt in dwell_times_str:
            try:
                dwell_times.append(float(dt))
            except (ValueError, TypeError):
                dwell_times.append(np.nan)

        for i, nid in enumerate(news_ids):
            rows.append({
                'user_id': user_id,
                'news_id': nid,
                'dwell_time': dwell_times[i] if i < len(dwell_times) else np.nan,
            })

    interactions = pd.DataFrame(rows)
    logger.info(f"Expanded to {len(interactions):,} interactions "
                f"({interactions['user_id'].nunique():,} users)")
    return interactions


def build_content_profiles(
    interactions: pd.DataFrame,
    news_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build per-user content preference profiles.

    Returns:
        user_content_profiles: One row per user with preference distributions
        user_interactions: Enriched interactions with news metadata
    """
    # Join interactions with news metadata
    enriched = interactions.merge(
        news_df[['news_id', 'category', 'topic', 'headline', 'title_entity']],
        on='news_id',
        how='left'
    )
    enriched = enriched.dropna(subset=['category'])
    logger.info(f"Enriched interactions: {len(enriched):,} rows")

    # Parse entities
    logger.info("Parsing entities...")
    enriched['entities'] = enriched['title_entity'].apply(parse_entities)

    # --- Per-user profiles ---
    logger.info("Building per-user content profiles...")
    profiles = []

    # Global category distribution (for computing over/under-representation)
    global_cat_dist = enriched['category'].value_counts(normalize=True)
    global_topic_dist = enriched['topic'].value_counts(normalize=True)

    for user_id, group in enriched.groupby('user_id'):
        n_interactions = len(group)

        # Category distribution
        cat_counts = group['category'].value_counts()
        cat_dist = cat_counts / cat_counts.sum()

        # Topic distribution (top 10)
        topic_counts = group['topic'].value_counts()
        topic_dist = topic_counts / topic_counts.sum()

        # Entity frequency
        all_entities = [e for elist in group['entities'] for e in elist]
        entity_counts = pd.Series(all_entities).value_counts() if all_entities else pd.Series(dtype=int)

        # Category over/under-representation vs population
        cat_lift = {}
        for cat in cat_dist.index:
            global_rate = global_cat_dist.get(cat, 0.001)
            cat_lift[cat] = cat_dist[cat] / global_rate

        # Top category preference (highest lift)
        top_cat = max(cat_lift, key=cat_lift.get) if cat_lift else None
        top_cat_lift = cat_lift.get(top_cat, 0) if top_cat else 0

        # Concentration: how focused is this user? (entropy-based)
        cat_entropy = -(cat_dist * np.log2(cat_dist.clip(lower=1e-10))).sum()
        max_entropy = np.log2(len(global_cat_dist))
        concentration = 1.0 - (cat_entropy / max_entropy) if max_entropy > 0 else 0

        # Dwell-weighted category preference
        if group['dwell_time'].notna().any():
            dwell_weighted = group.groupby('category')['dwell_time'].sum()
            dwell_weighted = dwell_weighted / dwell_weighted.sum()
            top_dwell_cat = dwell_weighted.idxmax()
        else:
            top_dwell_cat = top_cat

        profiles.append({
            'user_id': user_id,
            'n_interactions': n_interactions,
            # Category preferences
            'category_distribution': json.dumps(cat_dist.to_dict()),
            'top_category': cat_dist.idxmax(),
            'top_category_share': cat_dist.max(),
            'top_category_lift': top_cat_lift,
            'top_dwell_category': top_dwell_cat,
            'n_categories': len(cat_counts),
            'category_concentration': concentration,
            # Topic preferences
            'topic_distribution': json.dumps(topic_dist.head(10).to_dict()),
            'top_topic': topic_dist.idxmax(),
            'top_topic_share': topic_dist.max(),
            'n_topics': len(topic_counts),
            # Entity preferences
            'top_entities': json.dumps(entity_counts.head(10).to_dict()) if len(entity_counts) > 0 else '{}',
            'n_unique_entities': len(entity_counts),
            # Category lift vs population (JSON for all categories)
            'category_lift': json.dumps({k: round(v, 3) for k, v in cat_lift.items()}),
        })

    user_profiles = pd.DataFrame(profiles)
    logger.info(f"Built {len(user_profiles):,} user content profiles")

    return user_profiles, enriched


def main():
    parser = argparse.ArgumentParser(
        description='Build per-user content preference profiles from PENS data'
    )
    parser.add_argument('--pens_root', type=str, default='data/PENS')
    parser.add_argument('--out', type=str, default='outputs')
    parser.add_argument('--split', type=str, default='train',
                        choices=['train', 'valid'])
    parser.add_argument('--sample', type=int, default=None,
                        help='Sample N users for testing')
    args = parser.parse_args()

    pens_root = Path(args.pens_root)
    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    news_df = load_news_metadata(pens_root)
    interactions = load_interactions(pens_root, args.split)

    if args.sample:
        sampled = interactions['user_id'].unique()[:args.sample]
        interactions = interactions[interactions['user_id'].isin(sampled)]
        logger.info(f"Sampled to {len(sampled):,} users")

    # Build profiles
    user_profiles, enriched = build_content_profiles(interactions, news_df)

    # Save
    user_profiles.to_parquet(output_dir / "content_profiles.parquet", index=False)
    logger.info(f"Saved: {output_dir / 'content_profiles.parquet'}")

    # Save global distributions for the visualization
    global_stats = {
        'category_distribution': news_df['category'].value_counts(normalize=True).to_dict(),
        'topic_distribution': news_df['topic'].value_counts(normalize=True).head(50).to_dict(),
        'n_users': int(interactions['user_id'].nunique()),
        'n_interactions': int(len(interactions)),
        'n_articles': int(len(news_df)),
    }
    with open(output_dir / "content_global_stats.json", "w") as f:
        json.dump(global_stats, f, indent=2)
    logger.info(f"Saved: {output_dir / 'content_global_stats.json'}")

    # Print summary
    print("\n" + "=" * 60)
    print("CONTENT PROFILES SUMMARY")
    print("=" * 60)
    print(f"Users profiled: {len(user_profiles):,}")
    print(f"Total interactions: {len(enriched):,}")
    print(f"\nTop categories (by user preference):")
    top_cats = user_profiles['top_category'].value_counts().head(10)
    for cat, count in top_cats.items():
        print(f"  {cat}: {count:,} users ({count/len(user_profiles)*100:.1f}%)")
    print(f"\nMean category concentration: {user_profiles['category_concentration'].mean():.3f}")
    print(f"Mean categories per user: {user_profiles['n_categories'].mean():.1f}")
    print(f"Mean topics per user: {user_profiles['n_topics'].mean():.1f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
