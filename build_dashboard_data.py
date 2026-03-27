#!/usr/bin/env python3
"""
Preprocess profiling data into lightweight JSON for the surveillance dashboard.

Reads parquet outputs from the style profiler and content profiler,
merges them, computes vulnerability/predictability metrics, samples
500 representative subjects, and outputs JSON files for the dashboard.

Usage:
    python build_dashboard_data.py
    python build_dashboard_data.py --embed   # inline data into dashboard.html
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
log = logging.getLogger(__name__)

# Features to exclude (all zeros — Brysbaert lexicon wasn't loaded)
DEAD_FEATURES = {'mean_concreteness', 'concreteness_coverage'}

# Features used in the radar chart / behavioral fingerprint
DISPLAY_FEATURES = [
    'quote_char_count', 'number_count', 'proper_noun_count',
    'has_question_mark', 'curiosity_pattern_count',
    'sentiment_score', 'formality_score',
    'strong_verb_count', 'weak_verb_count',
    'char_count', 'avg_word_length',
]

FEATURE_LABELS = {
    'quote_char_count': 'QUOTES',
    'number_count': 'NUMBERS',
    'proper_noun_count': 'NAMED ENTITIES',
    'has_question_mark': 'QUESTIONS',
    'curiosity_pattern_count': 'CURIOSITY BAIT',
    'sentiment_score': 'SENTIMENT',
    'formality_score': 'FORMALITY',
    'strong_verb_count': 'STRONG VERBS',
    'weak_verb_count': 'PASSIVE VOICE',
    'char_count': 'HEADLINE LENGTH',
    'avg_word_length': 'WORD COMPLEXITY',
}

FEATURE_DESCRIPTIONS = {
    'quote_char_count': 'Preference for attributed speech and direct quotes',
    'number_count': 'Attraction to specific numbers, statistics, percentages',
    'proper_noun_count': 'Response to named people, places, organizations',
    'has_question_mark': 'Susceptibility to question-framed headlines',
    'curiosity_pattern_count': 'Vulnerability to curiosity gaps (\"secret\", \"revealed\", \"why\")',
    'sentiment_score': 'Emotional valence preference (negative ← 0 → positive)',
    'formality_score': 'Register preference (casual ← 0 → formal)',
    'strong_verb_count': 'Response to aggressive verbs (slam, blast, destroy)',
    'weak_verb_count': 'Tolerance for passive/weak verbs (is, was, gets)',
    'char_count': 'Preferred headline length (short ← → long)',
    'avg_word_length': 'Vocabulary complexity preference (simple ← → complex)',
}


def load_data():
    """Load all source parquet/JSON files."""
    style = pd.read_parquet('style_steering_PENS copy/outputs/user_profiles.parquet')
    content = pd.read_parquet('outputs/content_profiles.parquet')
    pop_stats = pd.read_parquet('style_steering_PENS copy/outputs/population_statistics.parquet')
    exemplars = pd.read_parquet('style_steering_PENS copy/outputs/style_alignment_examples.parquet')
    with open('outputs/content_global_stats.json') as f:
        global_stats = json.load(f)
    log.info(f"Loaded {len(style):,} style profiles, {len(content):,} content profiles")
    return style, content, pop_stats, exemplars, global_stats


def build_population_json(style, content, pop_stats, global_stats):
    """Build population-level aggregate statistics."""
    # Confidence buckets
    conf_high = int((style['confidence'] >= 0.8).sum())
    conf_med = int(((style['confidence'] >= 0.4) & (style['confidence'] < 0.8)).sum())
    conf_low = int((style['confidence'] < 0.4).sum())

    # Population feature stats (excluding dead features)
    features = []
    for _, row in pop_stats.iterrows():
        if row['feature'] not in DEAD_FEATURES:
            features.append({
                'feature': row['feature'],
                'label': FEATURE_LABELS.get(row['feature'], row['feature'].upper()),
                'description': FEATURE_DESCRIPTIONS.get(row['feature'], ''),
                'mean': round(float(row['population_mean']), 4),
                'std': round(float(row['population_std']), 4),
            })

    # Mean |z| per feature across population (how distinctive each feature is)
    z_cols = [f'{f}_z' for f in DISPLAY_FEATURES]
    mean_abs_z = {}
    for f in DISPLAY_FEATURES:
        col = f'{f}_z'
        if col in style.columns:
            mean_abs_z[f] = round(float(style[col].abs().mean()), 3)

    # Category distribution (top 15)
    cat_dist = content['top_category'].value_counts(normalize=True).head(15)

    # Engagement percentiles
    eng = style['history_size']

    return {
        'meta': {
            'n_subjects': int(len(style)),
            'n_interactions': global_stats.get('n_interactions', 0),
            'n_articles': global_stats.get('n_articles', 0),
            'n_categories': int(content['top_category'].nunique()),
            'n_topics': int(content['top_topic'].nunique()),
        },
        'confidence': {
            'high': conf_high,
            'medium': conf_med,
            'low': conf_low,
        },
        'features': features,
        'mean_abs_z': mean_abs_z,
        'category_distribution': {k: round(v, 4) for k, v in cat_dist.items()},
        'engagement_percentiles': {
            f'p{p}': int(eng.quantile(p / 100))
            for p in [10, 25, 50, 75, 90]
        },
        'concentration': {
            'mean': round(float(content['category_concentration'].mean()), 3),
            'median': round(float(content['category_concentration'].median()), 3),
        },
    }


def build_distributions_json(style, content):
    """Pre-bin histograms for z-score features and key content metrics."""
    distributions = {'style_z': {}, 'content': {}}

    # Style z-score histograms
    for feat in DISPLAY_FEATURES:
        col = f'{feat}_z'
        if col not in style.columns:
            continue
        vals = style[col].dropna().values
        # Clip to [-6, 6] for visualization
        vals = np.clip(vals, -6, 6)
        counts, bin_edges = np.histogram(vals, bins=40, range=(-6, 6))
        distributions['style_z'][feat] = {
            'bins': [round(float(b), 2) for b in bin_edges],
            'counts': [int(c) for c in counts],
        }

    # Content distributions
    for col, bins_range in [
        ('category_concentration', (0, 1, 30)),
        ('top_category_share', (0, 1, 30)),
        ('n_categories', (1, 17, 16)),
    ]:
        vals = content[col].dropna().values
        counts, bin_edges = np.histogram(vals, bins=bins_range[2],
                                          range=(bins_range[0], bins_range[1]))
        distributions['content'][col] = {
            'bins': [round(float(b), 3) for b in bin_edges],
            'counts': [int(c) for c in counts],
        }

    # Engagement distribution
    vals = np.clip(style['history_size'].values, 0, 300)
    counts, bin_edges = np.histogram(vals, bins=40, range=(0, 300))
    distributions['content']['history_size'] = {
        'bins': [round(float(b), 1) for b in bin_edges],
        'counts': [int(c) for c in counts],
    }

    return distributions


def sample_subjects(style, content, n=500):
    """Stratified sample of subjects for the dashboard."""
    merged = style.merge(content, on='user_id', how='inner')
    log.info(f"Merged profiles: {len(merged):,} users")

    # Stratified sampling: confidence x interaction quartiles
    merged['conf_bucket'] = pd.cut(merged['confidence'], bins=[0, 0.4, 0.8, 1.01],
                                    labels=['low', 'medium', 'high'], include_lowest=True)
    merged['eng_bucket'] = pd.qcut(merged['history_size'], q=4, labels=['q1', 'q2', 'q3', 'q4'],
                                    duplicates='drop')

    sampled = merged.groupby(['conf_bucket', 'eng_bucket'], observed=True).apply(
        lambda g: g.sample(n=min(len(g), max(1, n // 12)), random_state=42),
        include_groups=False
    ).reset_index(drop=True)

    # Top up if needed
    if len(sampled) < n:
        remaining = merged[~merged['user_id'].isin(sampled['user_id'])]
        extra = remaining.sample(n=min(n - len(sampled), len(remaining)), random_state=42)
        sampled = pd.concat([sampled, extra], ignore_index=True)

    sampled = sampled.head(n)
    log.info(f"Sampled {len(sampled)} subjects")
    return sampled


def build_subjects_json(sampled):
    """Build per-subject dossier records."""
    subjects = []
    for _, row in sampled.iterrows():
        # Z-scores for display features
        style_z = {}
        abs_zs = []
        for feat in DISPLAY_FEATURES:
            col = f'{feat}_z'
            if col in row.index:
                z = round(float(row[col]), 2) if pd.notna(row[col]) else 0
                style_z[feat] = z
                abs_zs.append(abs(z))

        max_abs_z = max(abs_zs) if abs_zs else 0
        mean_abs_z = sum(abs_zs) / len(abs_zs) if abs_zs else 0

        # Exploitability: primarily driven by how distinctive preferences are
        # 60% z-score signal (30% mean distinctiveness + 30% peak deviation)
        # 20% content concentration, 20% profile confidence
        confidence = float(row['confidence']) if pd.notna(row['confidence']) else 0
        concentration = float(row['category_concentration']) if pd.notna(row['category_concentration']) else 0
        vulnerability = round(np.clip(
            0.3 * min(mean_abs_z / 4, 1) + 0.3 * min(max_abs_z / 8, 1)
            + 0.2 * concentration + 0.2 * confidence,
            0, 1
        ), 3)

        # Predictability
        top_cat_share = float(row['top_category_share']) if pd.notna(row['top_category_share']) else 0
        predictability = round(np.clip(
            0.5 * top_cat_share + 0.5 * concentration, 0, 1
        ), 3)

        # Parse category distribution (top 5)
        try:
            cat_dist = json.loads(row['category_distribution'])
            cat_dist = dict(sorted(cat_dist.items(), key=lambda x: -x[1])[:5])
            cat_dist = {k: round(v, 3) for k, v in cat_dist.items()}
        except (json.JSONDecodeError, TypeError):
            cat_dist = {}

        # Parse entities (top 5)
        try:
            entities = json.loads(row['top_entities'])
            entities = list(dict(sorted(entities.items(), key=lambda x: -x[1])[:5]).keys())
        except (json.JSONDecodeError, TypeError):
            entities = []

        # Parse profile card
        try:
            profile_card = json.loads(row['profile_card_json'])
        except (json.JSONDecodeError, TypeError):
            profile_card = {'top_positive': [], 'top_negative': []}

        subjects.append({
            'id': row['user_id'],
            'interactions': int(row['history_size']),
            'confidence': round(confidence, 2),
            'confidence_label': 'HIGH' if confidence >= 0.8 else ('MEDIUM' if confidence >= 0.4 else 'LOW'),
            'top_category': row['top_category'],
            'top_category_share': round(top_cat_share, 3),
            'top_dwell_category': row.get('top_dwell_category', row['top_category']),
            'n_categories': int(row['n_categories']),
            'category_concentration': round(concentration, 3),
            'top_topic': row['top_topic'],
            'n_topics': int(row['n_topics']),
            'entities': entities,
            'n_entities': int(row['n_unique_entities']),
            'category_distribution': cat_dist,
            'style_z': style_z,
            'vulnerability': vulnerability,
            'predictability': predictability,
            'profile_card': profile_card,
        })

    return subjects


def build_exemplars_json(exemplars):
    """Format alignment examples for the dashboard."""
    records = []
    for _, row in exemplars.iterrows():
        records.append({
            'user_id': row['user_id'],
            'headline': row['headline'],
            'alignment_score': round(float(row['alignment_score']), 3),
            'category': row['category'],
            'dwell_time': round(float(row['dwell_time']), 1) if pd.notna(row['dwell_time']) else None,
        })
    return records


def embed_into_html(data_dir, html_path):
    """Inline JSON data files into the dashboard HTML as script tags."""
    html = Path(html_path).read_text()

    data_files = {
        'DATA_POPULATION': 'population.json',
        'DATA_DISTRIBUTIONS': 'distributions.json',
        'DATA_SUBJECTS': 'subjects.json',
        'DATA_EXEMPLARS': 'exemplars.json',
    }

    inject_lines = []
    for var_name, filename in data_files.items():
        fpath = data_dir / filename
        if fpath.exists():
            data = fpath.read_text()
            inject_lines.append(f'window.{var_name} = {data};')

    inject_block = '<script>\n' + '\n'.join(inject_lines) + '\n</script>'

    # Replace the placeholder in the HTML
    if '<!-- EMBEDDED_DATA -->' in html:
        html = html.replace('<!-- EMBEDDED_DATA -->', inject_block)
    else:
        # Insert before closing </head>
        html = html.replace('</head>', inject_block + '\n</head>')

    Path(html_path).write_text(html)
    log.info(f"Embedded data into {html_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--embed', action='store_true',
                        help='Inline JSON into dashboard.html')
    parser.add_argument('--html', default='dashboard.html',
                        help='Path to dashboard HTML file')
    parser.add_argument('--out', default='dashboard_data',
                        help='Output directory for JSON files')
    parser.add_argument('--n_subjects', type=int, default=500)
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # Load
    style, content, pop_stats, exemplars, global_stats = load_data()

    # Build JSONs
    log.info("Building population.json...")
    pop = build_population_json(style, content, pop_stats, global_stats)
    with open(out / 'population.json', 'w') as f:
        json.dump(pop, f, separators=(',', ':'))

    log.info("Building distributions.json...")
    dist = build_distributions_json(style, content)
    with open(out / 'distributions.json', 'w') as f:
        json.dump(dist, f, separators=(',', ':'))

    log.info("Building subjects.json...")
    sampled = sample_subjects(style, content, n=args.n_subjects)
    subjects = build_subjects_json(sampled)
    with open(out / 'subjects.json', 'w') as f:
        json.dump(subjects, f, separators=(',', ':'))

    log.info("Building exemplars.json...")
    exemp = build_exemplars_json(exemplars)
    with open(out / 'exemplars.json', 'w') as f:
        json.dump(exemp, f, separators=(',', ':'))

    # Report sizes
    total = 0
    for fpath in out.glob('*.json'):
        size = fpath.stat().st_size
        total += size
        log.info(f"  {fpath.name}: {size / 1024:.1f} KB")
    log.info(f"  TOTAL: {total / 1024:.1f} KB")

    # Optionally embed
    if args.embed:
        embed_into_html(out, args.html)


if __name__ == '__main__':
    main()
