"""Phase Magic-1 / Track 3.3: M3 Geological epoch cluster + lookup.

Plan: docs/research/2026-05-13-magic-features-brainstorm.dense.md §I M3.

Hypothesis: ROGII wells は same geological epoch (= same age formation) が **3D 空間
cluster** している。 (X, Y) 重心 + GR signature で hierarchical clustering、
test wells を「nearest cluster の wells」 にマッピング、 同 cluster の TVT pattern を
transfer。

期待 lift: -0.05 〜 -0.20 ft (= test wells と「地質的同 epoch」 train wells を特定、
LGB に強い prior 提供)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler


def extract_well_signatures(full_df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-well geological + GR signature features.

    Returns one row per well with columns:
        ['well', 'x_centroid', 'y_centroid', 'z_min', 'z_max',
         'gr_mean', 'gr_std', 'gr_min', 'gr_max',
         'md_min', 'md_max', 'n_rows']
    """
    required = {'well', 'X', 'Y', 'Z', 'GR', 'MD'}
    missing = required - set(full_df.columns)
    if missing:
        raise ValueError(f"missing columns: {missing}")

    agg = (
        full_df.groupby('well')
        .agg(
            x_centroid=('X', 'mean'),
            y_centroid=('Y', 'mean'),
            z_min=('Z', 'min'),
            z_max=('Z', 'max'),
            gr_mean=('GR', 'mean'),
            gr_std=('GR', 'std'),
            gr_min=('GR', 'min'),
            gr_max=('GR', 'max'),
            md_min=('MD', 'min'),
            md_max=('MD', 'max'),
            n_rows=('GR', 'size'),
        )
        .reset_index()
    )
    return agg


def cluster_wells_by_geo_signature(
    signature_df: pd.DataFrame,
    *,
    n_clusters: int = 8,
    feature_cols: list[str] | None = None,
    linkage: str = 'ward',
) -> pd.DataFrame:
    """Hierarchical clustering of wells by (X, Y) + GR signature.

    Returns signature_df with 'cluster_id' column added (= int 0..n_clusters-1).
    """
    if feature_cols is None:
        feature_cols = [
            'x_centroid', 'y_centroid',  # spatial
            'gr_mean', 'gr_std',           # GR distribution
        ]

    df = signature_df.copy()
    X = df[feature_cols].to_numpy(np.float32)
    # Z-normalize (= equal weight on spatial + GR)
    X_scaled = StandardScaler().fit_transform(X)
    # Agglomerative ward (= variance-based merging)
    n_clusters = min(n_clusters, len(df))
    model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
    df['cluster_id'] = model.fit_predict(X_scaled).astype(np.int32)
    return df


def compute_per_cluster_tvt_pattern(
    full_df: pd.DataFrame,
    well_cluster_map: pd.DataFrame,
    *,
    md_bin_size: int = 100,
) -> pd.DataFrame:
    """Per-cluster TVT-vs-MD pattern (= median + std + n by MD bin).

    Used as a lookup table: for any (cluster_id, md_bin), get the expected TVT
    distribution. LGB can use this as a prior feature.

    Returns DataFrame with columns ['cluster_id', 'md_bin', 'tvt_median', 'tvt_std', 'n'].
    """
    required_full = {'well', 'MD', 'TVT'}
    if not required_full.issubset(full_df.columns):
        raise ValueError(f"full_df must have {required_full}")
    if 'cluster_id' not in well_cluster_map.columns:
        raise ValueError("well_cluster_map must have 'cluster_id'")

    df = full_df.merge(well_cluster_map[['well', 'cluster_id']], on='well', how='inner')
    df = df[df['TVT'].notna()].copy()
    df['md_bin'] = (df['MD'] // md_bin_size).astype(int)

    pattern = (
        df.groupby(['cluster_id', 'md_bin'])
        .agg(
            tvt_median=('TVT', 'median'),
            tvt_std=('TVT', 'std'),
            n=('TVT', 'size'),
        )
        .reset_index()
    )
    return pattern
