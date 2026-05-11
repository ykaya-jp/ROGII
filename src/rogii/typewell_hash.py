"""Typewell content-hash for ROGII fold construction.

Discovered by subagent G commit `1bb6caf` (Edge Q design): 35 train wells
share their typewell file's (TVT, GR, Geology) byte content with another
well, forming **13 duplicate groups**. Without joining them into the same
CV fold, the validation-side well's typewell is also in the training-side
well's data → cross-fold leak.

Host-confirmed (discussion topic 698449 reply): typewell files come from
real offset wells, and some target wells share the same offset well.

This module provides:
    - ``compute_typewell_hashes`` — MD5 of (TVT, GR, Geology) per well
    - ``EXPECTED_DUPLICATE_GROUP_COUNT`` — 13 (snapshot for regression)
    - ``EXPECTED_DUPLICATE_TOTAL_WELLS`` — 35 (snapshot)
    - ``count_duplicate_groups`` — assert helper for tests

Used by:
    - ``src.rogii.cv.build_edge_q_folds`` (= Edge Q v2)
    - ``src.rogii.cv.build_stratified_edge_q_folds`` (= exp010 fold reform)
    - ``src.rogii.cv.build_multi_key_stratified_edge_q_folds`` (= C2)
    - kaggle_kernels/exp007 / exp008 / exp009 / exp010 (= inline copy for
      kernel self-containedness; keep API identical to this module)
"""

from __future__ import annotations

import hashlib
from collections import Counter
from pathlib import Path
from typing import Iterable, Tuple

import pandas as pd


# ────────────────────────────────────────────────────────────────────────────
# Snapshot constants (= reproducibility regression target)
# ────────────────────────────────────────────────────────────────────────────

EXPECTED_DUPLICATE_GROUP_COUNT: int = 13
"""Number of typewell-hash groups containing >1 well. Confirmed in commit
`1bb6caf` against train/ data as of 2026-05-10. Any change should be
investigated (= train set updated or hash algorithm drift)."""

EXPECTED_DUPLICATE_TOTAL_WELLS: int = 35
"""Total wells across all duplicate groups (= sum of group sizes). 35 was
confirmed against train/ as of 2026-05-10."""

HASH_COLS_DEFAULT: Tuple[str, ...] = ("TVT", "GR", "Geology")
"""Columns hashed when computing typewell content identity. Order matters
for hash stability; do not reorder without bumping the regression snapshot."""


# ────────────────────────────────────────────────────────────────────────────
# Core hashing
# ────────────────────────────────────────────────────────────────────────────


def compute_typewell_hashes(
    typewell_dir: Path,
    well_ids: Iterable[str],
    cols: Tuple[str, ...] = HASH_COLS_DEFAULT,
) -> dict[str, str]:
    """Return ``{well_id: hex_digest}`` for each well's typewell content.

    Wells with a missing or unreadable typewell file fall back to their own
    well_id as a singleton group identifier (= still safe under GroupKFold,
    just gets its own fold-private group).

    Hash semantics:
        - Object columns (= Geology label strings): fillna("NaN"), join with
          "|" separator, encode as UTF-8 bytes, MD5 update.
        - Numeric columns (= TVT, GR floats): fillna(-99999.0), round to
          4 decimals, take ``.values.tobytes()`` for MD5 update.

    The kernel scripts in ``kaggle_kernels/exp007..exp010`` keep an *inline*
    copy of this function so the production environment has no runtime
    import dependency on ``rogii``. **If you change this function, mirror
    the change in those kernel scripts** and bump
    ``EXPECTED_DUPLICATE_GROUP_COUNT`` if the snapshot moves.
    """
    typewell_dir = Path(typewell_dir)
    hashes: dict[str, str] = {}
    for wid in well_ids:
        tw_path = typewell_dir / f"{wid}__typewell.csv"
        if not tw_path.exists():
            hashes[wid] = wid
            continue
        try:
            df = pd.read_csv(tw_path)
        except Exception:
            hashes[wid] = wid
            continue
        h = hashlib.md5()
        for col in cols:
            if col not in df.columns:
                continue
            if df[col].dtype == "object":
                payload = (
                    df[col].fillna("NaN").astype(str).str.cat(sep="|").encode()
                )
            else:
                payload = (
                    df[col].fillna(-99999.0).round(4).values.tobytes()
                )
            h.update(payload)
        hashes[wid] = h.hexdigest()
    return hashes


def count_duplicate_groups(
    hashes: dict[str, str]
) -> tuple[int, int]:
    """Return ``(n_groups_with_more_than_one_well, total_wells_in_those_groups)``.

    Useful as a regression assertion in tests, e.g.::

        h = compute_typewell_hashes(train_dir, well_ids)
        n_groups, n_wells = count_duplicate_groups(h)
        assert n_groups == EXPECTED_DUPLICATE_GROUP_COUNT
        assert n_wells == EXPECTED_DUPLICATE_TOTAL_WELLS
    """
    counter = Counter(hashes.values())
    dup_groups = {g: c for g, c in counter.items() if c > 1}
    return len(dup_groups), sum(dup_groups.values())


__all__ = [
    "EXPECTED_DUPLICATE_GROUP_COUNT",
    "EXPECTED_DUPLICATE_TOTAL_WELLS",
    "HASH_COLS_DEFAULT",
    "compute_typewell_hashes",
    "count_duplicate_groups",
]
