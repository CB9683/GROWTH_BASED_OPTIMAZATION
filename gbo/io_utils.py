# gbo/io_utils.py
# ────────────────────────────────────────────────────────────────────────────
"""
Convenience helpers for persisting the state of a GBO growth run
at every iteration in **portable, human-readable** formats.

Written files  (##### = zero-padded iteration index)
────────────────────────────────────────────────────
nodes_#####.csv      – node table  (id, x, y, z, r, q, parent)
edges_#####.csv      – edge list   (parent id, child id)
tissue_#####.npz     – ownership & demand 3-D arrays
meta_#####.json      – tiny JSON with iter index & loss scalars
loss_history.csv     – append-only logfile of losses

If  write_pickle=True  is passed, an additional
snapshot_#####.pkl   – monolithic Python pickle (tree + tissue + losses)
is also written for convenience.

All paths are placed inside the *outdir* given by the caller.
"""

from __future__ import annotations

import copy
import csv
import json
import pickle
from pathlib import Path
from typing import Dict

import numpy as np

from .core import VascularTree, TissueGrid


# ────────────────────────────────────────────────────────────────────────────
def save_iter_snapshot(
    iter_idx: int,
    outdir:   Path,
    tree:     VascularTree,
    tissue:   TissueGrid,
    losses:   Dict[str, float],
    *,                       # keyword-only toggles ↓
    write_pickle: bool = False,
) -> None:
    """Persist **one** growth iteration to disk.

    Parameters
    ----------
    iter_idx
        Current iteration number.
    outdir
        Directory where files are written (created if missing).
    tree, tissue
        The current vascular tree and tissue grid.
    losses
        Anything you want to track – typically
        ``dict(L_total=..., L_visc=..., L_metab=...)``.
    write_pickle
        If *True*, also saves a monolithic ``snapshot_#####.pkl`` that
        contains deep copies of ``tree`` and ``tissue`` for quick reloads.
    """
    outdir.mkdir(parents=True, exist_ok=True)

    # ────────────────────────── 1. nodes CSV ──────────────────────────────
    nodes_csv = outdir / f"nodes_{iter_idx:05d}.csv"
    with nodes_csv.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(
            ["id", "x_mm", "y_mm", "z_mm", "radius_mm", "flow_uL_s", "parent_id"]
        )
        for nid, ndata in tree.graph.nodes(data="data"):
            parent = ndata.parent if ndata.parent is not None else -1
            w.writerow([nid, *ndata.pos, ndata.radius, ndata.flow, parent])

    # ────────────────────────── 2. edges CSV ──────────────────────────────
    edges_csv = outdir / f"edges_{iter_idx:05d}.csv"
    with edges_csv.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["parent_id", "child_id"])
        for u, v in tree.graph.edges():
            w.writerow([u, v])

    # ────────────────────────── 3. tissue arrays ──────────────────────────
    np.savez_compressed(
        outdir / f"tissue_{iter_idx:05d}.npz",
        ownership=tissue.ownership.astype(np.int32),
        demand=tissue.demand.astype(np.float32),
    )

    # ────────────────────────── 4. meta JSON ──────────────────────────────
    meta = dict(iter=iter_idx, losses=losses)
    (outdir / f"meta_{iter_idx:05d}.json").write_text(json.dumps(meta, indent=2))

    # ────────────────────────── 5. loss history CSV ───────────────────────
    hist_csv = outdir / "loss_history.csv"
    header = not hist_csv.exists()
    with hist_csv.open("a", newline="") as fh:
        w = csv.writer(fh)
        if header:
            w.writerow(["iter", *losses.keys()])
        w.writerow([iter_idx, *losses.values()])

    # ────────────────────────── 6. optional pickle ────────────────────────
    if write_pickle:
        snap = dict(
            iter=iter_idx,
            tree=tree.clone(),               # deep-copy graph structure
            tissue=copy.deepcopy(tissue),    # deep-copy ndarray fields
            losses=losses,
        )
        with (outdir / f"snapshot_{iter_idx:05d}.pkl").open("wb") as fh:
            pickle.dump(snap, fh)

# ────────────────────────────────────────────────────────────────────────────
# Nothing below – all functionality lives in save_iter_snapshot()
# ────────────────────────────────────────────────────────────────────────────
