"""
gbo/core.py
===========

Core data structures and helpers for Growth-Based Optimisation (GBO):
* VascularTree – directed graph of Node objects
* TissueGrid   – regular voxel domain and Ω-growth utilities
"""

from __future__ import annotations

import copy
import logging
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from typing import Dict, Iterable, List, Optional

import networkx as nx
import numpy as np
from scipy import ndimage as ndi
from math import pi   # single-source π

# ------------------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------------------
logger = logging.getLogger("gbo")
if not logger.handlers:                 # avoid duplicates under pytest -q
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(_h)
logger.setLevel(logging.INFO)

# ------------------------------------------------------------------------------
# Low-level helpers
# ------------------------------------------------------------------------------
def murray_radius(sum_cubed: float) -> float:
    """Return parent radius obeying Murray’s law r₀³ = Σ rᵢ³."""
    return sum_cubed ** (1.0 / 3.0)


def segment_length(p0: np.ndarray, p1: np.ndarray) -> float:
    """Euclidean distance in **millimetres**."""
    return float(np.linalg.norm(p1 - p0))


def poiseuille_resistance(mu_SI: float, length_m: float, radius_m: float) -> float:
    """
    Hydrodynamic resistance of a cylindrical vessel (SI units).

    R = 8 μ ℓ / (π r⁴) where  
        μ  – Pa·s,  
        ℓ  – m,  
        r  – m.
    Conversion from mm/μL is done by callers.
    """
    return 8.0 * mu_SI * length_m / (pi * radius_m ** 4)


# ------------------------------------------------------------------------------
# Graph primitives
# ------------------------------------------------------------------------------
@dataclass(slots=True)
class Node:
    id: int
    pos: np.ndarray                      # (mm, mm, mm)
    radius: float                        # mm
    flow: float = 0.0                    # μL/s
    parent: Optional[int] = None
    children: List[int] = field(default_factory=list)
    radius_max: Optional[float] = None   # anatomical clamp

    # --- helpers -------------------------------------------------------------
    def is_terminal(self) -> bool:
        return not self.children

    def as_dict(self) -> Dict:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class Segment:
    parent: int
    child: int
    length: float        # mm
    radius: float        # mm
    flow: float          # μL/s

    @property
    def resistance(self) -> float:
        """Poiseuille resistance with μ=1 Pa·s (callers multiply by μ)."""
        # Convert mm → m for internal use
        return poiseuille_resistance(1.0, self.length * 1e-3, self.radius * 1e-3)


# ------------------------------------------------------------------------------
# Vascular tree
# ------------------------------------------------------------------------------
class VascularTree:
    """
    Wrapper around a `networkx.DiGraph` of `Node`s plus domain-specific helpers.
    Multiple independent roots are allowed (parent is None).
    """

    def __init__(self) -> None:
        self._g: nx.DiGraph = nx.DiGraph()
        self._next_id = 0
        # run-time constants populated by `propagate_flows`
        self._mu: float = 0.0
        self._alpha: float = 0.0

    # ---------------------------------------------------------------- add / lookup
    def add_node(
        self,
        pos: np.ndarray,
        radius: float,
        parent: Optional[int] = None,
        radius_max: Optional[float] = None,
    ) -> int:
        nid = self._next_id
        self._next_id += 1
        node = Node(
            id=nid,
            pos=np.asarray(pos, dtype=float),
            radius=float(radius),
            parent=parent,
            radius_max=radius_max,
        )
        self._g.add_node(nid, data=node)
        if parent is not None:
            self._g.add_edge(parent, nid)
            self._g.nodes[parent]["data"].children.append(nid)

        logger.debug("Added node %d (parent=%s)", nid, parent)
        return nid

    # ---------------------------------------------------------------- iterators
    @property
    def graph(self) -> nx.DiGraph:
        return self._g

    def nodes(self, data: bool = False) -> Iterable:
        return self._g.nodes(data=data)

    def edges(self) -> Iterable[Segment]:
        for u, v in self._g.edges():
            nu: Node = self._g.nodes[u]["data"]
            nv: Node = self._g.nodes[v]["data"]
            yield Segment(
                parent=u,
                child=v,
                length=segment_length(nu.pos, nv.pos),
                radius=nv.radius,
                flow=nv.flow,
            )

    # ---------------------------------------------------------------- queries
    def terminals(self) -> List[int]:
        return [n for n, d in self._g.nodes(data="data") if d.is_terminal()]

    def sources(self) -> List[int]:
        return [n for n, d in self._g.nodes(data="data") if d.parent is None]

    # ---------------------------------------------------------------- flow + radii
    def propagate_flows(
        self,
        demand_map: dict[int, float],
        mu: float,
        q_min: float,
        r_min: float,
        metabolic_coeff: float,
    ) -> None:
        """
        Bottom-up flow propagation plus Murray radius update.

        `demand_map` – μL/s demand per terminal  
        `mu`         – Pa·s (blood viscosity)  
        `q_min/r_min` define κ = q_min / r_min³ used in Murray law  
        `metabolic_coeff` (α) stored for `_segment_loss`
        """
        kappa = q_min / r_min ** 3

        # 1) Set terminal flows/radii; clamp anatomical sources
        for nid, nd in self._g.nodes(data="data"):
            if nd.is_terminal():
                nd.flow = demand_map.get(nid, 0.0)
                nd.radius = max(r_min, (nd.flow / kappa) ** (1 / 3))
            elif nd.radius_max is not None:
                nd.radius = min(nd.radius, nd.radius_max)

        # 2) Upward aggregation (children → parent)
        for nid in reversed(list(nx.topological_sort(self._g))):
            nd = self._g.nodes[nid]["data"]
            if nd.parent is None:  # root(s) will be clamped later
                continue
            par = self._g.nodes[nd.parent]["data"]
            par.flow = sum(self._g.nodes[c]["data"].flow for c in par.children)
            par.radius = max(r_min, (par.flow / kappa) ** (1 / 3))

        # 3) Re-clamp sources *after* Murray recursion
        for sid in self.sources():
            nd = self._g.nodes[sid]["data"]
            if nd.radius_max is not None:
                nd.radius = min(nd.radius, nd.radius_max)

        # store constants for loss evaluation
        self._mu = mu
        self._alpha = metabolic_coeff

    # ---------------------------------------------------------------- loss
    def _segment_loss(self, seg: Segment) -> float:
        # convert to SI
        length_m = seg.length * 1e-3
        radius_m = seg.radius * 1e-3
        flow_m3s = seg.flow * 1e-9

        visc = 8 * self._mu * length_m * flow_m3s ** 2 / (pi * radius_m ** 4)
        metab = self._alpha * radius_m ** 2 * length_m

        logger.debug("visc=%.3g  metab=%.3g  L=%.3g", visc, metab, visc + metab)
        return visc + metab

    def total_loss(self) -> float:
        return sum(self._segment_loss(seg) for seg in self.edges())

    # ---------------------------------------------------------------- misc
    def clone(self) -> "VascularTree":
        return copy.deepcopy(self)

    def to_dict(self) -> Dict:
        return {nid: nd.as_dict() for nid, nd in self._g.nodes(data="data")}

    # manual radius fix-up (rarely used now)
    def enforce_murray(self, nid: int) -> None:
        nd = self._g.nodes[nid]["data"]
        if not nd.children:
            return
        nd.radius = murray_radius(
            sum(self._g.nodes[c]["data"].radius ** 3 for c in nd.children)
        )


# ------------------------------------------------------------------------------
# Trial-bifurcation context (stage-5 helper)
# ------------------------------------------------------------------------------
@contextmanager
def trial_bifurcation(
    tree: VascularTree,
    tip_id: int,
    bif_point: np.ndarray,
    new_tip_pos: np.ndarray,
    r_min: float,
):
    """
    Context manager that yields a *temporary* modified copy of `tree`
    containing a trial bifurcation.

    The original tree is untouched outside the context.
    """
    tmp = tree.clone()

    # --- split existing tip -----------------------------------------------
    old_tip = tmp.graph.nodes[tip_id]["data"]
    parent = tmp.graph.nodes[old_tip.parent]["data"]

    bif_id = tmp.add_node(bif_point, r_min, parent=parent.id)

    # reconnect edges
    tmp.graph.remove_edge(parent.id, old_tip.id)
    parent.children.remove(old_tip.id)

    parent.children.append(bif_id)
    tmp.graph.add_edge(parent.id, bif_id)

    old_tip.parent = bif_id
    tmp.graph.add_edge(bif_id, old_tip.id)
    tmp.graph.nodes[bif_id]["data"].children.append(old_tip.id)

    # --- new terminal ------------------------------------------------------
    tmp.add_node(new_tip_pos, r_min, parent=bif_id)

    try:
        yield tmp
    finally:
        del tmp  # allow GC


# ------------------------------------------------------------------------------
# Tissue grid
# ------------------------------------------------------------------------------
class TissueGrid:
    """
    Regular cubic voxel grid holding metabolic demand (constant for now) and
    Voronoi ownership of supplying terminals.
    """

    # -------------------- init --------------------
    def __init__(
        self,
        origin: np.ndarray,
        spacing: float,
        dims: tuple[int, int, int],
        demand_value: float = 1.0,
    ) -> None:
        self.origin = np.asarray(origin, float)
        self.spacing = float(spacing)
        self.dims = tuple(map(int, dims))

        self.demand_per_voxel = float(demand_value)
        self.demand = np.full(self.dims, self.demand_per_voxel, dtype=float)

        # ownership: −2 outside tissue, −1 unmet, ≥0 terminal id
        self.ownership = np.full(self.dims, -1, dtype=int)

    # -------------------- voxel utilities ----------
    @property
    def voxel_volume(self) -> float:
        return self.spacing ** 3

    def voxel_centers(self) -> np.ndarray:
        zi, yi, xi = np.indices(self.dims)
        idx = np.stack((xi, yi, zi), axis=-1).reshape(-1, 3)
        return self.origin + (idx + 0.5) * self.spacing

    # -------------------- Voronoi seeding ----------
    def voronoi_partition(self, tree: VascularTree) -> None:
        terminals = tree.terminals()
        if not terminals:
            logger.warning("Voronoi seeding skipped – no terminals")
            return

        unsup = (self.ownership == -1)
        if not unsup.any():
            return

        centres = self.voxel_centers()
        unsup_idx = np.flatnonzero(unsup)
        unsup_centres = centres[unsup_idx]

        for tid in terminals:
            if (self.ownership == tid).any():
                continue
            tip_pos = tree.graph.nodes[tid]["data"].pos
            nearest_flat = unsup_idx[np.argmin(((unsup_centres - tip_pos) ** 2).sum(1))]
            self.ownership.flat[nearest_flat] = tid

            # shrink search arrays
            keep = unsup_idx != nearest_flat
            unsup_idx = unsup_idx[keep]
            unsup_centres = unsup_centres[keep]
            if unsup_centres.size == 0:
                break

        logger.debug("Voronoi seeding: %d/%d terminals initialised",
                     (self.ownership >= 0).sum(), len(terminals))

    # -------------------- Ω-growth -----------------
    def grow_regions(self) -> tuple[int, float, float]:
        supplied = (self.ownership >= 0)
        if not supplied.any():
            logger.warning("grow_regions(): no supplied tissue yet")
            return 0, 0.0, 0.0

        kernel = ndi.generate_binary_structure(3, 1)  # 6-conn
        added = np.zeros_like(self.ownership, bool)

        for tid in np.unique(self.ownership[supplied]):
            new_vox = ndi.binary_dilation(self.ownership == tid, structure=kernel) \
                      & (self.ownership == -1)
            self.ownership[new_vox] = tid
            added |= new_vox
            if new_vox.any():
                logger.debug("Terminal %d gained %d vox", tid, new_vox.sum())

        n = int(added.sum())
        v_added = n * self.voxel_volume
        dq = n * self.demand_per_voxel
        if n:
            logger.info("Grew Ω: +%d vox (%.2f mm³)  ΔQ=%.2e", n, v_added, dq)
        return n, v_added, dq

    # -------------------- candidates -------------
    def candidate_centroids(self, k: int) -> list[np.ndarray]:
        mask = (self.ownership == -1)
        if not mask.any():
            return []

        struct = ndi.generate_binary_structure(3, 1)
        labels, nreg = ndi.label(mask, structure=struct)

        centres_all = self.voxel_centers()
        regions: list[tuple[int, np.ndarray]] = []
        for lab in range(1, nreg + 1):
            idx = np.flatnonzero(labels == lab)
            if idx.size:
                regions.append((idx.size, centres_all[idx].mean(0)))

        regions.sort(key=lambda t: t[0], reverse=True)
        return [c for _, c in regions[:k]]

    # -------------------- demand split -----------
    @staticmethod
    def terminals_needed(delta_q: float, q_per_terminal: float) -> int:
        if q_per_terminal <= 0:
            raise ValueError("q_per_terminal must be > 0")
        return max(1, int(np.ceil(delta_q / q_per_terminal)))

    # -------------------- misc -------------------
    def unmet_voxels(self) -> int:
        return int((self.ownership == -1).sum())

    def __repr__(self) -> str:
        pct = 100.0 * (1 - self.unmet_voxels() / np.prod(self.dims))
        return f"<TissueGrid {self.dims} vox | {pct:.1f}% supplied>"
