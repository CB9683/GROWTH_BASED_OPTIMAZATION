"""
gbo/search.py – bifurcation search for GBO vessel growth
Implements candidate generation using the farthest unsupplied frontier voxels.
"""

from __future__ import annotations
import logging
from typing import Optional, Tuple
import copy
import numpy as np
from scipy import ndimage as ndi

from .core import VascularTree, TissueGrid, trial_bifurcation

logger = logging.getLogger("gbo")


# ----------------------------------------------------------------------------- helpers
def terminal_demand(tree: VascularTree, tissue: TissueGrid) -> dict[int, float]:
    """
    Map terminal-node id → metabolic flow demand [µL s⁻¹].
    Currently 1 µL s⁻¹ per supplied voxel.
    """
    demand: dict[int, float] = {}
    centres = tree.terminals()
    for tid in centres:
        voxels = int(np.count_nonzero(tissue.ownership == tid))
        demand[tid] = float(voxels)          # one unit per voxel (Eq 2)
    return demand


# ----------------------------------------------------------------------------- main search
def find_best_bifurcation(
    tree: VascularTree,
    tissue: TissueGrid,
    l_min: float,
    mu: float,
    q_min: float,
    r_min: float,
    alpha: float,
    k: int = 10,
) -> Tuple[float, Optional[dict]]:
    """
    Explore up to *k* candidate centroids and return (best ΔL, config dict).

    The routine:
    1.  finds the farthest unsupplied voxel next to each terminal’s Ωᵢ,
    2.  tries three parent–child split fractions (0.3, 0.5, 0.7),
    3.  evaluates ΔL **after** full flow propagation on the trial tree.

    A candidate is accepted later only if ΔL < 0.
    """
    # ------------------------------------------------------------------ baseline loss
    base_demand = terminal_demand(tree, tissue)
    tree.propagate_flows(base_demand, mu, q_min, r_min, alpha)
    base_L = tree.total_loss()

    best_dL: float = 0.0
    best_cfg: Optional[dict] = None

    # ------------------------------------------------------------------ candidate centroids
    unsup = (tissue.ownership == -1)
    struct = ndi.generate_binary_structure(3, 1)
    centres = tissue.voxel_centers()
    centroids: list[np.ndarray] = []

    for tid in tree.terminals():
        region = (tissue.ownership == tid)
        if not region.any():
            continue
        frontier = ndi.binary_dilation(region, struct) & unsup
        if not frontier.any():
            continue
        idx = np.flatnonzero(frontier)
        tip_pos = tree.graph.nodes[tid]["data"].pos
        far_idx = idx[np.argmax(np.sum((centres[idx] - tip_pos) ** 2, axis=1))]
        centroids.append(centres[far_idx])
        if len(centroids) >= k:
            break

    if not centroids:
        logger.info("No candidate new-tip positions found.")
        return 0.0, None

    # ------------------------------------------------------------------ evaluate each
    cand_id = 0
    for cent in centroids:
        for tip_id in tree.terminals():
            tip = tree.graph.nodes[tip_id]["data"]
            if np.linalg.norm(tip.pos - cent) > l_min:
                continue
            parent_id = tip.parent
            if parent_id is None:
                continue
            parent = tree.graph.nodes[parent_id]["data"]

            for frac in (0.3, 0.5, 0.7):
                bif_pos = parent.pos + frac * (tip.pos - parent.pos)

                # --- build trial tree --------------------------------------------
                trial_tree = tree.clone()
                trial_tissue = trial_tissue = copy.deepcopy(tissue)    

                # remove the old parent→tip edge and insert a bifurcation node
                trial_tree.graph.remove_edge(parent_id, tip_id)
                bif_id = trial_tree.add_node(bif_pos, r_min, parent=parent_id)

                # reconnect the original tip to the new bifurcation node
                trial_tree.graph.add_edge(bif_id, tip_id)
                trial_tree.graph.nodes[bif_id]["data"].children.append(tip_id)
                trial_tree.graph.nodes[tip_id]["data"].parent = bif_id

                # add the NEW daughter tip and remember its ID
                new_tip_id = trial_tree.add_node(cent, r_min, parent=bif_id)

                # ----------  re-seed Voronoi on the CLONED tissue  -----------------------
                trial_tissue.voronoi_partition(trial_tree)  

                # --- update demand: split the old tip’s flow between the two ------
                demand = terminal_demand(trial_tree, tissue)
                demand[new_tip_id] = q_min            # <── add this line

                # --- evaluate the loss for the trial tree -------------------------
                trial_tree.propagate_flows(demand, mu, q_min, r_min, alpha)
                dL = trial_tree.total_loss() - base_L

                logger.debug(
                    f"cand {cand_id}: parent={parent_id} tip={tip_id} "
                    f"frac={frac:.2f} ΔL={dL:+.3e}"
                )
                cand_id += 1

                if dL < best_dL:
                    best_dL = dL
                    best_cfg = dict(
                        parent=parent_id,
                        tip=tip_id,
                        bif_pos=bif_pos,
                        new_tip_pos=cent
                    )

    if best_cfg is None:
        logger.info("No negative ΔL candidate found.")
    else:
        logger.info(
            "Best candidate: parent=%d tip=%d bif_pos=%s new_tip=%s ΔL=%+.3g",
            best_cfg["parent"], best_cfg["tip"],
            best_cfg["bif_pos"], best_cfg["new_tip_pos"], best_dL,
        )

    return best_dL, best_cfg
