"""
Stage-6: apply the best bifurcation to the *live* tree and refresh flows.
"""
from __future__ import annotations
import logging
import numpy as np
import networkx as nx

from .core import VascularTree, TissueGrid
from .search import terminal_demand

logger = logging.getLogger("gbo")

# --------------------------------------------------------------------------- #
def apply_bifurcation(tree: VascularTree,
                      tissue: TissueGrid,
                      cfg: dict,
                      mu: float, q_min: float, r_min: float, alpha: float) -> None:
    """
    Attach the winning bifurcation (returned by search.find_best_bifurcation)
    to *tree*, update TissueGrid seeding, and propagate flows/radii.

    Parameters
    ----------
    cfg : dict with keys parent, tip, bif_pos, new_tip_pos
          (exactly the structure from Stage-5)
    """
    if cfg is None:
        logger.info("No bifurcation applied (bestΔL >= 0)")
        return

    par_id = cfg["parent"]
    tip_id = cfg["tip"]
    bif_p = cfg["bif_pos"]
    new_tip_p = cfg["new_tip_pos"]

    # ---------------- insert bifurcation node --------------------------------
    par_node = tree.graph.nodes[par_id]["data"]
    old_tip = tree.graph.nodes[tip_id]["data"]

    # remove old edge parent→tip
    tree.graph.remove_edge(par_id, tip_id)
    par_node.children.remove(tip_id)

    # add bifurcation node
    bif_id = tree.add_node(bif_p, r_min, parent=par_id)

    # reconnect old tip
    tree.graph.add_edge(bif_id, tip_id)
    tree.graph.nodes[bif_id]["data"].children.append(tip_id)
    old_tip.parent = bif_id

    # add new terminal
    new_tip_id = tree.add_node(new_tip_p, r_min, parent=bif_id)
    # Murray update for the bifurcation just created
    tree.enforce_murray(bif_id)
    tree.enforce_murray(par_id)

    # ---------------- seed Voronoi & propagate flows -------------------------
    tissue.voronoi_partition(tree)              # each new tip gets one voxel
    demand = terminal_demand(tree, tissue)
    tree.propagate_flows(demand, mu, q_min, r_min, alpha)

    logger.info("Applied bifurcation: parent=%d → bif=%d (old tip %d, new tip %d)",
                par_id, bif_id, tip_id, new_tip_id)
