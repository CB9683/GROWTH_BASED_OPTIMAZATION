# tests/test_stage5.py
import numpy as np
from gbo import search                           # import the new search module
from gbo.core import VascularTree, TissueGrid

def test_loss_decreases():
    # --- minimal demo setup --------------------------------------------------
    tg = TissueGrid(origin=[-1, -1, -1], spacing=1.0, dims=(3, 3, 3))
    tree = VascularTree()
    root_tip = tree.add_node([0, 0, 0], radius=0.4)  # single terminal

    tg.voronoi_partition(tree)
    tg.grow_regions()                                # expand once for demand

    # --- parameters ----------------------------------------------------------
    mu = 3.6e-3          # Pa·s
    alpha = 1e-6         # metabolic coeff
    qmin = 1.0           # μL/s for r_min
    rmin = 0.4           # mm

    # --- propagate + baseline loss ------------------------------------------
    demand = search.terminal_demand(tg)
    tree.propagate_flows(demand, mu, qmin, rmin, alpha)
    L0 = tree.total_loss()

    # --- bifurcation search --------------------------------------------------
    dL, cfg = search.find_best_bifurcation(tree, tg, l_min=2.0,
                                           mu=mu, q_min=qmin,
                                           r_min=rmin, alpha=alpha)
    # Best ΔL should not be positive
    assert dL <= 0.0
