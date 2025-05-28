from gbo.core import VascularTree, TissueGrid
from gbo import search, update


def test_apply_bifurcation_clamps_root():
    tg = TissueGrid(origin=[-2, -2, -2], spacing=1, dims=(5, 5, 5))
    tree = VascularTree()
    root = tree.add_node([0, 0, 0], radius=1.0, radius_max=1.0)   # source
    tip = tree.add_node([1, 0, 0], radius=0.4, parent=root)

    tg.voronoi_partition(tree)
    tg.grow_regions()

    mu, alpha, qmin, rmin = 3.6e-3, 1e-6, 1.0, 0.4
    demand = search.terminal_demand(tg)
    tree.propagate_flows(demand, mu, qmin, rmin, alpha)

    # Either use the algorithm’s own proposal …
    dL, cfg = search.find_best_bifurcation(tree, tg, l_min=3.0,
                                           mu=mu, q_min=qmin,
                                           r_min=rmin, alpha=alpha)
    # … or fabricate a simple one if none improves loss
    if cfg is None:
        cfg = dict(parent=root, tip=tip,
                   bif_pos=[0.5, 0, 0], new_tip_pos=[0.5, 1.0, 0])

    update.apply_bifurcation(tree, tg, cfg, mu, qmin, rmin, alpha)

    # root radius must respect anatomical clamp
    assert tree.graph.nodes[root]["data"].radius <= 1.0
    # all segment radii strictly positive (no div-by-zero hazards)
    for seg in tree.edges():
        assert seg.radius > 0
