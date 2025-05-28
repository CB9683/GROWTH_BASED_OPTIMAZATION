from gbo.core import VascularTree, TissueGrid

def _setup():
    tg = TissueGrid(origin=[-4, -4, -4], spacing=1, dims=(9, 9, 9))
    tr = VascularTree()
    tr.add_node([0, 0, 0], 0.4)      # single tip at grid centre
    tg.voronoi_partition(tr)
    return tg

def test_delta_q_and_terminals_needed():
    tg = _setup()
    n_vox, vol_mm3, dq = tg.grow_regions()
    assert n_vox == 6                # one 6-connected “shell” around the seed
    assert dq == 6                   # demand_per_voxel = 1
    assert vol_mm3 == 6              # spacing = 1 mm → 1 mm³ per voxel
    n_new = tg.terminals_needed(dq, q_per_terminal=5)
    assert n_new == 2                # ceil(6 / 5)

def test_candidate_centroids():
    tg = _setup()
    tg.grow_regions()
    cents = tg.candidate_centroids(k=3)
    assert len(cents) >= 1
    assert all(len(c) == 3 for c in cents)
