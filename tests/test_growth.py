import numpy as np
from gbo.core import VascularTree, TissueGrid


def _make_grid():
    tg = TissueGrid(origin=[-4, -4, -4], spacing=1.0, dims=(9, 9, 9))
    tree = VascularTree()
    tree.add_node([0, 0, 0], 0.4)      # centre
    tg.voronoi_partition(tree)
    return tg


def test_voronoi_seed():
    tg = _make_grid()
    assert tg.unmet_voxels() == 728        # 1 voxel assigned


def test_single_growth_layer():
    tg = _make_grid()
    tg.grow_regions()
    assert tg.unmet_voxels() == 722       # 98 new voxels claimed
