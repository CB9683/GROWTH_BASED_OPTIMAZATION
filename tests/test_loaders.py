import numpy as np
import vtk

from gbo.io import tissue_mesh_to_grid, artery_polydata_to_tree


def _make_sphere(fname="sphere.vtp", r=5.0):
    sphere = vtk.vtkSphereSource()
    sphere.SetRadius(r)
    sphere.SetThetaResolution(24)
    sphere.SetPhiResolution(24)
    sphere.Update()
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(fname)
    writer.SetInputData(sphere.GetOutput())
    writer.Write()
    return fname


def _make_y(fname="yline.vtp"):
    pts = vtk.vtkPoints()
    # Shared root point
    pts.InsertNextPoint(0, 0, 0)   # point 0

    # Upper-left branch
    pts.InsertNextPoint(-3, 5, 0)  # point 1
    # Upper-right branch
    pts.InsertNextPoint(3, 5, 0)   # point 2

    lines = vtk.vtkCellArray()

    # First branch: point 0 to point 1
    l1 = vtk.vtkPolyLine()
    l1.GetPointIds().SetNumberOfIds(2)
    l1.GetPointIds().SetId(0, 0)
    l1.GetPointIds().SetId(1, 1)
    lines.InsertNextCell(l1)

    # Second branch: point 0 to point 2
    l2 = vtk.vtkPolyLine()
    l2.GetPointIds().SetNumberOfIds(2)
    l2.GetPointIds().SetId(0, 0)
    l2.GetPointIds().SetId(1, 2)
    lines.InsertNextCell(l2)

    poly = vtk.vtkPolyData()
    poly.SetPoints(pts)
    poly.SetLines(lines)

    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(fname)
    writer.SetInputData(poly)
    writer.Write()
    return fname



def test_tissue_grid_from_sphere(tmp_path):
    fn = _make_sphere(tmp_path / "sph.vtp")
    tg = tissue_mesh_to_grid(fn, spacing=2.0)
    assert tg.unmet_voxels() < tg.demand.size          # some voxels have demand


def test_tree_from_polyline(tmp_path):
    fn = _make_y(tmp_path / "y.vtp")
    tree = artery_polydata_to_tree(fn, r_root=0.4)
    assert len(tree.terminals()) == 2                  # the 'Y' has two tips
