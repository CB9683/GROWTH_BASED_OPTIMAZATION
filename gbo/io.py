# gbo/io.py
"""
Stage-2 loaders: mesh-to-TissueGrid and PolyData-to-VascularTree.

Only numpy, scipy, networkx, dataclasses, logging, vtk are used.
"""

from __future__ import annotations
import logging
from pathlib import Path
from typing import Tuple, Optional, List

import numpy as np
import vtk
from vtk.util import numpy_support
from .core import TissueGrid, VascularTree

logger = logging.getLogger("gbo")


# ------------------------------------------------------------------------------
# Helper -----------------------------------------------------------------------
# ------------------------------------------------------------------------------

def _read_any_vtk(filename: str) -> vtk.vtkDataObject:
    """
    Dispatch helper that picks the correct VTK reader based on file extension.
    """
    p = Path(filename)
    if p.suffix == ".vtk":
        reader = vtk.vtkGenericDataObjectReader()
    elif p.suffix == ".vtp":
        reader = vtk.vtkXMLPolyDataReader()
    elif p.suffix == ".vtu":
        reader = vtk.vtkXMLUnstructuredGridReader()
    else:
        raise ValueError(f"Unsupported mesh type {p.suffix}")
    reader.SetFileName(str(p))
    reader.Update()
    return reader.GetOutput()


# ------------------------------------------------------------------------------
# (a) tissue mesh → TissueGrid --------------------------------------------------
# ------------------------------------------------------------------------------

def tissue_mesh_to_grid(path: str | Path,
                        spacing: float,
                        demand_value: float = 1.0,
                        padding: float = 1.0) -> TissueGrid:
    """
    Voxelise an arbitrary closed tissue surface / volume into a regular grid.

    Parameters
    ----------
    path : str | Path
        Mesh file (.vtk, .vtp, .vtu, …) describing the tissue to perfuse.
        Must define a *closed* surface.
    spacing : float
        Requested voxel size (mm).
    demand_value : float, default 1.0
        Metabolic demand assigned to voxels inside the mesh.
    padding : float, default 1.0 mm
        Extra margin around the geometry to avoid boundary clipping.

    Returns
    -------
    TissueGrid
    """
    mesh = _read_any_vtk(str(path))
    if isinstance(mesh, vtk.vtkUnstructuredGrid):
        surf_f = vtk.vtkGeometryFilter()
        surf_f.SetInputData(mesh)
        surf_f.Update()
        mesh = surf_f.GetOutput()

    logger.info("Voxelising mesh '%s' …", path)

    # ------------------------------------------------------------------ bounds / dims
    bounds = mesh.GetBounds()  # (xmin,xmax,ymin,ymax,zmin,zmax)
    xmin, xmax, ymin, ymax, zmin, zmax = bounds
    xmin -= padding
    ymin -= padding
    zmin -= padding
    xmax += padding
    ymax += padding
    zmax += padding

    dims = (np.ceil((xmax - xmin) / spacing).astype(int),
            np.ceil((ymax - ymin) / spacing).astype(int),
            np.ceil((zmax - zmin) / spacing).astype(int))

    # ------------------------------------------------------------------ generate centres
    tg = TissueGrid(origin=np.array([xmin, ymin, zmin]),
                    spacing=float(spacing),
                    dims=tuple(dims),
                    demand_value=0.0)               # start with 0; fill later

    centres = tg.voxel_centers()

    # ------------------------------------------------------------------ inside–test with SelectEnclosedPoints
    pts = vtk.vtkPoints()
    pts.SetData(numpy_support.numpy_to_vtk(centres))
    cloud = vtk.vtkPolyData()
    cloud.SetPoints(pts)

    enclosed = vtk.vtkSelectEnclosedPoints()
    enclosed.SetInputData(cloud)
    enclosed.SetSurfaceData(mesh)
    enclosed.Update()

    inside_arr = numpy_support.vtk_to_numpy(
        enclosed.GetOutput().GetPointData().GetArray("SelectedPoints")
    ).astype(bool)

    tg.demand.flat[inside_arr] = demand_value
    tg.ownership.flat[~inside_arr] = -2  # -2 → outside tissue, never considered

    logger.info("TissueGrid created: %s", tg)
    return tg


# ------------------------------------------------------------------------------
# (b) major arteries → initial VascularTree ------------------------------------
# ------------------------------------------------------------------------------

def artery_polydata_to_tree(path: str | Path,
                            r_root: float = 0.5) -> VascularTree:
    """
    Build an initial supply tree from a centre-line polydata.

    The polydata must contain *Lines* (one or more polylines).  If a point-data
    array named “Radius” exists, it is used for child radii; otherwise a constant
    `r_root` is used everywhere and Murray’s law updates the parents at branch
    points.

    Parameters
    ----------
    path : str | Path
        PolyData file (.vtp or .vtk) with centre-lines.
    r_root : float, default 0.5 mm
        Fallback radius if the mesh has none.

    Returns
    -------
    VascularTree
    """
    poly = _read_any_vtk(str(path))
    if not isinstance(poly, vtk.vtkPolyData):
        raise TypeError("Expected PolyData with centre-lines")

    logger.info("Parsing artery file '%s' …", path)
    tree = VascularTree()

    # ---------------------------- get radii array if present
    radius_arr = None
    if poly.GetPointData().HasArray("Radius"):
        radius_arr = numpy_support.vtk_to_numpy(
            poly.GetPointData().GetArray("Radius"))
        logger.debug("Using per-point radii from 'Radius' array")

    # ---------------------------- iterate over line cells
    for c in range(poly.GetNumberOfCells()):
        cell = poly.GetCell(c)
        ids = cell.GetPointIds()
        npts = ids.GetNumberOfIds()
        parent_id: Optional[int] = None

        for k in range(npts):
            pid = ids.GetId(k)
            pos = np.array(poly.GetPoint(pid))
            rad = radius_arr[pid] if radius_arr is not None else r_root
            nid = tree.add_node(pos, radius=float(rad), parent=parent_id)
            parent_id = nid

        # after creating the polyline, retro-propagate Murray radii upwards
        # (children now known)
        # iterate backwards so cascaded updates happen bottom-up
        for nid in reversed(range(tree._next_id - npts, tree._next_id)):
            tree.enforce_murray(nid)

    logger.info("Initial tree from '%s': %d nodes", path, tree.graph.number_of_nodes())
    return tree
