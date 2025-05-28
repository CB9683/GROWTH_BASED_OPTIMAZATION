"""
Stage-8 exporters: CSV, JSON and VTK PolyData.

Functions are intentionally small; they are called from runner after the
growth loop finishes.
"""
from __future__ import annotations
import json, logging, csv
from pathlib import Path
from typing import Iterable

import numpy as np
import vtk
from vtk.util import numpy_support as nps

from .core import VascularTree

logger = logging.getLogger("gbo")


# ------------------------------------------------------------------------------
# 1. CSV of nodes
# ------------------------------------------------------------------------------

def _write_nodes_csv(tree: VascularTree, path: Path):
    hdr = ("id", "x_mm", "y_mm", "z_mm", "radius_mm",
           "flow_uL_s", "parent_id")
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(hdr)
        for nid, nd in tree.graph.nodes(data="data"):
            w.writerow([
                nid, *nd.pos.tolist(), f"{nd.radius:.6g}",
                f"{nd.flow:.6g}", nd.parent if nd.parent is not None else -1,
            ])
    logger.info("Wrote CSV %s", path)


# ------------------------------------------------------------------------------
# 2. JSON summary
# ------------------------------------------------------------------------------

def _write_run_json(path: Path,
                    params: dict,
                    loss_hist: list[float],
                    iters: int,
                    terminals: int):
    d = dict(
        parameters=params,
        iterations=iters,
        final_terminals=terminals,
        loss_history=loss_hist,
    )
    path.write_text(json.dumps(d, indent=2))
    logger.info("Wrote JSON %s", path)


# ------------------------------------------------------------------------------
# 3. VTK PolyData
# ------------------------------------------------------------------------------

def _tree_to_polydata(tree: VascularTree) -> vtk.vtkPolyData:
    pts = vtk.vtkPoints()
    # map node id → point id in vtk
    pid_of: dict[int, int] = {}
    for nid, nd in tree.graph.nodes(data="data"):
        pid = pts.InsertNextPoint(*nd.pos)
        pid_of[nid] = pid

    lines = vtk.vtkCellArray()
    radii = np.empty(tree.graph.number_of_edges(), float)
    flows = np.empty_like(radii)

    for k, seg in enumerate(tree.edges()):
        poly = vtk.vtkLine()
        poly.GetPointIds().SetId(0, pid_of[seg.parent])
        poly.GetPointIds().SetId(1, pid_of[seg.child])
        lines.InsertNextCell(poly)
        radii[k] = seg.radius
        flows[k] = seg.flow

    pd = vtk.vtkPolyData()
    pd.SetPoints(pts)
    pd.SetLines(lines)
    pd.GetCellData().AddArray(nps.numpy_to_vtk(radii, deep=True, array_type=vtk.VTK_DOUBLE))
    pd.GetCellData().GetArray(0).SetName("Radius_mm")
    pd.GetCellData().AddArray(nps.numpy_to_vtk(flows, deep=True, array_type=vtk.VTK_DOUBLE))
    pd.GetCellData().GetArray(1).SetName("Flow_uL_s")
    return pd


def _write_vtp(tree: VascularTree, path: Path):
    pd = _tree_to_polydata(tree)
    wr = vtk.vtkXMLPolyDataWriter()
    wr.SetFileName(str(path))
    wr.SetInputData(pd)
    wr.Write()
    logger.info("Wrote VTP %s", path)


# ------------------------------------------------------------------------------
# Public orchestration
# ------------------------------------------------------------------------------

def write_all_outputs(tree: VascularTree,
                      params: dict,
                      loss_hist: list[float],
                      outdir: Path,
                      tag: str = "final"):
    """
    Write <outdir>/<tag>_tree.vtp, <tag>_nodes.csv, <tag>.json, plus
    loss_history.csv for quick plotting elsewhere.
    """
    outdir.mkdir(parents=True, exist_ok=True)
    _write_vtp(tree, outdir / f"{tag}_tree.vtp")
    _write_nodes_csv(tree, outdir / f"{tag}_nodes.csv")
    _write_run_json(outdir / f"{tag}.json", params,
                    loss_hist, len(loss_hist) - 1, len(tree.terminals()))

    # save loss history as CSV for external plotting
    csv_path = outdir / f"{tag}_loss_history.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(("iter", "loss"))
        for i, L in enumerate(loss_hist):
            w.writerow((i, f"{L:.8g}"))
    logger.info("Wrote CSV %s", csv_path)
