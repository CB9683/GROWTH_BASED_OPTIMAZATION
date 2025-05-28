"""
Stage-7: growth-loop controller, checkpointing, and CLI front-end.
"""

from __future__ import annotations
import argparse, json, logging, os, pickle, signal, sys, time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Tuple

import numpy as np

from .core import TissueGrid, VascularTree
from .io import tissue_mesh_to_grid, artery_polydata_to_tree   # for real data
from .io_utils import save_iter_snapshot
from . import search, update


# ------------------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------------------

logger = logging.getLogger("gbo")
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(h)
logger.setLevel(logging.INFO)

# ------------------------------------------------------------------------------
# Parameters (could be made CLI args later)
# ------------------------------------------------------------------------------

DEFAULT_PARAMS = dict(
    mu=0.0036,              # Pa·s
    alpha=0.05,             # metabolic coeff (Kim & Rundfeldt)
    q_min=0.2,              # μL/s flow of a r_min terminal
    r_min=0.20,             # mm
    q_per_terminal=30.0,    # demand a single tip can handle
    l_min=10.0,              # search radius (mm)
    candidate_k=20,         # #centroids examined per iter
    checkpoint_interval=1,
)


# ------------------------------------------------------------------------------
# Demo-sphere helpers (no VTK needed)
# ------------------------------------------------------------------------------

def _demo_tree_sphere(r_min: float) -> VascularTree:
    """
    Root at (-5,0,0), single terminal at (+5,0,0) – with a tiny wiggle.
    """
    tree = VascularTree()

    # ---- root ----------------------------------------------------
    root_id = tree.add_node([-5.0, 0.0, 0.0],
                            radius=r_min*3,
                            radius_max=r_min*3)

    # ---- first terminal (+ a small offset) -----------------------
    voxel = 1.0           # current grid spacing
    wiggle = 1.5          # ≥ 1 voxel, any direction

    rng = np.random.default_rng(42)
    direction = rng.normal(size=3)
    direction /= np.linalg.norm(direction) + 1e-9

    tip_pos = np.array([+5.0, 0.0, 0.0]) + wiggle * direction
    tree.add_node(tip_pos, radius=r_min, parent=root_id)
    print("First tip placed at:", tip_pos)
    return tree


def _demo_tree(r_min: float) -> VascularTree:
    tree = VascularTree()
    root = tree.add_node([0, 0, 0], radius=r_min*3, radius_max=r_min*3)
    tree.add_node([r_min*2, 0, 0], radius=r_min, parent=root)
    return tree

# in your runner, for demo only:
def _demo_tree_asym(r_min: float) -> VascularTree:
    tree = VascularTree()
    # place the root 5 mm to the left of centre
    tree.add_node([-5.0, 0.0, 0.0], radius=r_min*3, radius_max=r_min*3)
    tree.add_node([-5.0 + r_min*2, 0.0, 0.0], radius=r_min, parent=0)
    return tree

def _demo_ellipsoid_grid(spacing=1.0,
                         radii=(25.0, 20.0, 15.0)) -> TissueGrid:
    # compute dims with padding
    dims = tuple(int(2*ri/spacing + 4) for ri in radii)
    tg = TissueGrid(origin=[-dims[i]/2*spacing for i in range(3)],
                    spacing=spacing, dims=dims)
    # mark voxels inside the ellipsoid x²/a² + y²/b² + z²/c² <= 1
    zi, yi, xi = np.indices(dims).astype(float)
    centers = (xi + 0.5, yi + 0.5, zi + 0.5)
    a, b, c = [ri/spacing for ri in radii]
    mask = ((centers[0]-dims[0]/2)**2/a**2 +
            (centers[1]-dims[1]/2)**2/b**2 +
            (centers[2]-dims[2]/2)**2/c**2) <= 1.0
    tg.ownership[~mask] = -2
    return tg

# ------------------------------------------------------------------------------
# GrowthController
# ------------------------------------------------------------------------------

class GrowthController:
    """
    Encapsulates the while-loop Ωᵢ ⊂ Ω_target → grow → ΔQ → search → attach.
    """

    def __init__(self,
                 tissue: TissueGrid,
                 tree: VascularTree,
                 outdir: Path,
                 params: dict[str, Any] | None = None,
                 resume: bool = False) -> None:
        self.tissue = tissue
        self.tree = tree
        self.outdir = outdir
        self.params = DEFAULT_PARAMS | (params or {})
        self.loss_history: list[float] = []
        self.iter = 0

        # snapshot parameters once
        if not resume:
            outdir.mkdir(parents=True, exist_ok=True)
            ts = time.strftime("%Y%m%d-%H%M%S")
            (outdir / f"run-{ts}.json").write_text(json.dumps(self.params, indent=2))

        # handle Ctrl-C graceful checkpoint
        signal.signal(signal.SIGINT, self._signal_handler)

    # ------------------------------------------------------------------ signals
    def _signal_handler(self, sig, frame):
        logger.warning("KeyboardInterrupt – writing checkpoint …")
        self.save_checkpoint()
        sys.exit(1)

    # ------------------------------------------------------------------ ckpt
    def save_checkpoint(self):
        fname = self.outdir / f"chkpt_{self.iter:05d}.pkl"
        with fname.open("wb") as f:
            pickle.dump(dict(
                tissue=self.tissue,
                tree=self.tree,
                loss_hist=self.loss_history,
                iter=self.iter,
                rng_state=np.random.get_state(),
            ), f)
        logger.info("Checkpoint saved to %s", fname)

    # ------------------------------------------------------------------ main loop
    def run(self):
        p = SimpleNamespace(**self.params)
        # initial Voronoi seed & flow propagation
        self.tissue.voronoi_partition(self.tree)
        demand = search.terminal_demand(self.tree, self.tissue)
        self.tree.propagate_flows(demand, p.mu, p.q_min, p.r_min, p.alpha)
        for tid in self.tree.terminals():
            nd = self.tree.graph.nodes[tid]["data"]
            logger.debug(f"[ITER {self.iter}] Terminal {tid}: flow={nd.flow:.2f}, radius={nd.radius:.3f}")

        self.loss_history.append(self.tree.total_loss())

        while self.tissue.unmet_voxels() > 0:
            self.iter += 1

            # ---------------- Stage-3 grow Ω --------------------------
            _, _, dQ = self.tissue.grow_regions()
            if dQ == 0:
                break

            # ---------------- refresh flows BEFORE branching ----------
            demand = search.terminal_demand(self.tree, self.tissue)
            self.tree.propagate_flows(demand, p.mu, p.q_min, p.r_min, p.alpha)
            L_before = self.tree.total_loss()

            # ---------------- Stage-4 decide how many new tips --------
            n_new = self.tissue.terminals_needed(dQ, p.q_per_terminal)

            best_dL_iter = 0.0
            for _ in range(n_new):
                dL, cfg = search.find_best_bifurcation(
                    self.tree, self.tissue, p.l_min, p.mu,
                    p.q_min, p.r_min, p.alpha)
                if cfg is None or dL >= 0:
                    break
                best_dL_iter = min(best_dL_iter, dL)
                update.apply_bifurcation(self.tree, self.tissue, cfg,
                                         p.mu, p.q_min, p.r_min, p.alpha)

            # ---------------- record loss after any branching ---------
            L_after = self.tree.total_loss()
            self.loss_history.append(L_after)

            logger.info(
                "iter %03d | terminals=%d | ΔQ=%.2e μL/s | bestΔL=%.2e | L=%.2e",
                self.iter, len(self.tree.terminals()), dQ, best_dL_iter, L_after
            )

            # periodic checkpoint
            if self.iter % p.checkpoint_interval == 0:
                self.save_checkpoint()
                L = self.tree.total_loss()
                loss_dict = dict(L_total=L)     # only the scalar you already have
                save_iter_snapshot( self.iter,
                                    self.outdir,
                                    self.tree,
                                    self.tissue,
                                    loss_dict,
                                    write_pickle=True)
                


        # final checkpoint
        self.save_checkpoint()

        # Stage-8 exports
        from . import export
        export.write_all_outputs(self.tree, self.params,
                                 self.loss_history, self.outdir)

# ------------------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------------------

def _parse_cli():
    ap = argparse.ArgumentParser(description="Synthetic vascular-tree growth (GBO)")
    ap.add_argument("--output", type=Path, default=Path("results"),
                    help="output directory (default ./results)")
    ap.add_argument("--resume", type=Path, default=None,
                    help="resume from checkpoint .pkl")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--demo-sphere", action="store_true",
                    help="generate demo 25 mm sphere instead of loading files")
    return ap.parse_args()


def _load_checkpoint(path: Path) -> Tuple[TissueGrid, VascularTree,
                                         list[float], int]:
    d = pickle.load(path.open("rb"))
    np.random.set_state(d["rng_state"])
    return d["tissue"], d["tree"], d["loss_hist"], d["iter"]


def main():
    args = _parse_cli()
    if args.quiet:
        logger.setLevel(logging.WARNING)
    elif args.verbose:
        logger.setLevel(logging.DEBUG)

    outdir: Path = args.output
    outdir.mkdir(parents=True, exist_ok=True)

    if args.resume:
        tissue, tree, loss_hist, it_start = _load_checkpoint(args.resume)
        controller = GrowthController(tissue, tree, outdir,
                                      resume=True)
        controller.loss_history = loss_hist
        controller.iter = it_start
    elif args.demo_sphere:
        tissue = _demo_ellipsoid_grid()
        tree = _demo_tree_sphere(r_min=DEFAULT_PARAMS["r_min"])
        controller = GrowthController(tissue, tree, outdir)
    else:
        logger.error("For now you must use --demo-sphere or --resume")
        sys.exit(1)

    controller.run()


if __name__ == "__main__":
    main()