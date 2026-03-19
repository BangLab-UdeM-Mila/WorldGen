"""
WorldGen Mesh Boundary Accuracy Analysis
=========================================
Quantifies the geometric fidelity of collision proxies against
high-resolution visual meshes for each benchmark object.

Metrics
-------
  1. Hausdorff Distance (HD)
       max( max_{p in A} min_{q in B} ||p-q||,
            max_{q in B} min_{p in A} ||p-q|| )
       The worst-case boundary deviation. Standard in shape matching
       literature (Aspert et al., VCIP 2002).

  2. Mean Surface Distance (MSD)
       mean of all one-sided point-to-surface distances.
       Captures average boundary error (used in medical image segmentation,
       Nikolov et al., 2018).

  3. 95th-Percentile Hausdorff (HD95)
       Robust to outliers. Common in MICCAI benchmarks.

  4. Volume Overlap (Jaccard Index / IoU)
       |V_vis ∩ V_col| / |V_vis ∪ V_col|
       Measures how well the collision proxy covers the true object volume.

  5. Face Count Reduction Ratio
       Compression factor from visual to collision mesh.

  6. Surface Area Relative Error
       |A_vis - A_col| / A_vis

All linear distances are reported in millimetres for readability.
All computations use trimesh for mesh loading and scipy KD-trees
for efficient nearest-neighbour queries.

Usage:
  python scripts/05_boundary_analysis.py --objects_dir ./objects --output_dir ./logs

Reference implementations:
  - Aspert et al. "MESH: Measuring Errors between Surfaces using the
    Hausdorff distance", ICME 2002.
  - Cignoni et al. "Metro: Measuring Error on Simplified Surfaces",
    Computer Graphics Forum 1998.
"""

import os
import sys
import argparse
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

try:
    import trimesh
    from scipy.spatial import cKDTree
except ImportError:
    print("ERROR: pip install trimesh scipy")
    sys.exit(1)


# ── Color palette (publication quality) ──────────────────────────────────────
COLORS = {
    "sphere":    {"vis": "#2E86AB", "col": "#A23B72", "accent": "#F18F01"},
    "bevel_box": {"vis": "#C73E1D", "col": "#3B1F2B", "accent": "#44BBA4"},
    "l_bracket": {"vis": "#393E41", "col": "#E94F37", "accent": "#F5F5F5"},
}

OBJECT_LABELS = {
    "sphere":    "UV-Sphere (Rubber, ρ=1100 kg/m³)",
    "bevel_box": "Beveled Box (Wood, ρ=700 kg/m³)",
    "l_bracket": "L-Bracket (Aluminum, ρ=2700 kg/m³)",
}


# ── Core metric functions ─────────────────────────────────────────────────────

def sample_surface(mesh, n=50000):
    """Uniformly sample points on mesh surface."""
    pts, _ = trimesh.sample.sample_surface(mesh, n)
    return pts


def hausdorff_metrics(pts_a, pts_b):
    """
    Compute directed and symmetric Hausdorff distances,
    MSD, and HD95 between two point clouds.

    Returns dict with keys: hd, hd_ab, hd_ba, msd, hd95, all in same units as input.
    """
    tree_b = cKDTree(pts_b)
    tree_a = cKDTree(pts_a)

    d_ab, _ = tree_b.query(pts_a, k=1)   # A → B
    d_ba, _ = tree_a.query(pts_b, k=1)   # B → A

    hd_ab = float(np.max(d_ab))
    hd_ba = float(np.max(d_ba))
    hd    = max(hd_ab, hd_ba)
    msd   = float(np.mean(np.concatenate([d_ab, d_ba])))
    hd95  = float(np.percentile(np.concatenate([d_ab, d_ba]), 95))

    return {"hd": hd, "hd_ab": hd_ab, "hd_ba": hd_ba,
            "msd": msd, "hd95": hd95, "d_ab": d_ab, "d_ba": d_ba}


def volume_iou(mesh_a, mesh_b):
    """
    Approximate volumetric IoU via voxel grids.
    Both meshes are voxelized at 1mm pitch.
    """
    pitch = 0.001   # 1 mm

    def voxelize(mesh):
        if not mesh.is_watertight:
            mesh = mesh.convex_hull
        return mesh.voxelized(pitch=pitch).fill()

    try:
        vox_a = voxelize(mesh_a)
        vox_b = voxelize(mesh_b)

        # Align to common grid
        origin = np.minimum(vox_a.origin, vox_b.origin)
        shape  = np.maximum(
            vox_a.origin + np.array(vox_a.matrix.shape) * pitch,
            vox_b.origin + np.array(vox_b.matrix.shape) * pitch
        )
        grid_shape = np.ceil((shape - origin) / pitch).astype(int) + 1

        def to_grid(vox):
            g = np.zeros(grid_shape, dtype=bool)
            offset = np.round((vox.origin - origin) / pitch).astype(int)
            s = vox.matrix.shape
            g[offset[0]:offset[0]+s[0],
              offset[1]:offset[1]+s[1],
              offset[2]:offset[2]+s[2]] = vox.matrix
            return g

        g_a = to_grid(vox_a)
        g_b = to_grid(vox_b)

        inter = np.logical_and(g_a, g_b).sum()
        union = np.logical_or(g_a,  g_b).sum()
        return float(inter / union) if union > 0 else 0.0
    except Exception:
        return float('nan')


def analyze_object(name, vis_path, col_path, n_samples=50000):
    """Full boundary accuracy analysis for one object pair."""
    print(f"\n  Loading {name}...")

    vis_mesh = trimesh.load(vis_path, force='mesh')
    col_mesh = trimesh.load(col_path, force='mesh')

    if not vis_mesh.is_watertight:
        vis_mesh = vis_mesh.convex_hull
    if not col_mesh.is_watertight:
        col_mesh = col_mesh.convex_hull

    # Surface sampling
    pts_vis = sample_surface(vis_mesh, n_samples)
    pts_col = sample_surface(col_mesh, n_samples)

    # Hausdorff metrics (convert m → mm)
    h = hausdorff_metrics(pts_vis * 1000, pts_col * 1000)

    # Volume IoU
    iou = volume_iou(vis_mesh, col_mesh)

    # Face counts
    nf_vis = len(vis_mesh.faces)
    nf_col = len(col_mesh.faces)

    # Surface areas (mm²)
    sa_vis = vis_mesh.area * 1e6
    sa_col = col_mesh.area * 1e6
    sa_err = abs(sa_vis - sa_col) / sa_vis

    results = {
        "name":             name,
        "label":            OBJECT_LABELS[name],
        "faces_visual":     nf_vis,
        "faces_collision":  nf_col,
        "face_reduction":   round(nf_vis / nf_col, 1),
        "hausdorff_mm":     round(h["hd"],    4),
        "hd_ab_mm":         round(h["hd_ab"], 4),
        "hd_ba_mm":         round(h["hd_ba"], 4),
        "msd_mm":           round(h["msd"],   4),
        "hd95_mm":          round(h["hd95"],  4),
        "volume_iou":       round(iou, 4),
        "surface_area_visual_mm2":    round(sa_vis, 2),
        "surface_area_collision_mm2": round(sa_col, 2),
        "surface_area_rel_error":     round(sa_err, 4),
        # Keep raw distances for plotting
        "_d_ab": h["d_ab"].tolist(),
        "_d_ba": h["d_ba"].tolist(),
        "_pts_vis": pts_vis,
        "_pts_col": pts_col,
    }

    print(f"    Hausdorff:       {h['hd']:.4f} mm")
    print(f"    MSD:             {h['msd']:.4f} mm")
    print(f"    HD95:            {h['hd95']:.4f} mm")
    print(f"    Volume IoU:      {iou:.4f}")
    print(f"    Face reduction:  {nf_vis} → {nf_col} ({nf_vis/nf_col:.1f}×)")
    print(f"    SA rel. error:   {sa_err*100:.2f}%")

    return results


# ── Publication-quality figure ────────────────────────────────────────────────

def plot_results(all_results, out_dir):
    """
    Three-panel figure per object:
      (A) Metric summary table
      (B) Error distribution histogram (HD distance per sample)
      (C) 3D scatter: visual mesh coloured by local HD error

    Plus one summary comparison figure across all objects.
    """

    names = [r["name"] for r in all_results]
    n_obj = len(all_results)

    # ── Per-object detail figures ────────────────────────────────────────────
    for res in all_results:
        name   = res["name"]
        d_ab   = np.array(res["_d_ab"])
        d_ba   = np.array(res["_d_ba"])
        pts    = np.array(res["_pts_vis"])
        colors = COLORS[name]

        fig = plt.figure(figsize=(16, 5))
        fig.patch.set_facecolor("#0D1117")
        gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35)

        # Panel A: metric table
        ax_t = fig.add_subplot(gs[0])
        ax_t.set_facecolor("#161B22")
        ax_t.axis('off')

        metrics = [
            ("Hausdorff Distance",       f"{res['hausdorff_mm']:.4f} mm"),
            ("Mean Surface Distance",    f"{res['msd_mm']:.4f} mm"),
            ("HD95",                     f"{res['hd95_mm']:.4f} mm"),
            ("Volume IoU",               f"{res['volume_iou']:.4f}"),
            ("SA Relative Error",        f"{res['surface_area_rel_error']*100:.2f}%"),
            ("Face Reduction",           f"{res['face_reduction']:.0f}×"),
            ("Visual Faces",             f"{res['faces_visual']:,}"),
            ("Collision Faces",          f"{res['faces_collision']:,}"),
        ]

        ax_t.text(0.5, 0.97, res["label"], ha='center', va='top',
                  fontsize=9, color='white', fontweight='bold',
                  transform=ax_t.transAxes, wrap=True)

        for i, (k, v) in enumerate(metrics):
            y = 0.85 - i * 0.10
            ax_t.text(0.05, y, k, ha='left', va='center',
                      fontsize=8.5, color='#8B949E', transform=ax_t.transAxes)
            ax_t.text(0.98, y, v, ha='right', va='center',
                      fontsize=8.5, color=colors["accent"],
                      fontweight='bold', transform=ax_t.transAxes)
            ax_t.plot([0.02, 0.98], [y - 0.04, y - 0.04],
                  color='#21262D', lw=0.5,
                  transform=ax_t.transAxes)

        ax_t.set_title("Boundary Metrics", color='white', fontsize=10,
                        pad=8, fontweight='bold')

        # Panel B: distance histograms
        ax_h = fig.add_subplot(gs[1])
        ax_h.set_facecolor("#161B22")
        bins = np.linspace(0, np.percentile(np.concatenate([d_ab, d_ba]), 99), 60)
        ax_h.hist(d_ab, bins=bins, color=colors["vis"],   alpha=0.7,
                  label="Visual→Collision", density=True)
        ax_h.hist(d_ba, bins=bins, color=colors["col"],   alpha=0.7,
                  label="Collision→Visual", density=True)
        ax_h.axvline(res["msd_mm"],   color=colors["accent"], lw=1.5,
                     ls='--', label=f"MSD = {res['msd_mm']:.3f} mm")
        ax_h.axvline(res["hd95_mm"],  color='white',     lw=1.0,
                     ls=':', label=f"HD95 = {res['hd95_mm']:.3f} mm")
        ax_h.set_xlabel("Distance (mm)", color='#8B949E', fontsize=8)
        ax_h.set_ylabel("Density",       color='#8B949E', fontsize=8)
        ax_h.tick_params(colors='#8B949E', labelsize=7)
        for sp in ax_h.spines.values(): sp.set_color('#21262D')
        leg = ax_h.legend(fontsize=7, framealpha=0.3, labelcolor='white')
        ax_h.set_title("Surface Distance Distribution", color='white',
                        fontsize=10, pad=8, fontweight='bold')

        # Panel C: 3D point cloud coloured by local error
        ax_3d = fig.add_subplot(gs[2], projection='3d')
        ax_3d.set_facecolor("#161B22")
        fig.patch.set_facecolor("#0D1117")

        # Subsample for speed
        idx = np.random.choice(len(pts), min(3000, len(pts)), replace=False)
        p   = pts[idx]
        e   = d_ab[idx]

        sc = ax_3d.scatter(p[:,0], p[:,1], p[:,2],
                           c=e, cmap='plasma', s=0.8, alpha=0.8,
                           vmin=0, vmax=np.percentile(d_ab, 95))
        cbar = fig.colorbar(sc, ax=ax_3d, shrink=0.6, pad=0.1)
        cbar.set_label("Error (mm)", color='#8B949E', fontsize=7)
        cbar.ax.tick_params(colors='#8B949E', labelsize=6)
        ax_3d.set_title("Local HD Error Map", color='white',
                         fontsize=10, pad=4, fontweight='bold')
        ax_3d.tick_params(colors='#21262D', labelsize=0)
        ax_3d.grid(False)
        ax_3d.xaxis.pane.fill = False
        ax_3d.yaxis.pane.fill = False
        ax_3d.zaxis.pane.fill = False

        fig.suptitle(
            f"WorldGen Boundary Accuracy Analysis — {name}",
            color='white', fontsize=12, fontweight='bold', y=1.02
        )

        path = os.path.join(out_dir, f"boundary_{name}.png")
        plt.savefig(path, dpi=180, bbox_inches='tight',
                    facecolor=fig.get_facecolor())
        plt.close()
        print(f"  Figure: {path}")

    # ── Summary comparison figure ─────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.patch.set_facecolor("#0D1117")

    metric_keys  = ["hausdorff_mm", "msd_mm",      "volume_iou"]
    metric_names = ["Hausdorff Distance (mm)",
                    "Mean Surface Distance (mm)",
                    "Volume IoU"]
    better       = ["lower",        "lower",        "higher"]

    for ax, key, mname, direction in zip(axes, metric_keys, metric_names, better):
        ax.set_facecolor("#161B22")
        vals  = [r[key] for r in all_results]
        clrs  = [COLORS[r["name"]]["accent"] for r in all_results]
        bars  = ax.bar(names, vals, color=clrs, width=0.5, edgecolor='none')
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + max(vals)*0.02,
                    f"{val:.4f}", ha='center', va='bottom',
                    color='white', fontsize=8, fontweight='bold')
        ax.set_title(mname, color='white', fontsize=10, fontweight='bold', pad=8)
        ax.set_ylabel(mname, color='#8B949E', fontsize=8)
        ax.tick_params(colors='#8B949E', labelsize=8)
        for sp in ax.spines.values(): sp.set_color('#21262D')
        note = "↓ better" if direction == "lower" else "↑ better"
        ax.text(0.98, 0.97, note, ha='right', va='top',
                transform=ax.transAxes, fontsize=7, color='#8B949E',
                style='italic')
        ax.set_facecolor("#161B22")

    fig.suptitle(
        "WorldGen Physics Benchmark — Collision Proxy Boundary Accuracy\n"
        "BangLab · Université de Montréal · Mila",
        color='white', fontsize=12, fontweight='bold'
    )
    plt.tight_layout()

    path = os.path.join(out_dir, "boundary_summary.png")
    plt.savefig(path, dpi=180, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"\n  Summary figure: {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="WorldGen Mesh Boundary Accuracy Analysis"
    )
    parser.add_argument("--objects_dir", default="./objects")
    parser.add_argument("--output_dir",  default="./logs")
    parser.add_argument("--n_samples",   type=int, default=50000,
                        help="Surface sample count per mesh (default 50000)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    objects = [
        ("sphere",    "sphere.obj",    "sphere_col.obj"),
        ("bevel_box", "bevel_box.obj", "bevel_box_col.obj"),
        ("l_bracket", "l_bracket.obj", "l_bracket_col.obj"),
    ]

    print("=" * 60)
    print("  WorldGen Boundary Accuracy Analysis")
    print("  BangLab · Université de Montréal · Mila")
    print("=" * 60)

    all_results = []
    for name, vis_file, col_file in objects:
        vis_path = os.path.join(args.objects_dir, vis_file)
        col_path = os.path.join(args.objects_dir, col_file)
        if not os.path.exists(vis_path):
            print(f"  [SKIP] {vis_path} not found")
            continue
        res = analyze_object(name, vis_path, col_path, args.n_samples)
        all_results.append(res)

    if not all_results:
        print("No objects found. Run 01_generate_objects.py first.")
        sys.exit(1)

    # Strip internal arrays before JSON serialization
    for r in all_results:
        r.pop("_d_ab",    None)
        r.pop("_d_ba",    None)
        r.pop("_pts_vis", None)
        r.pop("_pts_col", None)

    # Re-analyze for plotting (need raw distances)
    plot_data = []
    for name, vis_file, col_file in objects:
        vis_path = os.path.join(args.objects_dir, vis_file)
        col_path = os.path.join(args.objects_dir, col_file)
        if not os.path.exists(vis_path):
            continue
        vis_mesh = trimesh.load(vis_path, force='mesh')
        col_mesh = trimesh.load(col_path, force='mesh')
        if not vis_mesh.is_watertight: vis_mesh = vis_mesh.convex_hull
        if not col_mesh.is_watertight: col_mesh = col_mesh.convex_hull
        pts_vis = sample_surface(vis_mesh, 30000)
        pts_col = sample_surface(col_mesh, 30000)
        h = hausdorff_metrics(pts_vis * 1000, pts_col * 1000)
        r = next(x for x in all_results if x["name"] == name)
        r["_d_ab"]    = h["d_ab"]
        r["_d_ba"]    = h["d_ba"]
        r["_pts_vis"] = pts_vis * 1000
        r["_pts_col"] = pts_col * 1000
        plot_data.append(r)

    print("\n[Generating figures...]")
    plot_results(plot_data, args.output_dir)

    # Save JSON report
    clean = [{k:v for k,v in r.items() if not k.startswith("_")}
             for r in plot_data]
    report_path = os.path.join(args.output_dir, "boundary_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(clean, f, indent=2)

    print(f"\n  JSON report: {report_path}")
    print("\n" + "=" * 60)
    print("  SUMMARY TABLE")
    print("=" * 60)
    print(f"  {'Object':<12} {'HD (mm)':>10} {'MSD (mm)':>10} "
          f"{'HD95 (mm)':>10} {'IoU':>8} {'SA err':>8}")
    print("  " + "-" * 58)
    for r in clean:
        print(f"  {r['name']:<12} "
              f"{r['hausdorff_mm']:>10.4f} "
              f"{r['msd_mm']:>10.4f} "
              f"{r['hd95_mm']:>10.4f} "
              f"{r['volume_iou']:>8.4f} "
              f"{r['surface_area_rel_error']*100:>7.2f}%")
    print("=" * 60)
    print("\n[Done]")


if __name__ == "__main__":
    main()
