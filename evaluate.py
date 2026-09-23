"""20-episode pick-and-place evaluator.

Spawns the cube at a mix of explicit workspace-edge points and randomly sampled
points across the *validated* operating envelope (see workspace_probe.py and the
"near-base dead zone" note below), runs the full controller, and for each episode
checks both:
  - task success: cube ends up resting inside the bin
  - grasp quality: genuine pinch contact + no slippage during lift/carry (the same
    check as diagnose_grasp.py, run inline per episode)

Known limitation (found via workspace_probe.py): cube positions within about
r < 0.145m of the base (straight ahead; the safe radius is smaller off-axis) sit in
a near-singular zone for this 5-DOF arm where the wrist can't reliably reach a full
top-down orientation in the allotted phase duration, even though a pure kinematic
IK solve (given enough iterations) can technically converge there. That zone is
excluded from the sampling envelope below rather than silently failing 20 episodes
against a known issue -- see the summary printed at the end for the exact bound used.

Usage:
    MUJOCO_GL=egl python3 evaluate.py [--n 20] [--seed 0]
"""
import argparse
import json
import os

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np

from pick_and_place import contact_report, get_ids, run_episode, SCENE_PATH
from workspace_probe import phi_r_to_xy

# Validated safe envelope (see module docstring / workspace_probe.py findings).
R_MIN = 0.16
R_MAX = 0.28
PHI_MIN_DEG = -80
PHI_MAX_DEG = 80


def make_grasp_quality_tracker():
    """Returns (step_callback, get_result). Tracks pinch fraction and max relative-
    offset drift, restricted to the "lift" and "pre_place" phases (post-grasp,
    pre-release) -- not the whole episode, which would dilute pinch fraction with
    phases where the jaw is intentionally open, and drift with the cube's expected
    large displacement while the jaw is still closing around it during "grasp"."""
    state = {"n": 0, "n_pinched": 0, "offsets": []}
    TRACKED_PHASES = ("lift", "pre_place")

    def cb(model, data, ids, phase):
        if phase not in TRACKED_PHASES:
            return
        rep = contact_report(model, data, ids)
        state["n"] += 1
        state["n_pinched"] += int(rep["pinched"])
        site_pos = data.site_xpos[ids["site"]]
        cube_p = data.xpos[ids["cube_body"]]
        state["offsets"].append((cube_p - site_pos).copy())

    def get_result():
        if state["n"] == 0:
            return {"pinch_frac": 0.0, "max_drift_mm": None}
        offs = np.array(state["offsets"])
        drift = np.linalg.norm(offs - offs[0], axis=1).max()
        return {"pinch_frac": state["n_pinched"] / state["n"], "max_drift_mm": drift * 1000}

    return cb, get_result


BIN_EXCLUSION_MARGIN = 0.03  # m, beyond the bin's own footprint -- room for the
                             # cube's half-size plus gripper clearance, so a spawn
                             # point can't land the cube on/against the bin itself
                             # (found via a contact trace: 3 evaluator "successes"
                             # in an earlier run turned out to be spawns landing
                             # directly on the bin's bottom plate, so the cube was
                             # never actually picked up at all -- see conversation).


def sample_spawn_points(n, rng, bin_pos, bin_half_extent):
    """4 explicit edge points + (n-4) random draws across the safe envelope,
    excluding any point that would land the cube on/against the bin itself."""
    excl_half = bin_half_extent + BIN_EXCLUSION_MARGIN

    def overlaps_bin(xy):
        return (abs(xy[0] - bin_pos[0]) < excl_half) and (abs(xy[1] - bin_pos[1]) < excl_half)

    points = [
        ("edge:max_r_center", phi_r_to_xy(R_MAX, 0)),
        ("edge:max_r_left", phi_r_to_xy(R_MAX, PHI_MAX_DEG)),
        ("edge:max_r_right", phi_r_to_xy(R_MAX, PHI_MIN_DEG)),
        ("edge:min_r_center", phi_r_to_xy(R_MIN, 0)),
    ]
    assert not any(overlaps_bin(xy) for _, xy in points), "an edge point overlaps the bin -- adjust R_MIN/R_MAX/bin placement"

    n_edge = len(points)
    n_random = max(0, n - n_edge)
    attempts = 0
    while len(points) < n_edge + n_random and attempts < n_random * 50:
        attempts += 1
        r = rng.uniform(R_MIN, R_MAX)
        phi = rng.uniform(PHI_MIN_DEG, PHI_MAX_DEG)
        xy = phi_r_to_xy(r, phi)
        if overlaps_bin(xy):
            continue
        points.append((f"random:r={r:.3f},phi={phi:.1f}", xy))
    return points[:n]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    model = mujoco.MjModel.from_xml_path(SCENE_PATH)
    data = mujoco.MjData(model)
    ids = get_ids(model)
    rng = np.random.default_rng(args.seed)

    # model.body_pos (not data.xpos, which needs a forward pass first) -- valid here
    # since the bin has no joint, so its world position is always this fixed value.
    bin_pos = model.body_pos[ids["bin_body"]][:2].copy()
    bin_half_extent = model.geom_size[ids["bin_bottom_geom"]][0]  # square footprint, x==y half-extent
    spawn_points = sample_spawn_points(args.n, rng, bin_pos, bin_half_extent)

    results = []
    for i, (label, xy) in enumerate(spawn_points):
        cb, get_grasp_result = make_grasp_quality_tracker()
        result = run_episode(model, data, ids, rng, spawn_xy=xy, step_callback=cb, verbose=False)
        grasp = get_grasp_result()
        row = {
            "episode": i,
            "label": label,
            "spawn_xy": [float(v) for v in xy],
            "success": bool(result["success"]),
            "final_cube_pos": [float(v) for v in result["final_cube_pos"]],
            "pinch_frac": float(grasp["pinch_frac"]),
            "max_drift_mm": float(grasp["max_drift_mm"]) if grasp["max_drift_mm"] is not None else None,
        }
        results.append(row)
        status = "PASS" if row["success"] else "FAIL"
        drift_str = f"{row['max_drift_mm']:.2f}mm" if row["max_drift_mm"] is not None else "n/a"
        print(f"[{i:2d}] {label:28s} spawn=({xy[0]:+.3f},{xy[1]:+.3f})  {status}  "
              f"pinch={grasp['pinch_frac']*100:5.1f}%  drift={drift_str}")

    n_success = int(sum(r["success"] for r in results))
    n_genuine_grasp = int(sum(r["success"] and r["pinch_frac"] > 0.95 and
                               (r["max_drift_mm"] is not None and r["max_drift_mm"] < 3.0) for r in results))

    print(f"\n=== Summary ===")
    print(f"Sampling envelope: r in [{R_MIN}, {R_MAX}] m, phi in [{PHI_MIN_DEG}, {PHI_MAX_DEG}] deg "
          f"(near-base dead zone r<~0.145m excluded, see docstring)")
    print(f"Task success:        {n_success}/{len(results)}")
    print(f"Genuine grasp cases: {n_genuine_grasp}/{len(results)} (success AND pinch>95% AND drift<3mm)")

    failures = [r for r in results if not r["success"]]
    if failures:
        print("\nFailures:")
        for r in failures:
            print(f"  [{r['episode']}] {r['label']} spawn={r['spawn_xy']} final={r['final_cube_pos']}")

    with open("evaluation_results.json", "w") as f:
        json.dump({"results": results, "n_success": n_success, "n_genuine_grasp": n_genuine_grasp,
                   "envelope": {"r_min": R_MIN, "r_max": R_MAX, "phi_min_deg": PHI_MIN_DEG, "phi_max_deg": PHI_MAX_DEG}},
                  f, indent=2)
    print("\nsaved evaluation_results.json")

    # Scatter plot: spawn points colored by success, for visual confirmation.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 6))
    for r in results:
        x, y = r["spawn_xy"]
        color = "tab:green" if r["success"] else "tab:red"
        marker = "o" if r["label"].startswith("random") else "s"
        ax.scatter(x, y, c=color, marker=marker, s=80, edgecolors="k", zorder=3)
    bin_pos = data.xpos[ids["bin_body"]]
    ax.scatter(bin_pos[0], bin_pos[1], c="tab:blue", marker="*", s=200, label="bin", zorder=4)
    ax.scatter(0, 0, c="black", marker="+", s=100, label="base", zorder=4)
    ax.add_patch(plt.Circle((0, 0), R_MIN, fill=False, linestyle="--", color="gray", label=f"r_min={R_MIN}"))
    ax.add_patch(plt.Circle((0, 0), R_MAX, fill=False, linestyle="--", color="gray", label=f"r_max={R_MAX}"))
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_aspect("equal")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title(f"Evaluation: {n_success}/{len(results)} succeeded (circle=edge case, dot=random)")
    fig.tight_layout()
    fig.savefig("evaluation_workspace.png", dpi=120)
    print("saved evaluation_workspace.png")


if __name__ == "__main__":
    main()
