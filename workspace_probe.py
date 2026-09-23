"""Map the reachable top-down-grasp workspace and physically test cube spawns at
its edges (not just the original demo spot near x=0, y=-0.18).

Stage 1 (fast, kinematic only): sweep a polar grid of candidate cube positions
around the base and check whether solve_ik converges to a valid top-down grasp
pose for each -- this maps the envelope cheaply, without running physics.

Stage 2 (slow, physical): run full pick-and-place episodes (real contact dynamics,
same fixed bin) with the cube spawned at a handful of points at/near the edge of
that envelope, to see whether the controller actually succeeds there, not just
whether IK has a solution.

Usage:
    MUJOCO_GL=egl python3 workspace_probe.py
"""
import os

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np

from pick_and_place import (
    ARM_JOINTS, SCENE_PATH, get_ids, keyframe_arm_qpos, run_episode, solve_ik,
)

CUBE_Z = 0.01  # nominal resting height, matches the cube's half-thickness
R_RANGE = np.arange(0.04, 0.34, 0.02)
PHI_RANGE_DEG = np.arange(-100, 101, 5)  # measured from straight-ahead (-y), matches Rotation's ~+-110 deg range


def phi_r_to_xy(r, phi_deg):
    phi = np.radians(phi_deg)
    return np.array([r * np.sin(phi), -r * np.cos(phi)])


def kinematic_sweep(model, ids):
    home_q = keyframe_arm_qpos(model, ids, "home")
    reachable = np.zeros((len(R_RANGE), len(PHI_RANGE_DEG)), dtype=bool)

    for i, r in enumerate(R_RANGE):
        for j, phi in enumerate(PHI_RANGE_DEG):
            xy = phi_r_to_xy(r, phi)
            target = np.array([xy[0], xy[1], CUBE_Z])
            q_sol = solve_ik(model, ids, home_q, target, max_iters=300, tol=5e-4)
            # Check actual residual (solve_ik doesn't report convergence directly).
            scratch = mujoco.MjData(model)
            for k, jn in enumerate(ARM_JOINTS):
                scratch.qpos[ids["arm_qposadr"][jn]] = q_sol[k]
            mujoco.mj_forward(model, scratch)
            err = np.linalg.norm(scratch.site_xpos[ids["site"]] - target)
            reachable[i, j] = err < 0.003

    return reachable


def print_envelope(reachable):
    print("\nReachable envelope (rows = radius, cols = angle from straight-ahead):")
    header = "      " + "".join(f"{p:5d}" for p in PHI_RANGE_DEG[::4])
    print(header)
    for i, r in enumerate(R_RANGE):
        row = "".join(" X   " if reachable[i, j] else " .   " for j in range(0, len(PHI_RANGE_DEG), 4))
        print(f"r={r:.2f} {row}")

    # For each angle, find min/max reachable radius.
    print("\nPer-angle reachable radius range:")
    edge_points = []
    for j, phi in enumerate(PHI_RANGE_DEG):
        col = reachable[:, j]
        if not col.any():
            continue
        idxs = np.where(col)[0]
        r_min, r_max = R_RANGE[idxs[0]], R_RANGE[idxs[-1]]
        if phi % 20 == 0:
            print(f"  phi={phi:5.0f} deg: r in [{r_min:.2f}, {r_max:.2f}]")
        edge_points.append((phi, r_min, r_max))
    return edge_points


def main():
    model = mujoco.MjModel.from_xml_path(SCENE_PATH)
    data = mujoco.MjData(model)
    ids = get_ids(model)

    print("Stage 1: kinematic reachability sweep...")
    reachable = kinematic_sweep(model, ids)
    edge_points = print_envelope(reachable)

    phis = np.array([e[0] for e in edge_points])
    r_maxs = np.array([e[2] for e in edge_points])
    r_mins = np.array([e[1] for e in edge_points])

    # Pick physical test points: max radius straight ahead, max radius at the two
    # extreme angles tested, min radius, and one far corner (max angle + large r).
    candidates = {
        "max_r_center": phi_r_to_xy(r_maxs[np.argmin(np.abs(phis))], 0),
        "max_r_left": phi_r_to_xy(r_maxs[np.argmax(phis)], phis[np.argmax(phis)]),
        "max_r_right": phi_r_to_xy(r_maxs[np.argmin(phis)], phis[np.argmin(phis)]),
        "min_r_center": phi_r_to_xy(r_mins[np.argmin(np.abs(phis))], 0),
    }

    print("\nStage 2: physical pick-and-place tests at workspace-edge spawn points...")
    rng = np.random.default_rng(0)
    results = {}
    for name, xy in candidates.items():
        print(f"\n[{name}] spawn xy = {xy}")
        result = run_episode(model, data, ids, rng, spawn_xy=xy, verbose=True)
        results[name] = result["success"]

    print("\n--- Edge-of-workspace summary ---")
    for name, ok in results.items():
        print(f"  {name}: {'PASS' if ok else 'FAIL'}")


if __name__ == "__main__":
    main()
