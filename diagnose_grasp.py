"""Verify the cube is genuinely held by jaw-pad contact/friction during lift and
carry, not just resting on top of the closed jaw and being carried along by luck.

Two independent signals are logged at every physics step from grasp-close through
the end of pre_place (the lateral carry move, which also stress-tests the grip
against lateral acceleration, not just gravity):

  1. "pinched": is there simultaneous contact between the cube and a FIXED-side pad
     AND a MOVING-side pad? That means the cube is squeezed between both jaw
     surfaces (a real pinch), as opposed to sitting on/against only one of them.
  2. relative offset (cube center - grasp_site position): if the cube is rigidly
     held by friction, this should stay essentially constant through lift/carry. A
     cube that's just balanced or slipping would show this drifting.

Usage:
    MUJOCO_GL=egl python3 diagnose_grasp.py
"""
import os

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np

from pick_and_place import (
    ARM_JOINTS, JAW_OPEN, SCENE_PATH,
    build_waypoints, contact_report, dwell, get_ids, read_arm_qpos, reset_episode,
    run_trajectory,
)

CUBE_MASS_DENSITY = 500  # matches scene.xml cube density, for a weight sanity check


def main():
    model = mujoco.MjModel.from_xml_path(SCENE_PATH)
    data = mujoco.MjData(model)
    ids = get_ids(model)
    rng = np.random.default_rng(0)

    reset_episode(model, data, ids, rng)
    cube_pos = data.xpos[ids["cube_body"]].copy()
    bin_pos = data.xpos[ids["bin_body"]].copy()
    cube_mass = model.body_mass[ids["cube_body"]]
    cube_weight = cube_mass * 9.81
    print(f"cube mass: {cube_mass*1000:.2f} g, weight: {cube_weight:.4f} N")

    waypoints = build_waypoints(model, ids, cube_pos, bin_pos)

    log = {"phase": [], "t": [], "pinched": [], "n_contacts": [], "force_world": [],
           "rel_offset": []}

    def make_callback(phase_name):
        def cb(model, data, ids):
            rep = contact_report(model, data, ids)
            site_pos = data.site_xpos[ids["site"]]
            cube_p = data.xpos[ids["cube_body"]]
            log["phase"].append(phase_name)
            log["t"].append(data.time)
            log["pinched"].append(rep["pinched"])
            log["n_contacts"].append(rep["n_contacts"])
            log["force_world"].append(rep["total_force_world"].copy())
            log["rel_offset"].append((cube_p - site_pos).copy())
        return cb

    q = read_arm_qpos(data, ids)
    pos = data.site_xpos[ids["site"]].copy()
    jaw = JAW_OPEN

    # Run pre_grasp normally (no logging needed -- gripper is open, nothing to check yet).
    name, target_pos, jaw_target, duration, dwell_time = waypoints[0]
    assert name == "pre_grasp"
    run_trajectory(model, data, ids, q, pos, target_pos, jaw, jaw_target, duration)
    q, pos, jaw = read_arm_qpos(data, ids), data.site_xpos[ids["site"]].copy(), jaw_target

    # Log everything from "grasp" (jaw closing) through "pre_place" (the lateral carry).
    for name, target_pos, jaw_target, duration, dwell_time in waypoints[1:4]:
        run_trajectory(model, data, ids, q, pos, target_pos, jaw, jaw_target, duration,
                        step_callback=make_callback(name))
        q, pos, jaw = read_arm_qpos(data, ids), data.site_xpos[ids["site"]].copy(), jaw_target
        if dwell_time > 0:
            dwell(model, data, ids, q, jaw, dwell_time, step_callback=make_callback(name + "_dwell"))
        print(f"  reached {name}, cube z={data.xpos[ids['cube_body']][2]:.4f}")

    phases = np.array(log["phase"])
    pinched = np.array(log["pinched"])
    rel_offset = np.array(log["rel_offset"])
    force_world = np.array(log["force_world"])
    t = np.array(log["t"])

    print("\n--- Per-phase report ---")
    lift_start_offset = None
    all_ok = True
    for phase in ["grasp", "grasp_dwell", "lift", "pre_place"]:
        mask = phases == phase
        if not mask.any():
            continue
        frac_pinched = pinched[mask].mean()
        offs = rel_offset[mask]
        drift = np.linalg.norm(offs - offs[0], axis=1).max()
        mean_force_z = force_world[mask][:, 2].mean()
        print(f"{phase:14s}: pinched {frac_pinched*100:5.1f}% of steps, "
              f"max offset drift {drift*1000:.2f} mm, mean upward contact force {mean_force_z:.4f} N")
        if phase in ("lift", "pre_place"):
            if lift_start_offset is None:
                lift_start_offset = offs[0]
            drift_from_grasp = np.linalg.norm(offs - lift_start_offset, axis=1).max()
            phase_ok = frac_pinched > 0.95 and drift_from_grasp < 0.003
            print(f"{'':14s}  -> drift from grasp moment: {drift_from_grasp*1000:.2f} mm "
                  f"{'OK' if phase_ok else 'FAIL'} (need pinched>95% and drift<3mm)")
            all_ok = all_ok and phase_ok

    print(f"\nGenuine friction/contact grasp confirmed: {all_ok}")

    # Plot for visual confirmation.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    axes[0].plot(t, pinched.astype(int), drawstyle="steps-post")
    axes[0].set_ylabel("pinched (both sides)")
    axes[0].set_ylim(-0.1, 1.1)

    # Drift relative to the moment the grasp was established (start of "lift"), which
    # is what the pass/fail check above uses -- not drift from the very start of the
    # log, which would also count the jaw closing in on the cube during "grasp".
    offset_mag = np.linalg.norm(rel_offset - lift_start_offset, axis=1) * 1000
    axes[1].plot(t, offset_mag)
    axes[1].axhline(3.0, color="r", linestyle="--", label="3mm threshold")
    axes[1].set_ylabel("cube-site offset drift\nsince grasp established (mm)")
    axes[1].legend()

    force_mag = np.linalg.norm(force_world, axis=1)
    axes[2].plot(t, force_mag, label="|total contact force|")
    axes[2].axhline(cube_weight, color="g", linestyle="--", label="cube weight")
    axes[2].set_ylabel("force (N)")
    axes[2].set_xlabel("time (s)")
    axes[2].legend()

    # Mark phase boundaries.
    for ax in axes:
        prev = None
        for i, ph in enumerate(phases):
            if ph != prev:
                ax.axvline(t[i], color="gray", linestyle=":", linewidth=0.7)
                prev = ph

    fig.suptitle("Grasp verification: contact + relative-offset over grasp/lift/carry")
    fig.tight_layout()
    fig.savefig("grasp_diagnosis.png", dpi=120)
    print("saved grasp_diagnosis.png")


if __name__ == "__main__":
    main()
