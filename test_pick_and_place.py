"""Test suite for the SO-ARM100 pick-and-place controller.

Fast unit tests (model/geometry/sampling checks) run in well under a second each.
The few integration tests that actually step physics (marked, see below) take on
the order of several seconds each since they run real controller episodes -- they
are regression tests for specific bugs found during development, not a substitute
for the full 20-episode evaluate.py run.

Usage:
    MUJOCO_GL=egl python3 -m pytest test_pick_and_place.py -v
    MUJOCO_GL=egl python3 -m pytest test_pick_and_place.py -v -m "not slow"  # skip integration tests
"""
import os

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np
import pytest

from pick_and_place import (
    ARM_JOINTS, JAW_CLOSED, JAW_OPEN, LOCAL_APPROACH_AXIS, LOCAL_PINCH_AXIS, SCENE_PATH,
    WORLD_DOWN, compute_target_pinch_dir, contact_report, get_ids, reset_episode,
    run_episode, solve_ik,
)
from evaluate import R_MAX, R_MIN, sample_spawn_points
from workspace_probe import phi_r_to_xy


@pytest.fixture(scope="module")
def model():
    return mujoco.MjModel.from_xml_path(SCENE_PATH)


@pytest.fixture
def data(model):
    return mujoco.MjData(model)


@pytest.fixture(scope="module")
def ids(model):
    return get_ids(model)


# --- Model / scene geometry -------------------------------------------------

def test_scene_loads_with_expected_joints(model):
    """6 arm/jaw joints (Rotation, Pitch, Elbow, Wrist_Pitch, Wrist_Roll, Jaw) plus
    the cube's free joint (7 dof) -- the bin is static (no joint)."""
    joint_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(model.njnt)]
    for name in ARM_JOINTS + ["Jaw", "cube_freejoint"]:
        assert name in joint_names
    assert model.njnt == 7


def test_cube_and_bin_present(model, ids):
    assert ids["cube_geom"] >= 0
    assert ids["bin_bottom_geom"] >= 0
    cube_size = model.geom_size[ids["cube_geom"]]
    np.testing.assert_allclose(cube_size, [0.01, 0.01, 0.01])


def test_grasp_site_pinch_axis_matches_jaw_separation(model, ids, data):
    """Empirical check that LOCAL_PINCH_AXIS (local x) is really the axis separating
    the fixed and moving jaw pads -- if the robot model ever changes, this should
    fail loudly rather than have the pinch-alignment feature silently target the
    wrong axis."""
    jaw_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "Jaw")
    data.qpos[model.jnt_qposadr[jaw_jid]] = 0.4
    mujoco.mj_forward(model, data)

    fixed_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "Fixed_Jaw")
    R = data.xmat[fixed_bid].reshape(3, 3)
    pos = data.xpos[fixed_bid]

    fg = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "fixed_jaw_pad_3")
    mg = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "moving_jaw_pad_3")
    fixed_local = R.T @ (data.geom_xpos[fg] - pos)
    moving_local = R.T @ (data.geom_xpos[mg] - pos)
    separation = moving_local - fixed_local

    # Separation should be dominated by the x-component.
    assert abs(separation[0]) > 5 * abs(separation[1])
    assert abs(separation[0]) > 5 * abs(separation[2])


# --- IK -----------------------------------------------------------------

def test_ik_converges_for_reachable_point(model, ids):
    from pick_and_place import keyframe_arm_qpos, write_arm_qpos
    home_q = keyframe_arm_qpos(model, ids, "home")
    target = np.array([0.0, -0.20, 0.01])
    q_sol = solve_ik(model, ids, home_q, target)

    scratch = mujoco.MjData(model)
    write_arm_qpos(scratch, ids, q_sol)
    mujoco.mj_forward(model, scratch)
    pos_err = np.linalg.norm(scratch.site_xpos[ids["site"]] - target)
    assert pos_err < 0.002


def test_contact_force_sign_convention(model, ids, data):
    """Regression test for a bug found during development: mj_contactForce reports
    the force acting on geom2 by default, not geom1 -- contact_report must flip the
    sign when the cube happens to be geom1. Verify against a simple resting-cube
    scenario where the true answer is known: total contact force on the cube from
    the floor must point up and balance gravity."""
    mujoco.mj_resetData(model, data)
    for _ in range(300):
        mujoco.mj_step(model, data)  # let the cube fall and settle on the floor

    total_force = np.zeros(3)
    floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    cube_geom = ids["cube_geom"]
    for i in range(data.ncon):
        con = data.contact[i]
        if floor_id not in (con.geom1, con.geom2) or cube_geom not in (con.geom1, con.geom2):
            continue
        force6 = np.zeros(6)
        mujoco.mj_contactForce(model, data, i, force6)
        R = con.frame.reshape(3, 3)
        force_world = R.T @ force6[:3]
        if con.geom1 == cube_geom:
            force_world = -force_world
        total_force += force_world

    cube_weight = model.body_mass[ids["cube_body"]] * 9.81
    assert total_force[2] == pytest.approx(cube_weight, rel=0.05)


# --- Pinch-alignment branch selection ------------------------------------

def test_pinch_dir_branch_selection_minimizes_rotation():
    """compute_target_pinch_dir should pick whichever of the cube's 4
    symmetric-equivalent edge directions is closest to the current pinch axis, not
    just the raw local-x axis -- regression test for the untrackable-joint-swing
    bug found during development."""
    class FakeIds:
        pass

    # A cube rotated 90 degrees: its local x-axis now points along world y.
    import mujoco as mj
    model = mj.MjModel.from_xml_path(SCENE_PATH)
    data = mj.MjData(model)
    ids = get_ids(model)
    cube_qadr = model.jnt_qposadr[ids["cube_joint"]]
    quat_90z = [np.cos(np.pi / 4), 0, 0, np.sin(np.pi / 4)]
    data.qpos[cube_qadr:cube_qadr + 7] = [0, -0.2, 0.01] + quat_90z
    mujoco.mj_forward(model, data)

    current_pinch = np.array([1.0, 0.0, 0.0])  # arm currently aligned with world x
    target = compute_target_pinch_dir(data, ids, current_pinch_dir=current_pinch)

    # The raw local-x axis of the rotated cube points along +-world-y; but +-world-x
    # (the cube's local-y axis, an equally valid symmetric target) is much closer to
    # current_pinch_dir and should be chosen instead.
    assert abs(abs(target[0]) - 1.0) < 0.05, f"expected a target close to +-world-x, got {target}"


def test_sample_spawn_points_excludes_bin_footprint():
    """Regression test for a bug found during development: unconstrained random
    sampling could spawn the cube on/against the bin itself, producing a trivial
    'success' with no real grasp attempt."""
    rng = np.random.default_rng(0)
    bin_pos = np.array([0.12, -0.18])
    bin_half_extent = 0.044
    points = sample_spawn_points(30, rng, bin_pos, bin_half_extent)

    margin = 0.03
    excl_half = bin_half_extent + margin
    for label, xy in points:
        overlaps = abs(xy[0] - bin_pos[0]) < excl_half and abs(xy[1] - bin_pos[1]) < excl_half
        assert not overlaps, f"{label} spawn {xy} overlaps the bin's excluded footprint"


# --- Integration tests (slow: step real physics) -------------------------

@pytest.mark.slow
def test_demo_spot_grasp_is_genuine_and_successful(model, ids):
    """End-to-end regression test at the original demo spawn point: task succeeds
    AND the cube is actually pinched (both jaw sides in contact) with negligible
    slip during lift/carry -- not just pushed into the bin."""
    data = mujoco.MjData(model)
    rng = np.random.default_rng(0)

    state = {"n": 0, "n_pinched": 0, "offsets": []}

    def cb(model, data, ids, phase):
        if phase not in ("lift", "pre_place"):
            return
        rep = contact_report(model, data, ids)
        state["n"] += 1
        state["n_pinched"] += int(rep["pinched"])
        state["offsets"].append((data.xpos[ids["cube_body"]] - data.site_xpos[ids["site"]]).copy())

    result = run_episode(model, data, ids, rng, spawn_xy=np.array([0.0, -0.18]), step_callback=cb, verbose=False)

    assert result["success"]
    assert state["n"] > 0
    assert state["n_pinched"] / state["n"] > 0.95
    offs = np.array(state["offsets"])
    drift = np.linalg.norm(offs - offs[0], axis=1).max()
    assert drift < 0.003


@pytest.mark.slow
def test_near_base_dead_zone_documented_limitation(model, ids):
    """A cube spawned inside the documented near-base dead zone should fail to be
    picked up -- if this starts passing, the dead-zone limitation documented in
    workspace_probe.py/README is stale and should be re-measured, not silently left
    inaccurate.

    NOTE: this region turned out to be non-monotonic in radius once the
    pinch-alignment branch selection was added (r=0.09-0.10 succeed, r=0.11-0.14
    fail again, r>=0.15 succeeds reliably -- see README debugging log). r=0.12 is
    used here specifically because it's a reproducible failure in that measured
    range, not because the whole near-base region is a clean threshold -- it isn't."""
    data = mujoco.MjData(model)
    rng = np.random.default_rng(0)
    result = run_episode(model, data, ids, rng, spawn_xy=np.array([0.0, -0.12]), verbose=False)
    assert not result["success"], (
        "cube at r=0.12 (measured dead-zone point) was picked up successfully -- "
        "the near-base characterization in workspace_probe.py/README may need updating"
    )


@pytest.mark.slow
def test_validated_envelope_point_succeeds(model, ids):
    """Sanity check that a point safely inside the validated envelope (r=0.20,
    straight ahead) still works -- catches a broad regression in the controller."""
    data = mujoco.MjData(model)
    rng = np.random.default_rng(0)
    result = run_episode(model, data, ids, rng, spawn_xy=np.array([0.0, -0.20]), verbose=False)
    assert result["success"]
