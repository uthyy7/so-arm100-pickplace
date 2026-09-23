"""Pick-and-place controller for the SO-ARM100.

Approach (see conversation for full rationale):
  - Task-level Cartesian waypoints (pre-grasp / grasp / lift / pre-place / release)
    are computed fresh each episode from the *live* cube and bin positions read out
    of the simulation state -- never from hardcoded coordinates.
  - Each waypoint's joint targets are solved with damped-least-squares (Jacobian)
    inverse kinematics against a `grasp_site` added to the Fixed_Jaw body, targeting
    both the site's position and a "point straight down" approach direction.
  - Waypoints are connected with a smoothstep-interpolated *Cartesian* trajectory (IK
    re-solved at several points along the straight-line path, ctrl blended between
    those closely-spaced joint solutions) fed to the position actuators and mj_step'd
    in real time -- plain joint-space interpolation let the gripper sweep through the
    table/cube between distant configurations, so the path is planned in task space.
  - The gripper (Jaw) is driven directly (open/close), independent of the arm IK.

Usage:
    MUJOCO_GL=egl python3 pick_and_place.py [--episodes N] [--seed S] [--render]
"""
import argparse
import os

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np

SCENE_PATH = "mujoco_menagerie/trs_so_arm100/scene.xml"

ARM_JOINTS = ["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll"]
SITE_NAME = "grasp_site"
# The grasp site sits in Fixed_Jaw's frame; its local -y axis points along the
# fingers toward the fingertips (see so_arm100.xml pad geoms), i.e. the direction
# that should point straight down at the object during a top-down grasp.
LOCAL_APPROACH_AXIS = np.array([0.0, -1.0, 0.0])
WORLD_DOWN = np.array([0.0, 0.0, -1.0])
# Local x is the pinch/gap axis -- the direction separating the fixed and moving jaw
# pads (confirmed empirically: their separation vector in Fixed_Jaw's frame is
# ~(-0.047, -0.001, 0.000), i.e. almost entirely along local x). Aligning this axis
# with the cube's own edge direction is what makes the pincer close on two flat
# faces instead of a corner.
LOCAL_PINCH_AXIS = np.array([1.0, 0.0, 0.0])

JAW_OPEN = 1.0
JAW_CLOSED = -0.174  # matches the Jaw joint's range minimum (fully closed)

# Task motion parameters (hand-picked, not derived from geometry).
PRE_GRASP_CLEARANCE = 0.06   # m above the cube center for the approach pose
LIFT_CLEARANCE = 0.08        # m above the cube center once grasped, to clear the bin walls
RELEASE_CLEARANCE = 0.02     # m above the bin floor when releasing

PHASE_DURATION = 1.2         # s, nominal time to interpolate an arm move
DWELL_GRASP = 0.5            # s, pause with jaw closing to let contact/friction settle
DWELL_RELEASE = 0.4          # s, pause after opening the jaw before retracting


def get_ids(model):
    ids = {}
    ids["site"] = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, SITE_NAME)
    ids["cube_body"] = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cube")
    ids["bin_body"] = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "bin")
    ids["cube_joint"] = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "cube_freejoint")
    ids["bin_bottom_geom"] = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "bin_bottom")
    ids["cube_geom"] = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "cube_geom")
    ids["fixed_pad_geoms"] = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"fixed_jaw_pad_{i}") for i in range(1, 5)]
    ids["moving_pad_geoms"] = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"moving_jaw_pad_{i}") for i in range(1, 5)]
    ids["jaw_actuator"] = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "Jaw")
    ids["arm_qposadr"] = {j: model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j)]
                           for j in ARM_JOINTS}
    ids["arm_dofadr"] = {j: model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j)]
                          for j in ARM_JOINTS}
    ids["arm_range"] = {j: model.jnt_range[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j)].copy()
                         for j in ARM_JOINTS}
    ids["arm_actuator"] = {j: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, j) for j in ARM_JOINTS}
    return ids


def read_arm_qpos(data, ids):
    return np.array([data.qpos[ids["arm_qposadr"][j]] for j in ARM_JOINTS])


def write_arm_qpos(data, ids, q):
    for j, val in zip(ARM_JOINTS, q):
        data.qpos[ids["arm_qposadr"][j]] = val


def keyframe_arm_qpos(model, ids, key_name):
    kid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, key_name)
    return np.array([model.key_qpos[kid][ids["arm_qposadr"][j]] for j in ARM_JOINTS])


def _solve_ik_once(model, ids, q_init, target_pos, target_dir, target_pinch_dir, pinch_weight,
                    max_iters, tol, damping):
    """One damped least-squares IK solve. See solve_ik for the weighting rationale."""
    scratch = mujoco.MjData(model)
    write_arm_qpos(scratch, ids, q_init)
    mujoco.mj_forward(model, scratch)

    site_id = ids["site"]
    dof_ids = [ids["arm_dofadr"][j] for j in ARM_JOINTS]
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))

    for _ in range(max_iters):
        site_pos = scratch.site_xpos[site_id]
        site_mat = scratch.site_xmat[site_id].reshape(3, 3)
        cur_dir = site_mat @ LOCAL_APPROACH_AXIS

        pos_err = target_pos - site_pos
        approach_err = np.cross(cur_dir, target_dir)  # drives cur_dir toward target_dir

        if np.linalg.norm(pos_err) < tol and np.linalg.norm(approach_err) < tol and target_pinch_dir is None:
            break

        mujoco.mj_jacSite(model, scratch, jacp, jacr, site_id)
        Jp, Ja = jacp[:, dof_ids], jacr[:, dof_ids]  # 3x5 each

        if target_pinch_dir is not None:
            cur_pinch = site_mat @ LOCAL_PINCH_AXIS
            pinch_err = np.cross(cur_pinch, target_pinch_dir)
            w = np.sqrt([1, 1, 1, 1, 1, 1, pinch_weight, pinch_weight, pinch_weight])
            J = np.vstack([Jp, Ja, Ja]) * w[:, None]
            err = np.concatenate([pos_err, approach_err, pinch_err]) * w
        else:
            J = np.vstack([Jp, Ja])
            err = np.concatenate([pos_err, approach_err])

        JJt = J @ J.T
        dq = J.T @ np.linalg.solve(JJt + (damping ** 2) * np.eye(J.shape[0]), err)

        q = read_arm_qpos(scratch, ids) + dq
        for k, j in enumerate(ARM_JOINTS):
            lo, hi = ids["arm_range"][j]
            q[k] = np.clip(q[k], lo, hi)
        write_arm_qpos(scratch, ids, q)
        mujoco.mj_forward(model, scratch)

    return read_arm_qpos(scratch, ids)


def solve_ik(model, ids, q_init, target_pos, target_dir=WORLD_DOWN, target_pinch_dir=None,
             pinch_weight=0.3, max_pos_cost=0.003, max_iters=200, tol=1e-4, damping=0.05):
    """Damped least-squares IK for the grasp_site: position + approach-direction,
    optionally also biasing the pinch axis toward `target_pinch_dir` (a horizontal
    unit vector) so the jaws tend to close on two flat faces of the cube instead of
    a corner.

    This arm has exactly 5 joints. Position (3) + approach-direction (effectively 2,
    since roll about the approach axis is free) already uses all 5 -- adding a third
    orientation constraint (pinch alignment) makes it a 6-target problem on 5 DOF,
    which is generically *not* exactly solvable. A plain hard constraint (tried
    first) measurably hurt position accuracy, and a fixed soft weight alone still
    wasn't safe: at some configurations even a tiny pinch weight is enough to drive
    a joint (typically Wrist_Roll) straight into its range limit, because the
    unconstrained solution was already sitting close to that limit -- the trade-off
    is a sharp cliff, not a smooth dial, so no single fixed weight is safe
    everywhere.

    So this solves BOTH ways from the same warm start -- unconstrained, and
    pinch-weighted (row-scaled damped least squares, weight `pinch_weight`) -- and
    only takes the pinch-aligned solution if it doesn't cost more than
    `max_pos_cost` (default 3mm) of extra position error versus the unconstrained
    solve; otherwise it falls back to the unconstrained one. This makes pinch
    alignment a strict best-effort improvement: it should never make position
    accuracy meaningfully worse.

    Runs against scratch MjData objects so the live simulation state is untouched.
    Returns the solved 5-vector of arm joint angles (clamped to joint ranges).
    """
    q_uncon = _solve_ik_once(model, ids, q_init, target_pos, target_dir, None, None,
                              max_iters, tol, damping)
    if target_pinch_dir is None:
        return q_uncon

    q_pinch = _solve_ik_once(model, ids, q_init, target_pos, target_dir, target_pinch_dir, pinch_weight,
                              max_iters, tol, damping)

    def pos_error(q):
        scratch = mujoco.MjData(model)
        write_arm_qpos(scratch, ids, q)
        mujoco.mj_forward(model, scratch)
        return np.linalg.norm(scratch.site_xpos[ids["site"]] - target_pos)

    if pos_error(q_pinch) <= pos_error(q_uncon) + max_pos_cost:
        return q_pinch
    return q_uncon


def contact_report(model, data, ids):
    """Inspect the live contact list for cube <-> jaw-pad contacts and report whether
    the cube is genuinely pinched (contact from both the fixed and moving jaw sides
    simultaneously) versus just resting against one surface, plus the total contact
    force in the world frame (its vertical component is what would have to balance
    gravity if the cube is being held rather than momentarily coasting)."""
    cube_geom = ids["cube_geom"]
    fixed_pads = set(ids["fixed_pad_geoms"])
    moving_pads = set(ids["moving_pad_geoms"])

    fixed_contact = False
    moving_contact = False
    total_force_world = np.zeros(3)
    n_contacts = 0

    for i in range(data.ncon):
        con = data.contact[i]
        g1, g2 = con.geom1, con.geom2
        if cube_geom not in (g1, g2):
            continue
        other = g2 if g1 == cube_geom else g1
        if other in fixed_pads:
            fixed_contact = True
        elif other in moving_pads:
            moving_contact = True
        else:
            continue  # contact with floor/bin, not the gripper

        force6 = np.zeros(6)
        mujoco.mj_contactForce(model, data, i, force6)
        R = con.frame.reshape(3, 3)  # rows are the contact frame axes, expressed in world coords
        force_world = R.T @ force6[:3]
        # mj_contactForce reports the force acting on geom2; flip when the cube is
        # geom1 so the result is consistently "force on the cube".
        if g1 == cube_geom:
            force_world = -force_world
        total_force_world += force_world
        n_contacts += 1

    return {
        "pinched": fixed_contact and moving_contact,
        "n_contacts": n_contacts,
        "total_force_world": total_force_world,
    }


def _solve_ik_chain(model, ids, q_start, pos_start, pos_end, target_pinch_dir, n_ik_waypoints):
    """Solve IK at n_ik_waypoints points along a straight Cartesian line, each
    warm-started from the previous -- a single continuous chain (no per-step
    accept/reject), so it can't develop a mid-path branch jump on its own."""
    q_prev = q_start
    q_waypoints = [q_start]
    for i in range(1, n_ik_waypoints + 1):
        s = i / n_ik_waypoints
        s = 3 * s ** 2 - 2 * s ** 3  # smoothstep, applied to the Cartesian path
        pos = pos_start + s * (pos_end - pos_start)
        q_prev = _solve_ik_once(model, ids, q_prev, pos, WORLD_DOWN, target_pinch_dir,
                                 0.3 if target_pinch_dir is not None else None, 200, 1e-4, 0.05)
        q_waypoints.append(q_prev)
    return q_waypoints


def run_trajectory(model, data, ids, q_start, pos_start, pos_end, jaw_start, jaw_end, duration,
                    target_pinch_dir=None, max_pos_cost=0.003, n_ik_waypoints=40,
                    renderer=None, frames=None, step_callback=None):
    """Move the gripper from pos_start to pos_end along a straight Cartesian line,
    not a joint-space blend: linear joint interpolation between two valid but
    distant configurations doesn't keep the end-effector on a straight path, and in
    practice let the gripper dip down and sweep through the table/cube mid-move. So
    instead we solve IK at `n_ik_waypoints` points along the line (each warm-started
    from the previous) and interpolate ctrl between those closely-spaced solutions.

    If `target_pinch_dir` is given, TWO full chains are solved -- one unconstrained,
    one pinch-weighted -- and we pick whichever *whole chain* to use by comparing
    their final-waypoint position error (falling back to unconstrained if the
    pinch-weighted chain's final error is more than `max_pos_cost` worse). This
    decision is made once per call, not per sub-waypoint: making it independently at
    every sub-waypoint (tried first) let the accept/reject flip partway through a
    move, splicing together two different IK branches into one path with a
    discontinuous jump in the middle -- which is far more disruptive than either
    chain alone, since the actuators then have to cover a large joint-space gap in
    one tiny sub-segment's time slice.

    Returns the final solved joint targets.
    """
    n_steps = max(1, int(duration / model.opt.timestep))
    n_ik_waypoints = min(n_ik_waypoints, n_steps)

    q_waypoints = _solve_ik_chain(model, ids, q_start, pos_start, pos_end, None, n_ik_waypoints)
    if target_pinch_dir is not None:
        q_pinch_waypoints = _solve_ik_chain(model, ids, q_start, pos_start, pos_end, target_pinch_dir, n_ik_waypoints)

        def final_pos_error(chain):
            scratch = mujoco.MjData(model)
            write_arm_qpos(scratch, ids, chain[-1])
            mujoco.mj_forward(model, scratch)
            return np.linalg.norm(scratch.site_xpos[ids["site"]] - pos_end)

        if final_pos_error(q_pinch_waypoints) <= final_pos_error(q_waypoints) + max_pos_cost:
            q_waypoints = q_pinch_waypoints

    steps_per_segment = n_steps // n_ik_waypoints
    step_count = 0
    for seg in range(1, len(q_waypoints)):
        qA, qB = q_waypoints[seg - 1], q_waypoints[seg]
        seg_steps = steps_per_segment if seg < len(q_waypoints) - 1 else (n_steps - step_count)
        for j in range(seg_steps):
            t = (j + 1) / seg_steps
            q = qA + t * (qB - qA)
            overall_t = (step_count + 1) / n_steps
            jaw = jaw_start + overall_t * (jaw_end - jaw_start)
            for jn, val in zip(ARM_JOINTS, q):
                data.ctrl[ids["arm_actuator"][jn]] = val
            data.ctrl[ids["jaw_actuator"]] = jaw
            mujoco.mj_step(model, data)
            step_count += 1
            if step_callback is not None:
                step_callback(model, data, ids)

    if renderer is not None and frames is not None:
        renderer.update_scene(data, camera=frames["cam"])
        frames["list"].append(renderer.render().copy())

    return q_waypoints[-1]


def dwell(model, data, ids, q, jaw, duration, step_callback=None):
    n_steps = max(1, int(duration / model.opt.timestep))
    for j, val in zip(ARM_JOINTS, q):
        data.ctrl[ids["arm_actuator"][j]] = val
    data.ctrl[ids["jaw_actuator"]] = jaw
    for _ in range(n_steps):
        mujoco.mj_step(model, data)
        if step_callback is not None:
            step_callback(model, data, ids)


def reset_episode(model, data, ids, rng, spawn_xy=None):
    """Reset arm to rest pose and drop the cube at a spawn point, then settle it
    under gravity. Returns nothing -- caller reads live positions from `data`
    afterward.

    `spawn_xy`: explicit (x, y) to drop the cube at (for edge-of-workspace or
    evaluator sampling). If None, a small random jitter around the original demo
    spot is used, as before.
    """
    mujoco.mj_resetData(model, data)
    rest_q = keyframe_arm_qpos(model, ids, "rest")
    write_arm_qpos(data, ids, rest_q)

    jaw_qid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "Jaw")
    jaw_qadr = model.jnt_qposadr[jaw_qid]
    data.qpos[jaw_qadr] = JAW_OPEN

    cube_qadr = model.jnt_qposadr[ids["cube_joint"]]
    if spawn_xy is None:
        spawn_xy = np.array([0.0, -0.18]) + rng.uniform(-0.02, 0.02, size=2)
    data.qpos[cube_qadr:cube_qadr + 7] = [spawn_xy[0], spawn_xy[1], 0.03, 1, 0, 0, 0]

    for j, val in zip(ARM_JOINTS, rest_q):
        data.ctrl[ids["arm_actuator"][j]] = val
    data.ctrl[ids["jaw_actuator"]] = JAW_OPEN

    mujoco.mj_forward(model, data)
    for _ in range(300):  # let the cube fall and settle
        mujoco.mj_step(model, data)


def compute_target_pinch_dir(data, ids, current_pinch_dir=None):
    """Read the cube's live orientation and return a horizontal unit vector along
    one of its edge directions, for the jaw's pinch axis to align with. The cube is
    a symmetric square prism, so its 4 horizontal edge directions (its local x-axis,
    and the 90/180/270-degree rotations of it, i.e. +-ex and +-ey) are all equally
    valid grasp targets -- a top-down parallel-jaw grasp doesn't care which of the 4
    it lands on.

    That choice is not just cosmetic: picking whichever of the 4 is closest to
    `current_pinch_dir` (the site's pinch axis *before* any motion, e.g. at the rest
    pose) minimizes the rotation the wrist actually has to swing through to satisfy
    the constraint. Picking the "wrong" branch (e.g. the raw local-x axis when local
    -x was actually closer to where the arm already is) can demand a 90-180 degree
    Wrist_Roll swing that's kinematically valid but not trackable within one phase's
    duration -- this is what caused the "IK converges perfectly, but the actuators
    can't get there in time" failures seen when the branch wasn't chosen. Without a
    `current_pinch_dir` reference, the raw local x-axis is used as before.
    """
    R_cube = data.xmat[ids["cube_body"]].reshape(3, 3)
    ex, ey = R_cube[:, 0].copy(), R_cube[:, 1].copy()
    ex[2] = ey[2] = 0.0

    candidates = []
    for axis in (ex, ey):
        norm = np.linalg.norm(axis)
        if norm > 1e-6:
            candidates.extend([axis / norm, -axis / norm])
    if not candidates:
        # Both local x and y are (nearly) vertical -- cube resting on a side/corner
        # rather than a face. Fall back to no pinch constraint in this degenerate case.
        return None

    if current_pinch_dir is None:
        return candidates[0]
    return max(candidates, key=lambda c: np.dot(c, current_pinch_dir))


def build_waypoints(model, ids, cube_pos, bin_pos):
    cube_half_h = cube_pos[2]  # cube resting on the floor => center height == half-thickness
    bin_floor_top = bin_pos[2] + 2 * model.geom_size[ids["bin_bottom_geom"]][2]

    return [
        ("pre_grasp", cube_pos + np.array([0, 0, PRE_GRASP_CLEARANCE]), JAW_OPEN, PHASE_DURATION, 0.0),
        ("grasp", cube_pos, JAW_CLOSED, PHASE_DURATION, DWELL_GRASP),
        ("lift", cube_pos + np.array([0, 0, LIFT_CLEARANCE]), JAW_CLOSED, PHASE_DURATION, 0.0),
        ("pre_place", np.array([bin_pos[0], bin_pos[1], cube_pos[2] + LIFT_CLEARANCE]), JAW_CLOSED, PHASE_DURATION, 0.0),
        ("release", np.array([bin_pos[0], bin_pos[1], bin_floor_top + cube_half_h + RELEASE_CLEARANCE]),
         JAW_OPEN, PHASE_DURATION, DWELL_RELEASE),
        ("retract", np.array([bin_pos[0], bin_pos[1], cube_pos[2] + LIFT_CLEARANCE]), JAW_OPEN, PHASE_DURATION, 0.0),
    ]


def run_episode(model, data, ids, rng, renderer=None, frames=None, spawn_xy=None,
                 step_callback=None, verbose=True):
    """`step_callback`, if given, is called as step_callback(model, data, ids, phase)
    after every physics step, where `phase` is the current waypoint's name (e.g.
    "grasp", "lift") or "<name>_dwell" during that waypoint's post-move dwell -- so
    callers can restrict logging/checks to specific phases (e.g. only while the jaw
    is actually supposed to be closed) instead of averaging over the whole episode,
    including phases where the gripper is intentionally open."""
    reset_episode(model, data, ids, rng, spawn_xy=spawn_xy)

    cube_pos = data.xpos[ids["cube_body"]].copy()
    bin_pos = data.xpos[ids["bin_body"]].copy()
    if verbose:
        print(f"  live cube pos: {cube_pos}")
        print(f"  live bin pos:  {bin_pos}")

    waypoints = build_waypoints(model, ids, cube_pos, bin_pos)
    # Read the cube's orientation once, before anything touches it, and hold the
    # jaw's pinch axis aligned to it for the whole episode -- so the approach
    # already comes in correctly rolled instead of twisting into alignment right at
    # the moment of contact (which could just as easily knock the cube aside).
    # Pick whichever of the cube's 4 symmetric-equivalent edge directions is closest
    # to the arm's current pinch axis (at the rest pose, before any motion), so the
    # constraint asks for the smallest wrist swing rather than an arbitrary branch.
    current_pinch_dir = data.site_xmat[ids["site"]].reshape(3, 3) @ LOCAL_PINCH_AXIS
    target_pinch_dir = compute_target_pinch_dir(data, ids, current_pinch_dir=current_pinch_dir)
    if verbose:
        print(f"  target pinch dir: {target_pinch_dir}")

    q = read_arm_qpos(data, ids)
    pos = data.site_xpos[ids["site"]].copy()
    jaw = JAW_OPEN

    if renderer is not None and frames is not None:
        renderer.update_scene(data, camera=frames["cam"])
        frames["list"].append(renderer.render().copy())

    for name, target_pos, jaw_target, duration, dwell_time in waypoints:
        phase_cb = (lambda m, d, i, _name=name: step_callback(m, d, i, _name)) if step_callback else None
        run_trajectory(model, data, ids, q, pos, target_pos, jaw, jaw_target, duration,
                        target_pinch_dir=target_pinch_dir, renderer=renderer, frames=frames, step_callback=phase_cb)
        # Track the arm's *actual* simulated state (contact/force limits can make it
        # diverge from the commanded IK target) for warm-starting the next phase.
        q = read_arm_qpos(data, ids)
        pos = data.site_xpos[ids["site"]].copy()
        jaw = jaw_target
        if dwell_time > 0:
            dwell_cb = (lambda m, d, i, _name=name: step_callback(m, d, i, _name + "_dwell")) if step_callback else None
            dwell(model, data, ids, q, jaw, dwell_time, step_callback=dwell_cb)
            q = read_arm_qpos(data, ids)
            pos = data.site_xpos[ids["site"]].copy()
            if renderer is not None and frames is not None:
                renderer.update_scene(data, camera=frames["cam"])
                frames["list"].append(renderer.render().copy())
        if verbose:
            print(f"  reached {name}")

    final_cube = data.xpos[ids["cube_body"]].copy()
    final_vel = np.linalg.norm(data.qvel[model.jnt_dofadr[ids["cube_joint"]]:model.jnt_dofadr[ids["cube_joint"]] + 3])

    bin_half = model.geom_size[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "bin_bottom")][:2]
    in_xy = np.all(np.abs(final_cube[:2] - bin_pos[:2]) < bin_half)
    bin_floor_top = bin_pos[2] + 2 * model.geom_size[ids["bin_bottom_geom"]][2]
    in_z = final_cube[2] < bin_floor_top + 0.05
    settled = final_vel < 0.02
    success = bool(in_xy and in_z and settled)

    if verbose:
        print(f"  final cube pos: {final_cube}, vel: {final_vel:.4f}")
        print(f"  SUCCESS: {success}")
    return {"success": success, "final_cube_pos": final_cube, "final_vel": final_vel,
            "cube_pos": cube_pos, "bin_pos": bin_pos}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--render", action="store_true", help="save a per-phase contact sheet image")
    args = parser.parse_args()

    model = mujoco.MjModel.from_xml_path(SCENE_PATH)
    data = mujoco.MjData(model)
    ids = get_ids(model)
    rng = np.random.default_rng(args.seed)

    renderer = None
    successes = 0
    for ep in range(args.episodes):
        print(f"Episode {ep}:")
        frames = None
        if args.render:
            if renderer is None:
                renderer = mujoco.Renderer(model, height=480, width=640)
            cam = mujoco.MjvCamera()
            mujoco.mjv_defaultFreeCamera(model, cam)
            cam.lookat = np.array([0.06, -0.18, 0.05])
            cam.distance = 0.55
            cam.azimuth = 135
            cam.elevation = -30
            frames = {"cam": cam, "list": []}

        result = run_episode(model, data, ids, rng, renderer, frames)
        successes += result["success"]

        if args.render and frames["list"]:
            from PIL import Image
            imgs = frames["list"]
            cols = min(4, len(imgs))
            rows = (len(imgs) + cols - 1) // cols
            h, w, _ = imgs[0].shape
            sheet = np.full((rows * h, cols * w, 3), 255, dtype=np.uint8)
            for i, im in enumerate(imgs):
                r, c = divmod(i, cols)
                sheet[r * h:(r + 1) * h, c * w:(c + 1) * w] = im
            out_path = f"pick_place_episode_{ep}.png"
            Image.fromarray(sheet).save(out_path)
            print(f"  saved {out_path}")

    print(f"\n{successes}/{args.episodes} episodes succeeded")


if __name__ == "__main__":
    main()
