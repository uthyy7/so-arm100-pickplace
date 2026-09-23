SO-ARM100 Pick-and-Place (MuJoCo)
Overview

Scripted (not RL) pick-and-place controller for the SO-ARM100 5-DOF arm. Picks up a 2cm cube from a random table position, places it in a fixed bin. Cube/bin positions are read live from the sim each episode, not hardcoded. Grasp is verified with contact force, not just final position. 18/20 genuine grasps over a 20-episode evaluation (see Key finding).

How to run
bash
python3 -m venv venv && source venv/bin/activate
pip install mujoco numpy pillow matplotlib pytest

python3 check_joints.py                                  # sanity check
MUJOCO_GL=egl python3 render_scene.py                     # render scene + cube/bin placement
MUJOCO_GL=egl python3 pick_and_place.py --episodes 2 --seed 1 --render
MUJOCO_GL=egl python3 diagnose_grasp.py                    # verify grasp is real contact, not luck
MUJOCO_GL=egl python3 workspace_probe.py                   # map + test workspace edges
MUJOCO_GL=egl python3 evaluate.py --n 20 --seed 0           # 20-episode evaluator
MUJOCO_GL=egl python3 -m pytest test_pick_and_place.py -v

MUJOCO_GL=egl needed for offscreen rendering (no display in WSL).

Approach

State machine, six waypoints per episode: pre_grasp, grasp, lift, pre_place, release, retract. I picked the waypoint positions myself, everything else is computed.

IK: damped least-squares Jacobian IK against a grasp_site on Fixed_Jaw. Solves for position + pointing straight down. Also softly biased to line up with the cube's edge so the jaw grabs flat faces, not a corner.
Trajectory: re-solve IK at ~40 points along a straight Cartesian line between waypoints, not interpolated joint angles. Found out why this matters the hard way (entry 3).
Gripper: open/close, separate from the arm.
Cube alignment: arm only has 5 joints, position + top-down orientation already uses all of them, so alignment is a second candidate solution, only kept if it doesn't cost much position accuracy (entries 6-7). Cube has 4 equally-valid edges to align to, pick whichever needs the smallest wrist turn (entry 8).
Debugging log

Roughly chronological.

1. WSL / GitHub auth npm install -g failed with EACCES. Ubuntu's default npm global path is root-owned. Fixed with sudo. Separately, git clone kept asking for a password and failing (GitHub dropped password auth in 2021, and I sign in with Google so had nothing to give it). Fixed with gh auth login (browser-based) then gh auth setup-git.

2. Cube position not applying from keyframe Cube didn't settle at its intended spot. Keyframe predates the cube's free joint, so MuJoCo zero-pads instead of using the declared pose. Fixed by setting cube qpos directly after applying the keyframe.

3. Linear joint-space interpolation broke the trajectory Arm swung through a contorted pose and knocked the cube away before reaching it. A straight line in joint angles isn't a straight line for the gripper in real space. Fixed by re-solving IK along the actual Cartesian path instead.

4. Wrong assumption: Wrist_Roll doesn't move the gripper Caught this before it became a bug, checked empirically first. Assumed the grasp site sat on the roll axis so twisting wouldn't move it. Wrong: it moves ~1.2cm sideways. Meant the alignment fix had to be solved jointly with position, not bolted on after.

5. Contact force sign was backwards Measured upward force came out negative during a static hold, which is impossible: something has to cancel gravity. mj_contactForce reports force on geom2 by default, my flip condition had it backwards. Fixed, then force matched the cube's actual weight almost exactly (0.0392N).

6. Hard alignment constraint over-constrained the arm First alignment attempt (hard IK constraint) made things worse, 19/20 → 14/20. Position + pointing down already uses all 5 joints, adding a 3rd constraint makes it unsolvable exactly, so the solver traded off position accuracy. Fixed by making it a soft term, only used if it doesn't cost much accuracy.

7. Splicing two IK branches mid-path Even with the soft constraint, some episodes had huge tracking error. The accept/reject decision was per sub-waypoint, so it could flip mid-path and stitch together two different solutions with a jump in the middle. Fixed by deciding once per whole phase.

8. Untrackable swing from an arbitrary alignment angle IK would converge perfectly to a target that needed a ~99° base joint swing from the current pose, and the actuators couldn't cover it in time. Cube has 4 equally-valid edges to align to; code always used the same one regardless of arm position. Fixed by picking whichever needs the smallest swing.

9. Remaining failures were in my evaluator, not the controller After fixing #8, 3 episodes still showed zero pinch contact. Traced one and found the cube touching the bin floor on step one, before the arm moved at all. Evaluator's random sampling didn't exclude the bin's own footprint, so those cubes started already in the bin. Fixed by excluding the bin footprint from spawn sampling. Four real controller-side fixes (4, 6, 7, 8), and the last symptom was actually a test harness bug.

10. Fixing #8 turned a clean dead zone into a messy one A test expecting failure at r=0.10 started passing. Re-swept r=0.05–0.15: not a clean cutoff anymore, it's patchy. Didn't chase further since my evaluator's range starts at r=0.16, well clear of it. Test updated to check a reliable failure point instead. Open follow-up.

Key finding

Naive check (cube ends up in bin) said 19/20. That check can be fooled: something can land in the right spot by being shoved rather than picked up. Added a real check (both jaw pads in contact, force matching the cube's weight, no slip) and the real number was 12/20. Cause: IK never constrained jaw rotation relative to the cube, so it sometimes closed on a corner. Fixed (entries 6-9), genuine rate went to 18/20. One failure (episode 6, r=0.258m) predates this work, unrelated.

AI tools used

Claude (chat) for planning, explaining concepts I hadn't used before (Jacobians, MuJoCo/menagerie), and WSL/GitHub setup. Claude Code for most of the implementation: sanity check, scene, controller, diagnostics, evaluator, tests, this README.

Where it was wrong:

First trajectory interpolated joint angles directly, let the arm sweep through the cube (entry 3)
Assumed Wrist_Roll didn't move the gripper. Wrong, only caught because I asked it to check first (entry 4)
Contact force sign backwards until I noticed the number was physically impossible (entry 5)
First alignment fix made the evaluator worse, had to point out the regression before it found the real cause (entry 6)
Introduced a second bug fixing the first, and the actual remaining cause was in my own evaluator, not the controller, only found by asking it to trace contacts instead of guessing again (entry 9)
Tests

MUJOCO_GL=egl python3 -m pytest test_pick_and_place.py -v, 10 tests. Fast (7, <1s): model/joint checks, cube+bin geometry, pinch-axis assumption, IK convergence, contact-force sign regression, branch-selection regression, bin-footprint exclusion regression. Slow (3, ~35s, real physics): demo spot succeeds + is a genuine grasp, dead-zone point fails, validated-range point succeeds. pytest -m "not slow" skips the physics tests.

Time spent

Went well past the suggested 6 hours. Core build was roughly in range; the grasp-verification work turned up a real chain of issues (entries 4-10) worth chasing down properly rather than leaving half-diagnosed.

Status

Controller works, grasp verified genuine. Tested across the workspace including edges. 20-episode eval: 19/20 task success, 18/20 genuine grasp. Tests and README done.

Open, not blocking:

Episode 6 (r=0.258m) fails, not root-caused, predates this work
Near-base region (entry 10) not fully characterized. Evaluator's range stays clear of it, but I don't fully understand it yet