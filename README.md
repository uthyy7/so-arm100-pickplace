SO-ARM100 Pick-and-Place (MuJoCo)
Overview

Scripted pick-and-place controller for the SO-ARM100 5-DOF arm. It picks up a 2cm cube from a random table position and places it in a fixed bin. Cube and bin positions are read live from the sim each episode, not hardcoded. The grasp is verified with contact force, not just final position. 18/20 genuine grasps over a 20-episode evaluation, more on this in the Key finding section.

python3 -m venv venv && source venv/bin/activate
pip install mujoco numpy pillow matplotlib pytest

python3 check_joints.py                                   # sanity check
MUJOCO_GL=egl python3 render_scene.py                      # render scene and cube/bin placement
MUJOCO_GL=egl python3 pick_and_place.py --episodes 2 --seed 1 --render
MUJOCO_GL=egl python3 diagnose_grasp.py                    # verify grasp is real contact, not luck
MUJOCO_GL=egl python3 workspace_probe.py                   # map and test workspace edges
MUJOCO_GL=egl python3 evaluate.py --n 20 --seed 0           # 20-episode evaluator
MUJOCO_GL=egl python3 -m pytest test_pick_and_place.py -v
MUJOCO_GL=egl is needed for offscreen rendering since there is no display in WSL.

Approach

It's a state machine with six waypoints per episode, pre_grasp, grasp, lift, pre_place, release, retract. I picked the waypoint positions myself, everything else is computed.

IK. Damped least-squares Jacobian IK against a grasp_site on Fixed_Jaw. Solves for position plus pointing straight down. Also softly biased to line up with the cube's edge so the jaw grabs flat faces instead of a corner.

Trajectory. Re-solves IK at around 40 points along a straight Cartesian line between waypoints, instead of interpolating joint angles directly. I found out why this matters the hard way, see entry 3 below.

Gripper. Open and close, separate from the arm.

Cube alignment. The arm only has 5 joints, and position plus top-down orientation already uses all of them. So alignment is a second candidate solution, only kept if it doesn't cost much position accuracy, entries 6 and 7. The cube also has 4 equally valid edges to align to, so I pick whichever needs the smallest wrist turn, entry 8.

Debugging log

Roughly in the order things happened.

WSL and GitHub auth. npm install failed. Turned out Ubuntu's default npm global path is root owned. Fixed with sudo. Separately git clone kept asking for a password and failing. Fixed by installing the GitHub CLI, running gh auth login through the browser, then gh auth setup-git.
Cube position not applying from the keyframe. The cube didn't settle at its intended spot. The keyframe was made before the cube's free joint existed, so MuJoCo zero pads instead of using the declared pose. Fixed by setting the cube's qpos directly after applying the keyframe.
Linear joint-space interpolation broke the trajectory. The arm swung through a contorted pose and knocked the cube away before reaching it. A straight line in joint angles isn't a straight line for the gripper in real space. Fixed by re-solving IK along the actual Cartesian path instead.
Wrong assumption, Wrist_Roll doesn't move the gripper. Caught this before it became a bug, checked it myself first instead of trusting the plan. I'd assumed the grasp site sat on the roll axis so twisting wouldn't move it. That was wrong, it moves about 1.2cm sideways. Meant the alignment fix had to be solved together with position, not added afterward.
Contact force sign was backwards. The measured upward force came out negative during a static hold, which had to be wrong since something has to cancel gravity. mj_contactForce reports force on geom2 by default, and my flip condition had it backwards. Fixed it, and the force then matched the cube's actual weight almost exactly, 0.0392N.
Hard alignment constraint overloaded the arm. My first alignment attempt made things worse, 19/20 down to 14/20. Position plus pointing down already uses all 5 joints, so adding a third requirement left nothing spare, the solver gave up accuracy elsewhere to satisfy it. Fixed by making alignment optional, only used when it doesn't cost much accuracy.
Switching between two solutions mid-move caused jumps. Even with the optional alignment, some episodes still had big tracking errors. The code was deciding aligned or not separately at each small step along a move, so it could switch mid-way and jump between two different solutions. Fixed by deciding once for the whole move instead of step by step.
Picked a bad rotation angle that needed an impossible swing. Sometimes the IK solution was mathematically perfect but needed the base joint to swing nearly 100 degrees to get there, more than the arm could do in time. The cube has 4 sides that are all equally fine to grab, but the code always picked the same one regardless of where the arm already was. Fixed by picking whichever side needs the smallest turn.
Remaining failures were a bug in my own testing code, not the arm. After fixing entry 8, 3 episodes still showed no real grip. Traced one and found the cube was already touching the bin floor on the very first frame, before the arm even moved. My evaluator was placing some cubes randomly inside the bin itself. Fixed by excluding the bin's footprint from where cubes can spawn. So 4 real fixes to the controller, and this last one wasn't the controller's fault at all.
That last fix made a clean failure zone messy. A test expecting the arm to fail at a close in position started passing instead. Checked a range of close positions and found it's no longer a clean cutoff, it passes and fails in a patchwork rather than a straight line. Left this alone since my evaluator only tests positions well outside that zone anyway. Updated the test to use a position that still reliably fails. Something to look into later, not fixed.
Key finding

My naive check, does the cube end up in the bin, said 19/20. But that check can be fooled, something can land in the right spot by being shoved rather than picked up. I added a real check, both jaw pads in contact, force matching the cube's weight, no slip, and the real number was 12/20. The cause was that IK never constrained jaw rotation relative to the cube, so it sometimes closed on a corner. After fixing that, entries 6 through 9, the genuine rate went to 18/20. One failure, episode 6 at r=0.258m, was there from the start and is unrelated to any of this.

AI tools used

Claude chat for planning, explaining concepts I hadn't used before like Jacobians and MuJoCo/menagerie, and WSL/GitHub setup.

Claude Code for most of the implementation, sanity check, scene, controller, diagnostics, evaluator, tests, this README.

Where it was wrong

First trajectory interpolated joint angles directly, which let the arm sweep through the cube, entry 3.
Assumed Wrist_Roll didn't move the gripper. Wrong, only caught because I asked it to check first, entry 4.
Contact force sign was backwards until I noticed the number was physically impossible, entry 5.
First alignment fix made the evaluator worse, and I had to point out the regression before it found the real cause, entry 6.
Introduced a second bug while fixing the first one, and the actual remaining cause was in my own evaluator, not the controller. Only found because I asked it to trace contacts instead of guessing again, entry 9.
Tests

MUJOCO_GL=egl python3 -m pytest test_pick_and_place.py -v, 10 tests total.

Fast, 7 of them, under a second total. Model and joint checks, cube and bin geometry, the pinch-axis assumption, IK convergence, the contact-force sign regression, the branch-selection regression, and the bin-footprint exclusion regression.

Slow, 3 of them, around 35 seconds total, real physics. The demo spot succeeds and is a genuine grasp, a dead-zone point fails, a validated-range point succeeds.

pytest -m "not slow" skips the physics tests for a quicker check.

Time spent

Went well past the suggested 6 hours. The core build was roughly in that range, but the grasp-verification work turned up a real chain of issues, entries 4 through 10, that felt worth chasing down properly rather than leaving half diagnosed.

Status

Controller works, grasp verified genuine. Tested across the workspace including the edges. 20-episode eval, 19/20 task success, 18/20 genuine grasp. Tests and this README are done.

Still open, not blocking anything

Episode 6 at r=0.258m fails and hasn't been root caused. Not a regression, it was there from the first baseline run.
The near-base region, entry 10, isn't fully characterized. The evaluator's range stays clear of it, so it doesn't affect the reported numbers, but I don't fully understand it yet.