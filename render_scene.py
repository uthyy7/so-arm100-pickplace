"""Render a single frame of the scene to an image file for a visual sanity check.

Usage:
    MUJOCO_GL=egl python3 render_scene.py [output_path]
"""
import os
import sys

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np
from PIL import Image

SCENE_PATH = "mujoco_menagerie/trs_so_arm100/scene.xml"
DEFAULT_OUTPUT = "scene_check.png"


def main():
    output_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_OUTPUT

    model = mujoco.MjModel.from_xml_path(SCENE_PATH)
    data = mujoco.MjData(model)

    # Apply the "rest" keyframe for the arm joints. Note: that keyframe predates the
    # cube's free joint, so MuJoCo zero-pads the extra dofs rather than using the
    # cube body's declared pos/quat -- we have to set the cube's qpos explicitly.
    mujoco.mj_resetDataKeyframe(model, data, mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "rest"))
    cube_joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "cube_freejoint")
    qadr = model.jnt_qposadr[cube_joint]
    data.qpos[qadr : qadr + 7] = [0, -0.18, 0.01, 1, 0, 0, 0]

    # Settle the cube onto the floor under gravity so it isn't shown mid-air.
    mujoco.mj_forward(model, data)
    for _ in range(200):
        mujoco.mj_step(model, data)

    renderer = mujoco.Renderer(model, height=720, width=960)
    renderer.update_scene(data, camera=-1)  # -1 = default free camera
    # Aim the free camera at the workspace where the cube and bin sit.
    renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 1

    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(model, cam)
    cam.lookat = np.array([0.06, -0.18, 0.05])
    cam.distance = 0.55
    cam.azimuth = 135
    cam.elevation = -30
    renderer.update_scene(data, camera=cam)

    img = renderer.render()
    Image.fromarray(img).save(output_path)
    print(f"Saved render to {output_path}")

    cube_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cube")
    bin_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "bin")
    print(f"cube pos: {data.xpos[cube_body]}")
    print(f"bin pos:  {data.xpos[bin_body]}")


if __name__ == "__main__":
    main()
