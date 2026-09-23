import mujoco

model = mujoco.MjModel.from_xml_path("mujoco_menagerie/trs_so_arm100/scene.xml")

print(f"Number of joints: {model.njnt}")
for i in range(model.njnt):
    print(f"  {i}: {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)}")
