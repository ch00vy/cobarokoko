import gymnasium as gym
import myosuite
import mujoco
import numpy as np

ENV_NAME = "myoHandPoseFixed-v0"


def print_body_positions(model, data):
    print("\nBODY POSITIONS")
    print("-" * 80)

    body_names = [
        "hand",
        "thumb",
        "index",
        "middle",
        "ring",
        "pinky",
    ]

    for name in body_names:
        body_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_BODY,
            name
        )

        if body_id >= 0:
            pos = data.xpos[body_id]
            print(
                f"{name:<10} "
                f"x={pos[0]: .4f} "
                f"y={pos[1]: .4f} "
                f"z={pos[2]: .4f}"
            )


def test_joint(env, joint_index, angle_deg):
    model = env.unwrapped.mj_model
    data = env.unwrapped.mj_data

    # Reset qpos
    data.qpos[:] = env.unwrapped.init_qpos
    data.qvel[:] = 0

    # Set target joint
    angle_rad = np.deg2rad(angle_deg)
    data.qpos[joint_index] = angle_rad

    # Forward kinematics
    mujoco.mj_forward(model, data)

    print("\n" + "=" * 80)
    print(
        f"TEST qpos[{joint_index}] = "
        f"{angle_deg} degrees ({angle_rad:.4f} rad)"
    )
    print("=" * 80)

    print(f"qpos[{joint_index}] = {data.qpos[joint_index]:.6f}")

    print_body_positions(model, data)


def main():

    print("=" * 80)
    print(" MYOSUITE QPOS MOVEMENT TEST")
    print("=" * 80)

    env = gym.make(ENV_NAME)

    model = env.unwrapped.mj_model

    print(f"\nEnvironment : {ENV_NAME}")
    print(f"nq          : {model.nq}")
    print(f"njnt        : {model.njnt}")

    # ==========================================================
    # TEST JOINT-JOINT UTAMA
    # ==========================================================

    tests = [
        (0, 30),    # wrist pronation/supination
        (1, 10),    # wrist deviation
        (2, 20),    # wrist flexion

        (3, 20),    # thumb CMC abduction
        (4, 20),    # thumb CMC flexion
        (5, 20),    # thumb MP flexion
        (6, 20),    # thumb IP flexion

        (7, 30),    # index MCP flexion
        (8, 10),    # index MCP abduction
        (9, 30),    # index proximal
        (10, 30),   # index distal

        (11, 30),   # middle MCP
        (12, 10),   # middle abduction
        (13, 30),   # middle proximal
        (14, 30),   # middle distal

        (15, 30),   # ring MCP
        (16, 10),   # ring abduction
        (17, 30),   # ring proximal
        (18, 30),   # ring distal

        (19, 30),   # pinky MCP
        (20, 10),   # pinky abduction
        (21, 30),   # pinky proximal
        (22, 30),   # pinky distal
    ]

    for joint_index, angle in tests:
        test_joint(env, joint_index, angle)

    env.close()

    print("\n" + "=" * 80)
    print("TEST SELESAI")
    print("=" * 80)


if __name__ == "__main__":
    main()