import gymnasium as gym
import myosuite
import mujoco
import numpy as np


ENV_NAME = "myoHandPoseFixed-v0"


def main():

    env = gym.make(ENV_NAME)

    model = env.unwrapped.mj_model
    data = env.unwrapped.mj_data

    # ---------------------------------------------------------
    # Site IFtip
    # ---------------------------------------------------------

    site_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_SITE,
        "IFtip"
    )

    if site_id < 0:
        print("[ERROR] Site IFtip tidak ditemukan.")
        env.close()
        return

    print("=" * 80)
    print(" INDEX FINGER KINEMATICS TEST")
    print("=" * 80)

    print(f"IFtip site ID : {site_id}")

    # ---------------------------------------------------------
    # Initial qpos
    # ---------------------------------------------------------

    q0 = data.qpos.copy()

    mujoco.mj_forward(model, data)

    initial_pos = data.site_xpos[site_id].copy()

    print("\nInitial IFtip:")
    print(
        f"x = {initial_pos[0]: .6f}, "
        f"y = {initial_pos[1]: .6f}, "
        f"z = {initial_pos[2]: .6f}"
    )

    # ---------------------------------------------------------
    # Index qpos
    # ---------------------------------------------------------

    index_qpos = [
        7,   # mcp2_flexion
        8,   # mcp2_abduction
        9,   # pm2_flexion
        10   # md2_flexion
    ]

    index_names = [
        "mcp2_flexion",
        "mcp2_abduction",
        "pm2_flexion",
        "md2_flexion"
    ]

    # ---------------------------------------------------------
    # Test each joint
    # ---------------------------------------------------------

    print("\n" + "=" * 80)
    print(" JOINT RESPONSE")
    print("=" * 80)

    for q_index, name in zip(index_qpos, index_names):

        data.qpos[:] = q0

        # Tambahkan 10 derajat
        angle_deg = 10.0
        angle_rad = np.deg2rad(angle_deg)

        data.qpos[q_index] += angle_rad

        mujoco.mj_forward(model, data)

        pos = data.site_xpos[site_id].copy()

        delta = pos - initial_pos

        print(f"\n{name}")
        print(f"  qpos index : {q_index}")
        print(f"  angle      : {angle_deg:.1f} deg")

        print(
            f"  IFtip      : "
            f"({pos[0]: .6f}, "
            f"{pos[1]: .6f}, "
            f"{pos[2]: .6f})"
        )

        print(
            f"  delta      : "
            f"({delta[0]: .6f}, "
            f"{delta[1]: .6f}, "
            f"{delta[2]: .6f})"
        )

    # ---------------------------------------------------------
    # Restore
    # ---------------------------------------------------------

    data.qpos[:] = q0
    mujoco.mj_forward(model, data)

    env.close()

    print("\n" + "=" * 80)
    print(" TEST SELESAI")
    print("=" * 80)


if __name__ == "__main__":
    main()