import gymnasium as gym
import myosuite
import mujoco


ENV_NAME = "myoHandPoseFixed-v0"


def main():

    print("=" * 80)
    print(" MYOSUITE JOINT / QPOS INSPECTOR")
    print("=" * 80)

    env = gym.make(ENV_NAME)

    # MyoSuite 2.12.2
    model = env.unwrapped.mj_model
    data = env.unwrapped.mj_data

    print(f"\n[OK] Model : {type(model)}")
    print(f"[OK] Data  : {type(data)}")

    print("\n" + "=" * 80)
    print(" MODEL DIMENSIONS")
    print("=" * 80)

    print(f"nq    : {model.nq}")
    print(f"nv    : {model.nv}")
    print(f"njnt  : {model.njnt}")
    print(f"nbody : {model.nbody}")

    print("\n" + "=" * 80)
    print(" JOINT LIST")
    print("=" * 80)

    for j in range(model.njnt):

        name = mujoco.mj_id2name(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            j
        )

        joint_type = model.jnt_type[j]

        qpos_start = model.jnt_qposadr[j]
        qvel_start = model.jnt_dofadr[j]

        body_id = model.jnt_bodyid[j]

        body_name = mujoco.mj_id2name(
            model,
            mujoco.mjtObj.mjOBJ_BODY,
            body_id
        )

        print(
            f"Joint {j:02d} | "
            f"{str(name):25s} | "
            f"type={joint_type} | "
            f"qpos={qpos_start:02d} | "
            f"qvel={qvel_start:02d} | "
            f"body={body_name}"
        )

    print("\n" + "=" * 80)
    print(" QPOS INITIAL")
    print("=" * 80)

    for i in range(model.nq):
        print(
            f"qpos[{i:02d}] = "
            f"{data.qpos[i]: .6f}"
        )

    print("\n" + "=" * 80)
    print(" QPOS JOINT LIMITS")
    print("=" * 80)

    for j in range(model.njnt):

        name = mujoco.mj_id2name(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            j
        )

        qpos_start = model.jnt_qposadr[j]

        limited = model.jnt_limited[j]

        if limited:

            low, high = model.jnt_range[j]

            low_deg = low * 180.0 / 3.141592653589793
            high_deg = high * 180.0 / 3.141592653589793

            print(
                f"{name:25s} | "
                f"qpos[{qpos_start:02d}] | "
                f"range=[{low:.6f}, {high:.6f}] rad | "
                f"[{low_deg:.2f}, {high_deg:.2f}] deg"
            )

        else:

            print(
                f"{name:25s} | "
                f"qpos[{qpos_start:02d}] | "
                f"range=UNLIMITED"
            )

    env.close()


if __name__ == "__main__":
    main()