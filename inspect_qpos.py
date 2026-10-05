import gymnasium as gym
import myosuite
import mujoco

ENV_NAME = "myoHandPoseFixed-v0"


def main():
    print("=" * 80)
    print(" MYOSUITE QPOS / JOINT INSPECTOR")
    print("=" * 80)

    env = gym.make(ENV_NAME)
    unwrapped = env.unwrapped

    model = unwrapped.mj_model
    data = unwrapped.mj_data

    print("\n[MODEL DIMENSIONS]")
    print(f"nq       : {model.nq}")
    print(f"nv       : {model.nv}")
    print(f"nu       : {model.nu}")
    print(f"nbody    : {model.nbody}")
    print(f"njnt     : {model.njnt}")
    print(f"ngeom    : {model.ngeom}")
    print(f"ntendon  : {model.ntendon}")

    print("\n" + "=" * 80)
    print(" JOINT -> QPOS MAPPING")
    print("=" * 80)

    qpos_index = 0

    for j in range(model.njnt):
        name = mujoco.mj_id2name(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            j
        )

        joint_type = model.jnt_type[j]
        qpos_start = model.jnt_qposadr[j]
        dof_start = model.jnt_dofadr[j]

        if joint_type == mujoco.mjtJoint.mjJNT_FREE:
            qpos_size = 7
        elif joint_type == mujoco.mjtJoint.mjJNT_BALL:
            qpos_size = 4
        else:
            qpos_size = 1

        print(
            f"Joint {j:02d} | "
            f"{name:<20} | "
            f"type={joint_type} | "
            f"qpos={qpos_start}..{qpos_start + qpos_size - 1} | "
            f"dof={dof_start}"
        )

    print("\n" + "=" * 80)
    print(" CURRENT QPOS")
    print("=" * 80)

    for i, value in enumerate(data.qpos):
        print(f"qpos[{i:02d}] = {value: .8f}")

    print("\n" + "=" * 80)
    print(" JOINT LIMITS")
    print("=" * 80)

    for j in range(model.njnt):
        name = mujoco.mj_id2name(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            j
        )

        limited = model.jnt_limited[j]
        joint_type = model.jnt_type[j]

        if joint_type == mujoco.mjtJoint.mjJNT_HINGE:
            if limited:
                low, high = model.jnt_range[j]
                print(
                    f"{j:02d} {name:<20} "
                    f"LIMITED  [{low:.6f}, {high:.6f}] rad "
                    f"({low * 180 / 3.14159265:.2f}°, "
                    f"{high * 180 / 3.14159265:.2f}°)"
                )
            else:
                print(f"{j:02d} {name:<20} UNLIMITED")

        elif joint_type == mujoco.mjtJoint.mjJNT_BALL:
            print(f"{j:02d} {name:<20} BALL JOINT")

        elif joint_type == mujoco.mjtJoint.mjJNT_FREE:
            print(f"{j:02d} {name:<20} FREE JOINT")

    print("\n" + "=" * 80)
    print(" INITIAL QPOS FROM ENV")
    print("=" * 80)

    print(unwrapped.init_qpos)

    env.close()


if __name__ == "__main__":
    main()