import gymnasium as gym
import myosuite
import mujoco

ENV_NAME = "myoHandPoseFixed-v0"


def main():
    print("=" * 80)
    print(" MYOSUITE BODY INSPECTOR")
    print("=" * 80)

    env = gym.make(ENV_NAME)

    model = env.unwrapped.mj_model
    data = env.unwrapped.mj_data

    print(f"\nTotal bodies : {model.nbody}")

    print("\n" + "=" * 80)
    print(" BODY LIST")
    print("=" * 80)

    for body_id in range(model.nbody):

        name = mujoco.mj_id2name(
            model,
            mujoco.mjtObj.mjOBJ_BODY,
            body_id
        )

        parent_id = model.body_parentid[body_id]

        parent_name = mujoco.mj_id2name(
            model,
            mujoco.mjtObj.mjOBJ_BODY,
            parent_id
        )

        pos = data.xpos[body_id]

        print(
            f"Body {body_id:02d} | "
            f"{str(name):<25} | "
            f"parent={str(parent_name):<25} | "
            f"pos=({pos[0]: .4f}, {pos[1]: .4f}, {pos[2]: .4f})"
        )

    env.close()


if __name__ == "__main__":
    main()