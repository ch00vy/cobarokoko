import gymnasium as gym
import myosuite
import mujoco

ENV_NAME = "myoHandPoseFixed-v0"


def main():
    print("=" * 80)
    print(" MYOSUITE SITE INSPECTOR")
    print("=" * 80)

    env = gym.make(ENV_NAME)

    model = env.unwrapped.mj_model
    data = env.unwrapped.mj_data

    print(f"\nTotal sites : {model.nsite}")

    print("\n" + "=" * 80)
    print(" SITE LIST")
    print("=" * 80)

    for site_id in range(model.nsite):

        name = mujoco.mj_id2name(
            model,
            mujoco.mjtObj.mjOBJ_SITE,
            site_id
        )

        body_id = model.site_bodyid[site_id]

        body_name = mujoco.mj_id2name(
            model,
            mujoco.mjtObj.mjOBJ_BODY,
            body_id
        )

        pos = data.site_xpos[site_id]

        print(
            f"Site {site_id:02d} | "
            f"{str(name):<25} | "
            f"body={str(body_name):<20} | "
            f"pos=({pos[0]: .4f}, "
            f"{pos[1]: .4f}, "
            f"{pos[2]: .4f})"
        )

    env.close()


if __name__ == "__main__":
    main()