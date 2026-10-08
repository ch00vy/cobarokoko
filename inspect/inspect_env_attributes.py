import gymnasium as gym
import myosuite


ENV_NAME = "myoHandPoseFixed-v0"


def main():

    print("=" * 80)
    print(" MYOSUITE ENVIRONMENT ATTRIBUTE INSPECTOR")
    print("=" * 80)

    env = gym.make(ENV_NAME)

    print("\nEnvironment:")
    print(type(env))

    print("\nUnwrapped:")
    print(type(env.unwrapped))

    print("\n" + "=" * 80)
    print(" ENVIRONMENT ATTRIBUTES")
    print("=" * 80)

    attrs = dir(env.unwrapped)

    for name in attrs:

        if not name.startswith("_"):
            print(name)

    print("\n" + "=" * 80)
    print(" POSSIBLE MODEL / SIMULATION ATTRIBUTES")
    print("=" * 80)

    keywords = [
        "model",
        "data",
        "sim",
        "physics",
        "mujoco",
        "robot",
        "hand",
    ]

    for name in attrs:

        name_lower = name.lower()

        if any(keyword in name_lower for keyword in keywords):

            try:
                value = getattr(env.unwrapped, name)

                print(
                    f"{name:<25} "
                    f"type={type(value)}"
                )

            except Exception as e:

                print(
                    f"{name:<25} "
                    f"ERROR: {e}"
                )

    env.close()


if __name__ == "__main__":
    main()