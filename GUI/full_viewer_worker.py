import importlib
import queue
from multiprocessing.queues import Queue

import gymnasium as gym
import mujoco
import mujoco.viewer


def run_full_viewer(
    state_queue: Queue,
    status_queue: Queue,
) -> None:
    env = None
    viewer = None
    try:
        importlib.import_module("myosuite")
        env = gym.make("myoHandPoseFixed-v0")
        unwrapped = env.unwrapped
        model = getattr(unwrapped, "mj_model")
        data = getattr(unwrapped, "mj_data")
        data.qpos[:] = getattr(unwrapped, "init_qpos")
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)  # type: ignore[attr-defined]
        viewer = mujoco.viewer.launch_passive(
            model,
            data,
            show_left_ui=True,
            show_right_ui=True,
        )
        status_queue.put(("ready", ""))

        while viewer.is_running():
            try:
                latest_state = state_queue.get(timeout=1 / 60)
            except queue.Empty:
                viewer.sync()
                continue

            if latest_state is None:
                break
            try:
                while True:
                    newer_state = state_queue.get_nowait()
                    if newer_state is None:
                        latest_state = None
                        break
                    latest_state = newer_state
            except queue.Empty:
                pass

            if latest_state is None:
                break
            qpos, ctrl = latest_state
            if qpos.shape != data.qpos.shape or ctrl.shape != data.ctrl.shape:
                raise ValueError("Dimensi state GUI tidak cocok dengan model viewer.")
            data.qpos[:] = qpos
            data.qvel[:] = 0.0
            data.ctrl[:] = ctrl
            mujoco.mj_forward(model, data)  # type: ignore[attr-defined]

            viewer.sync()
    except Exception as error:
        status_queue.put(("error", str(error)))
    finally:
        if viewer is not None:
            viewer.close()
        if env is not None:
            env.close()
        status_queue.put(("closed", ""))
