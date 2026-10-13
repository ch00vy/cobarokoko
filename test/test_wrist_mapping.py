import importlib
import unittest

import gymnasium as gym
import mujoco
import numpy as np

from smartglove.config import ENV_NAME
from smartglove.quaternion import quaternion_conjugate, quaternion_multiply
from smartglove.wrist_mapping import WristQposMapper
from smartglove_realtime import parse_packet_with_error

importlib.import_module("myosuite")


class WristMappingTests(unittest.TestCase):
    def test_parser_reads_wrist_sensor_records(self) -> None:
        packet = bytearray(624)
        records = {
            0x28: 0x03000000,
            0x48: 0x00000100,
            0x68: 0x00010200,
            0x88: 0x00000300,
            0xA8: 0x00000400,
            0xC8: 0x01010500,
            0xE8: 0x00000600,
        }
        for offset, identifier in records.items():
            packet[offset : offset + 32] = (
                identifier.to_bytes(4, "little")
                + np.array(
                    (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
                    dtype="<f4",
                ).tobytes()
            )

        parsed, error = parse_packet_with_error(bytes(packet))

        self.assertIsNone(error)
        if parsed is None:
            self.fail("A packet with seven valid sensor records was rejected.")
        self.assertEqual(
            set(parsed),
            {"THUMB", "INDEX", "MIDDLE", "RING", "PINKY", "FOREARM", "HAND"},
        )

    def test_wrist_mapper_tracks_model_wrist_pose(self) -> None:
        env = gym.make(ENV_NAME)
        try:
            unwrapped = env.unwrapped
            model = getattr(unwrapped, "mj_model")
            data = getattr(unwrapped, "mj_data")
            data.qpos[:] = getattr(unwrapped, "init_qpos")
            data.qvel[:] = 0.0
            mujoco.mj_forward(model, data)  # type: ignore[attr-defined]
            mapper = WristQposMapper(model, data)
            neutral_forearm = np.array((1.0, 0.0, 0.0, 0.0))
            neutral_hand = np.array((1.0, 0.0, 0.0, 0.0))
            mapper.calibrate(neutral_forearm, neutral_hand)
            if mapper.sensor_to_model is None:
                self.fail("Wrist calibration did not produce sensor alignment.")

            qpos_ids = mapper.qpos_ids
            neutral_qpos = data.qpos.copy()
            neutral_result = mapper.map_frame(
                {
                    "FOREARM": neutral_forearm,
                    "HAND": neutral_hand,
                }
            )
            np.testing.assert_allclose(
                neutral_result[qpos_ids],
                neutral_qpos[qpos_ids],
                atol=1e-3,
            )

            target_qpos = neutral_qpos.copy()
            target_qpos[qpos_ids] += np.array((0.2, 0.1, -0.1))
            target_qpos[qpos_ids] = np.clip(
                target_qpos[qpos_ids], mapper.lower, mapper.upper
            )
            data.qpos[qpos_ids] = target_qpos[qpos_ids]
            mujoco.mj_forward(model, data)  # type: ignore[attr-defined]
            target_orientations = {
                sensor: data.xquat[body].copy()
                for sensor, body in mapper.bodies.items()
            }
            sensor_orientations = {
                sensor: quaternion_multiply(
                    quaternion_conjugate(mapper.sensor_to_model[sensor]),
                    target_orientations[sensor],
                )
                for sensor in mapper.bodies
            }

            data.qpos[:] = neutral_qpos
            mujoco.mj_forward(model, data)  # type: ignore[attr-defined]
            mapped_qpos = mapper.map_frame(sensor_orientations)

            np.testing.assert_allclose(
                mapped_qpos[qpos_ids],
                target_qpos[qpos_ids],
                atol=1e-3,
            )
        finally:
            env.close()
