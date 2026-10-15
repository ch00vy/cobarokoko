import unittest

import mujoco
import numpy as np

from smartglove.quaternion import (
    quaternion_conjugate,
    quaternion_multiply,
    quaternion_to_matrix,
    quaternion_to_rotvec,
    rotvec_to_quaternion,
)
from smartglove.wrist_alignment import (
    principal_axis,
    relative_hand_in_forearm,
    sweep_rotvecs,
)
from smartglove.wrist_mapping import WristQposMapper
from smartglove_realtime import parse_packet_with_error

# Rantai wrist yang meniru myohand_body.xml: radius(pro_sup) -> lunate(deviation, flexion).
WRIST_XML = """
<mujoco>
  <compiler angle="radian"/>
  <worldbody>
    <body name="forearm_root" pos="0 0 1" euler="0.3 -0.4 0.2">
      <geom type="sphere" size="0.01" mass="1"/>
      <body name="radius" pos="0 -0.01 0.02">
        <joint name="pro_sup" axis="-0.017161 0.992666 -0.119668" range="-1.5708 1.5708"/>
        <geom type="sphere" size="0.01" mass="1"/>
        <body name="lunate" pos="0.018 -0.24 0.025">
          <joint name="deviation" axis="0 0 1" range="-0.174533 0.436332"/>
          <joint name="flexion" axis="1 0 0" range="-0.785398 0.785398"/>
          <geom type="sphere" size="0.01" mass="1"/>
        </body>
      </body>
    </body>
  </worldbody>
</mujoco>
"""


def random_rotation(rng: np.random.Generator) -> np.ndarray:
    q = rng.normal(size=4)
    return q / np.linalg.norm(q)


class WristFixture:
    """Simulasi sensor: pemasangan sensor acak + gerak lengan global acak."""

    def __init__(self, seed: int = 0) -> None:
        rng = np.random.default_rng(seed)
        self.model = mujoco.MjModel.from_xml_string(WRIST_XML)
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)
        self.rng = rng
        self.mount_hand = random_rotation(rng)  # Sh: sensor tangan terhadap lunate
        self.mount_forearm = random_rotation(rng)  # Sf
        self.joint_qpos = [
            int(self.model.jnt_qposadr[mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_JOINT, name)])
            for name in ("pro_sup", "deviation", "flexion")
        ]

    def sensors(self, angles, arm_world=None):
        """Quaternion sensor (wxyz) untuk sudut sendi (pro_sup, dev, flex) rad."""
        data = mujoco.MjData(self.model)
        data.qpos[self.joint_qpos] = angles
        mujoco.mj_forward(self.model, data)
        parent = self.model.body_parentid[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "radius")]
        lunate = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "lunate")
        rel = quaternion_multiply(
            quaternion_conjugate(data.xquat[parent]), data.xquat[lunate]
        )
        world = np.array((1.0, 0, 0, 0)) if arm_world is None else arm_world
        forearm = quaternion_multiply(world, self.mount_forearm)
        hand = quaternion_multiply(
            quaternion_multiply(world, rel), self.mount_hand
        )
        return {"FOREARM": forearm, "HAND": hand}

    def sweep(self, joint_index, lo_deg, hi_deg, frames=60):
        angles_deg = np.concatenate(
            (np.linspace(0, hi_deg, frames // 2), np.linspace(hi_deg, lo_deg, frames))
        )
        forearm, hand = [], []
        for deg in angles_deg:
            angles = np.zeros(3)
            angles[joint_index] = np.radians(deg)
            s = self.sensors(angles)
            forearm.append(s["FOREARM"])
            hand.append(s["HAND"])
        return forearm, hand


class ParserTests(unittest.TestCase):
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
            packet[offset : offset + 32] = identifier.to_bytes(4, "little") + np.array(
                (0, 0, 0, 0, 0, 0, 1.0), dtype="<f4"
            ).tobytes()
        parsed, error = parse_packet_with_error(bytes(packet))
        self.assertIsNone(error)
        self.assertEqual(
            set(parsed),
            {"THUMB", "INDEX", "MIDDLE", "RING", "PINKY", "FOREARM", "HAND"},
        )

    def test_parser_rejects_swapped_wrist_slots(self) -> None:
        packet = bytearray(624)
        records = {0x28: 0x0, 0x48: 0x100, 0x68: 0x200, 0x88: 0x300,
                   0xA8: 0x400, 0xC8: 0x600, 0xE8: 0x500}  # FOREARM/HAND tertukar
        for offset, identifier in records.items():
            packet[offset : offset + 32] = identifier.to_bytes(4, "little") + np.array(
                (0, 0, 0, 0, 0, 0, 1.0), dtype="<f4"
            ).tobytes()
        parsed, error = parse_packet_with_error(bytes(packet))
        self.assertIsNone(parsed)
        self.assertIn("FOREARM", error)


class WristMappingTests(unittest.TestCase):
    def build_mapper(self, fixture):
        axes = {}
        # Sapuan satu-sumbu; sisi simpangan terbesar dibuat positif -> pakai
        # rentang searah rentang model (fleksi 0..+60, deviasi 0..+25).
        for name, index, hi in (("flexion", 2, 60), ("deviation", 1, 25)):
            f, h = fixture.sweep(index, -5, hi)
            axes[name] = principal_axis(sweep_rotvecs(f, h))
        mapper = WristQposMapper(
            fixture.model, fixture.data,
            sensor_axes={k: v["axis"] for k, v in axes.items()},
        )
        s0 = fixture.sensors(np.zeros(3))
        mujoco.mj_resetData(fixture.model, fixture.data)
        mapper.calibrate(s0["FOREARM"], s0["HAND"])
        return mapper

    def test_recovers_joint_angles_with_tilted_sensors_and_arm_motion(self) -> None:
        fixture = WristFixture(seed=3)
        mapper = self.build_mapper(fixture)
        for ps, dev, flex in ((0.3, 0.1, 0.5), (-0.5, 0.3, -0.4), (0.0, -0.1, 0.0)):
            arm = random_rotation(fixture.rng)  # seluruh lengan bergerak di dunia
            mapped = mapper.map_frame(fixture.sensors(np.array((ps, dev, flex)), arm))
            np.testing.assert_allclose(
                mapped[mapper.qpos_ids], (ps, dev, flex), atol=np.radians(0.2)
            )

    def test_saturation_does_not_leak_into_other_joints(self) -> None:
        fixture = WristFixture(seed=5)
        mapper = self.build_mapper(fixture)
        # Fleksi 70° melewati batas model (45°); pro_sup/deviasi harus tetap.
        target = np.array((0.2, 0.1, np.radians(70.0)))
        # Sensor disimulasikan dengan model tanpa batas sendi.
        unlimited = WristFixture(seed=5)
        unlimited.model.jnt_limited[:] = 0
        mapped = mapper.map_frame(unlimited.sensors(target))
        flex_id = mapper.qpos_ids[2]
        self.assertAlmostEqual(mapped[flex_id], mapper.upper[2], places=3)
        self.assertTrue(mapper.saturated[2])
        np.testing.assert_allclose(mapped[mapper.qpos_ids[:2]], target[:2], atol=np.radians(0.2))


if __name__ == "__main__":
    unittest.main()
