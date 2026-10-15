"""Pemetaan pergelangan Rokoko Smartglove -> sendi wrist MyoHand.

Perbedaan utama dari versi sebelumnya:

* Yang dipetakan adalah rotasi RELATIF tangan terhadap lengan bawah
  (conj(FOREARM) * HAND), bukan orientasi absolut kedua sensor. Gerakan seluruh
  lengan di ruang dunia tidak lagi "bocor" ke sendi pro_sup/deviation/flexion.
* Sumbu lokal sensor diselaraskan ke sumbu sendi model lewat rotasi tetap A
  (lihat wrist_alignment.py), sehingga fleksi sensor menjadi fleksi model
  walaupun sensor terpasang miring.
* IK diselesaikan TANPA batas sendi, lalu tiap sendi di-clamp ke ROM model.
  Dengan begitu saturasi satu sendi tidak menggeser sudut sendi lain.
"""

import warnings

import mujoco
import numpy as np

from smartglove.quaternion import (
    normalize_quaternion,
    quaternion_conjugate,
    quaternion_error,
    quaternion_multiply,
    quaternion_to_matrix,
    quaternion_to_rotvec,
    rotvec_to_quaternion,
)
from smartglove.wrist_alignment import (
    DEFAULT_AXES_FILE,
    fit_alignment,
    load_axes,
    relative_hand_in_forearm,
)

IK_MAX_ITERATIONS = 30
IK_TOLERANCE = 1e-6
IK_DAMPING = 1e-6
IK_MAX_STEP = 0.3


class WristQposMapper:
    JOINT_NAMES = ("pro_sup", "deviation", "flexion")

    def __init__(
        self,
        model,
        data,
        quaternion_order: str = "xyzw",
        axes_file=DEFAULT_AXES_FILE,
        sensor_axes: dict[str, np.ndarray] | None = None,
        gains: tuple[float, float, float] = (1.0, 1.0, 1.0),
    ) -> None:
        self.model = model
        self.data = data
        self.quaternion_order = quaternion_order
        self.gains = np.asarray(gains, dtype=float)
        if self.gains.shape != (3,) or not np.all(np.isfinite(self.gains)):
            raise ValueError("gains harus berisi tiga nilai finite.")
        self.bodies = {
            "FOREARM": self._name_id(mujoco.mjtObj.mjOBJ_BODY, "radius"),  # type: ignore[attr-defined]
            "HAND": self._name_id(mujoco.mjtObj.mjOBJ_BODY, "lunate"),  # type: ignore[attr-defined]
        }
        # Frame induk radius tidak bergantung pada sendi wrist.
        self.parent_body = int(model.body_parentid[self.bodies["FOREARM"]])
        self.joints = [
            self._name_id(mujoco.mjtObj.mjOBJ_JOINT, name)  # type: ignore[attr-defined]
            for name in self.JOINT_NAMES
        ]
        self.qpos_ids = [int(model.jnt_qposadr[j]) for j in self.joints]
        self.dof_ids = [int(model.jnt_dofadr[j]) for j in self.joints]
        self.lower = np.array(
            [model.jnt_range[j][0] if model.jnt_limited[j] else -np.inf for j in self.joints]
        )
        self.upper = np.array(
            [model.jnt_range[j][1] if model.jnt_limited[j] else np.inf for j in self.joints]
        )

        self._sensor_axes = sensor_axes or load_axes(axes_file, quaternion_order)
        self.alignment = np.eye(3)
        self.alignment_source = "identity (belum dikalibrasi)"
        self.neutral_relative: np.ndarray | None = None
        self.model_relative_neutral: np.ndarray | None = None
        self.neutral_qpos: np.ndarray | None = None
        self.saturated = np.zeros(3, dtype=bool)
        self.last_angles_deg = np.zeros(3)

    def _name_id(self, object_type, name: str) -> int:
        object_id = mujoco.mj_name2id(self.model, object_type, name)  # type: ignore[attr-defined]
        if object_id < 0:
            raise ValueError(f"Objek MuJoCo tidak ditemukan: {name}")
        return object_id

    def _model_axes_in_hand_frame(self) -> dict[str, np.ndarray]:
        rotation = quaternion_to_matrix(self.data.xquat[self.bodies["HAND"]])
        return {
            name: rotation.T @ np.asarray(self.data.xaxis[joint], dtype=float)
            for name, joint in zip(self.JOINT_NAMES, self.joints)
        }

    def calibrate(self, forearm_quaternion: np.ndarray, hand_quaternion: np.ndarray) -> None:
        """Simpan pose netral: sensor (relatif) dan model (relatif)."""
        mujoco.mj_forward(self.model, self.data)  # type: ignore[attr-defined]
        self.neutral_qpos = self.data.qpos.copy()
        self.neutral_relative = relative_hand_in_forearm(
            forearm_quaternion, hand_quaternion
        )
        self.model_relative_neutral = normalize_quaternion(
            quaternion_multiply(
                quaternion_conjugate(normalize_quaternion(self.data.xquat[self.parent_body])),
                normalize_quaternion(self.data.xquat[self.bodies["HAND"]]),
            )
        )
        model_axes = self._model_axes_in_hand_frame()
        if self._sensor_axes and {"flexion", "deviation"} <= self._sensor_axes.keys():
            self.alignment = fit_alignment(self._sensor_axes, model_axes)
            self.alignment_source = "sapuan kalibrasi (wrist_axes.json)"
        else:
            self.alignment = np.eye(3)
            self.alignment_source = "identity (belum dikalibrasi)"
            warnings.warn(
                "Sumbu sensor wrist belum dikalibrasi: diasumsikan sama dengan "
                "sumbu model. Jalankan tools/calibrate_wrist_axes.py agar derajat "
                "gerak wrist sesuai.",
                stacklevel=2,
            )

    def _solve(self, target_world: np.ndarray) -> np.ndarray:
        """Gauss-Newton teredam pada 3 sendi wrist, tanpa batas sendi."""
        jacobian_position = np.zeros((3, self.model.nv))
        jacobian_rotation = np.zeros((3, self.model.nv))
        variables = self.data.qpos[self.qpos_ids].copy()
        body = self.bodies["HAND"]
        for _ in range(IK_MAX_ITERATIONS):
            self.data.qpos[self.qpos_ids] = variables
            mujoco.mj_forward(self.model, self.data)  # type: ignore[attr-defined]
            error = quaternion_error(
                target_world, normalize_quaternion(self.data.xquat[body])
            )
            if np.linalg.norm(error) < IK_TOLERANCE:
                break
            mujoco.mj_jacBody(  # type: ignore[attr-defined]
                self.model, self.data, jacobian_position, jacobian_rotation, body
            )
            jacobian = jacobian_rotation[:, self.dof_ids]
            delta = jacobian.T @ np.linalg.solve(
                jacobian @ jacobian.T + IK_DAMPING * np.eye(3), error
            )
            variables = variables + np.clip(delta, -IK_MAX_STEP, IK_MAX_STEP)
        return variables

    def map_frame(self, sensor_quaternions: dict[str, np.ndarray]) -> np.ndarray:
        if self.neutral_relative is None or self.model_relative_neutral is None:
            raise RuntimeError("Mapper pergelangan harus dikalibrasi lebih dulu.")
        if self.neutral_qpos is None:
            raise RuntimeError("Pose netral model belum tersimpan.")
        missing = {"FOREARM", "HAND"} - sensor_quaternions.keys()
        if missing:
            raise ValueError(
                "Quaternion sensor pergelangan tidak tersedia: "
                f"{', '.join(sorted(missing))}."
            )

        relative = relative_hand_in_forearm(
            sensor_quaternions["FOREARM"], sensor_quaternions["HAND"]
        )
        delta_sensor = quaternion_multiply(
            quaternion_conjugate(self.neutral_relative), relative
        )
        # Rotasi tetap A memetakan rotvec frame-sensor ke frame-lunate model.
        delta_model = rotvec_to_quaternion(
            self.alignment @ quaternion_to_rotvec(delta_sensor)
        )

        mujoco.mj_forward(self.model, self.data)  # type: ignore[attr-defined]
        parent = normalize_quaternion(self.data.xquat[self.parent_body])
        target_world = normalize_quaternion(
            quaternion_multiply(
                quaternion_multiply(parent, self.model_relative_neutral), delta_model
            )
        )

        solved = self._solve(target_world)
        neutral = self.neutral_qpos[self.qpos_ids]
        scaled = neutral + self.gains * (solved - neutral)
        clamped = np.clip(scaled, self.lower, self.upper)
        self.saturated = clamped != scaled
        self.last_angles_deg = np.degrees(clamped - neutral)

        self.data.qpos[self.qpos_ids] = clamped
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)  # type: ignore[attr-defined]
        return self.data.qpos.copy()
