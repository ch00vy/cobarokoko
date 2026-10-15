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
* Pronasi/supinasi (pro_sup) dibaca dari TWIST sensor FOREARM terhadap sumbu
  panjang lengan bawah ("forearm_twist" di wrist_axes.json). Sensor di manset
  ikut berputar bersama tangan, sehingga rotasi RELATIF hand|forearm nyaris nol
  saat lengan bawah diputar; tanpa kanal twist ini pro_sup tidak dapat dibaca.
  Fleksi/deviasi tetap dari rotasi relatif, dan diselesaikan dengan pro_sup
  yang sudah terkunci. Tanpa "forearm_twist", mapper kembali ke perilaku lama
  (3 sendi dari rotasi relatif).
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
    twist_angle,
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
        twist = (self._sensor_axes or {}).get("forearm_twist")
        self.twist_axis = None if twist is None else np.asarray(twist, dtype=float)
        self.neutral_forearm: np.ndarray | None = None
        self.lunate_in_radius_neutral: np.ndarray | None = None
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
        self.neutral_forearm = normalize_quaternion(forearm_quaternion)
        self.lunate_in_radius_neutral = normalize_quaternion(
            quaternion_multiply(
                quaternion_conjugate(normalize_quaternion(self.data.xquat[self.bodies["FOREARM"]])),
                normalize_quaternion(self.data.xquat[self.bodies["HAND"]]),
            )
        )
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

    def _solve(self, target_world: np.ndarray, free: tuple[int, ...] = (0, 1, 2)) -> np.ndarray:
        """Gauss-Newton teredam pada sendi wrist `free`, tanpa batas sendi."""
        jacobian_position = np.zeros((3, self.model.nv))
        jacobian_rotation = np.zeros((3, self.model.nv))
        variables = self.data.qpos[self.qpos_ids].copy()
        body = self.bodies["HAND"]
        dofs = [self.dof_ids[i] for i in free]
        for _ in range(IK_MAX_ITERATIONS):
            self.data.qpos[self.qpos_ids] = variables
            mujoco.mj_forward(self.model, self.data)  # type: ignore[attr-defined]
            error = quaternion_error(
                target_world, normalize_quaternion(self.data.xquat[body])
            )
            mujoco.mj_jacBody(  # type: ignore[attr-defined]
                self.model, self.data, jacobian_position, jacobian_rotation, body
            )
            jacobian = jacobian_rotation[:, dofs]
            step = np.linalg.solve(
                jacobian.T @ jacobian + IK_DAMPING * np.eye(len(free)), jacobian.T @ error
            )
            if np.linalg.norm(step) < IK_TOLERANCE:
                break
            variables[list(free)] += np.clip(step, -IK_MAX_STEP, IK_MAX_STEP)
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

        if self.twist_axis is not None and self.lunate_in_radius_neutral is not None:
            clamped, scaled = self._map_with_twist(sensor_quaternions["FOREARM"], delta_model)
        else:
            clamped, scaled = self._map_relative_only(delta_model)
        neutral = self.neutral_qpos[self.qpos_ids]
        self.saturated = clamped != scaled
        self.last_angles_deg = np.degrees(clamped - neutral)

        self.data.qpos[self.qpos_ids] = clamped
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)  # type: ignore[attr-defined]
        return self.data.qpos.copy()

    def _finish(self, solved: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        neutral = self.neutral_qpos[self.qpos_ids]
        scaled = neutral + self.gains * (solved - neutral)
        return np.clip(scaled, self.lower, self.upper), scaled

    def _map_relative_only(self, delta_model: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Perilaku lama: tiga sendi dari rotasi relatif hand|forearm."""
        mujoco.mj_forward(self.model, self.data)  # type: ignore[attr-defined]
        parent = normalize_quaternion(self.data.xquat[self.parent_body])
        target_world = normalize_quaternion(
            quaternion_multiply(
                quaternion_multiply(parent, self.model_relative_neutral), delta_model
            )
        )
        return self._finish(self._solve(target_world))

    def _map_with_twist(
        self, forearm: np.ndarray, delta_model: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """pro_sup dari twist FOREARM; deviasi/fleksi dari rotasi relatif."""
        delta_forearm = quaternion_multiply(
            quaternion_conjugate(self.neutral_forearm), normalize_quaternion(forearm)
        )
        neutral = self.neutral_qpos[self.qpos_ids]
        pro_sup = neutral[0] + self.gains[0] * twist_angle(delta_forearm, self.twist_axis)
        pro_sup = float(np.clip(pro_sup, self.lower[0], self.upper[0]))
        self.data.qpos[self.qpos_ids[0]] = pro_sup
        mujoco.mj_forward(self.model, self.data)  # type: ignore[attr-defined]
        radius = normalize_quaternion(self.data.xquat[self.bodies["FOREARM"]])
        target_world = normalize_quaternion(
            quaternion_multiply(
                quaternion_multiply(radius, self.lunate_in_radius_neutral), delta_model
            )
        )
        solved = self._solve(target_world, free=(1, 2))
        solved[0] = pro_sup
        # gain pro_sup sudah diterapkan pada twist; hindari menerapkannya dua kali
        gains = self.gains.copy()
        gains[0] = 1.0
        scaled = neutral + gains * (solved - neutral)
        return np.clip(scaled, self.lower, self.upper), scaled
