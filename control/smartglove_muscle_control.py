from collections.abc import Mapping

import mujoco
import numpy as np


_FINGER_MUSCLES = {
    "FDS2": ("INDEX", "flexion"),
    "FDP2": ("INDEX", "flexion"),
    "EDC2": ("INDEX", "extension"),
    "EIP": ("INDEX", "extension"),
    "RI2": ("INDEX", "abduction"),
    "LU_RB2": ("INDEX", "flexion_assist"),
    "UI_UB2": ("INDEX", "abduction"),
    "FDS3": ("MIDDLE", "flexion"),
    "FDP3": ("MIDDLE", "flexion"),
    "EDC3": ("MIDDLE", "extension"),
    "RI3": ("MIDDLE", "abduction"),
    "LU_RB3": ("MIDDLE", "flexion_assist"),
    "UI_UB3": ("MIDDLE", "abduction"),
    "FDS4": ("RING", "flexion"),
    "FDP4": ("RING", "flexion"),
    "EDC4": ("RING", "extension"),
    "RI4": ("RING", "abduction"),
    "LU_RB4": ("RING", "flexion_assist"),
    "UI_UB4": ("RING", "abduction"),
    "FDS5": ("PINKY", "flexion"),
    "FDP5": ("PINKY", "flexion"),
    "EDC5": ("PINKY", "extension"),
    "EDM": ("PINKY", "extension"),
    "RI5": ("PINKY", "abduction"),
    "LU_RB5": ("PINKY", "flexion_assist"),
    "UI_UB5": ("PINKY", "abduction"),
    "FPL": ("THUMB", "flexion"),
    "EPL": ("THUMB", "extension"),
    "EPB": ("THUMB", "extension"),
    "APL": ("THUMB", "abduction"),
    "OP": ("THUMB", "opposition"),
    "FCR": ("ALL", "wrist_flexion"),
    "FCU": ("ALL", "wrist_flexion"),
    "PL": ("ALL", "wrist_flexion"),
    "ECRL": ("ALL", "wrist_extension"),
    "ECRB": ("ALL", "wrist_extension"),
    "ECU": ("ALL", "wrist_extension"),
    "PT": ("ALL", "thumb_abduction"),
    "PQ": ("ALL", "thumb_abduction"),
}


class SmartGloveMuscleController:
    """Map calibrated hand joint positions to MyoSuite muscle activations."""

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        finger_specs: Mapping[str, Mapping[str, object]],
    ) -> None:
        self.model = model
        self.data = data
        self.neutral_qpos = data.qpos.copy()
        self.fingers: dict[str, dict[str, object]] = {}
        self.actuator_names = []
        for actuator_id in range(model.nu):
            name = mujoco.mj_id2name(
                model,
                mujoco.mjtObj.mjOBJ_ACTUATOR,
                actuator_id,
            )
            if name is None:
                raise ValueError(f"Aktuator MuJoCo tidak memiliki nama: {actuator_id}")
            self.actuator_names.append(name)

        missing_mappings = set(self.actuator_names) - _FINGER_MUSCLES.keys()
        if missing_mappings:
            raise ValueError(
                "Belum ada pemetaan Smartglove untuk aktuator: "
                + ", ".join(sorted(missing_mappings))
            )

        for finger, specification in finger_specs.items():
            joint_names = specification["joints"]
            joint_ids = []
            for joint_name in joint_names:
                joint_id = mujoco.mj_name2id(
                    model,
                    mujoco.mjtObj.mjOBJ_JOINT,
                    joint_name,
                )
                if joint_id < 0:
                    raise ValueError(
                        f"Joint MuJoCo untuk otot {finger} tidak ditemukan: "
                        f"{joint_name}"
                    )
                joint_ids.append(joint_id)

            self.fingers[finger] = {
                "qpos_ids": [
                    int(model.jnt_qposadr[joint_id]) for joint_id in joint_ids
                ],
                "joint_ids": joint_ids,
                "flexion_slots": specification["flexion_slots"],
                "flexion_closed_limit": specification.get(
                    "flexion_closed_limit", "high"
                ),
            }

    def _finger_motion(self, qpos: np.ndarray) -> dict[str, dict[str, float]]:
        motion = {}
        for finger, finger_data in self.fingers.items():
            joint_ids = finger_data["joint_ids"]
            qpos_ids = finger_data["qpos_ids"]
            flexion_slots = finger_data["flexion_slots"]
            flexion_values = []

            for slot in flexion_slots:
                joint_id = joint_ids[slot]
                qpos_id = qpos_ids[slot]
                neutral = self.neutral_qpos[qpos_id]
                low, high = self.model.jnt_range[joint_id]
                closed = (
                    low
                    if finger_data["flexion_closed_limit"] == "low"
                    else high
                )
                span = closed - neutral
                progress = (
                    0.0
                    if abs(span) < 1e-8
                    else (qpos[qpos_id] - neutral) / span
                )
                flexion_values.append(float(np.clip(progress, 0.0, 1.0)))

            abduction_joint_id = joint_ids[0 if finger == "THUMB" else 1]
            abduction_qpos_id = qpos_ids[0 if finger == "THUMB" else 1]
            low, high = self.model.jnt_range[abduction_joint_id]
            neutral = self.neutral_qpos[abduction_qpos_id]
            abduction_span = max(neutral - low, high - neutral)
            abduction = (
                0.0
                if abduction_span < 1e-8
                else abs(qpos[abduction_qpos_id] - neutral) / abduction_span
            )

            motion[finger] = {
                "flexion": float(np.mean(flexion_values)),
                "abduction": float(np.clip(abduction, 0.0, 1.0)),
            }
        return motion

    def update(self, qpos: np.ndarray) -> np.ndarray:
        """Write normalized activations for all model actuators to ``data.ctrl``."""
        if qpos.shape != self.data.qpos.shape:
            raise ValueError(
                f"qpos harus memiliki bentuk {self.data.qpos.shape}, "
                f"bukan {qpos.shape}."
            )
        if not np.isfinite(qpos).all():
            raise ValueError("qpos harus hanya berisi nilai finite.")

        motion = self._finger_motion(qpos)
        mean_flexion = float(
            np.mean([values["flexion"] for values in motion.values()])
        )
        mean_extension = 1.0 - mean_flexion
        thumb_abduction = motion["THUMB"]["abduction"]
        activations = np.empty(self.model.nu, dtype=float)

        for actuator_id, muscle_name in enumerate(self.actuator_names):
            finger, action = _FINGER_MUSCLES[muscle_name]
            if finger == "ALL":
                if action == "wrist_flexion":
                    activation = 0.1 * mean_flexion
                elif action == "wrist_extension":
                    activation = 0.1 * mean_extension
                else:
                    activation = 0.1 * thumb_abduction
            else:
                finger_motion = motion[finger]
                if action == "flexion":
                    activation = finger_motion["flexion"]
                elif action == "extension":
                    activation = 1.0 - finger_motion["flexion"]
                elif action == "abduction":
                    activation = finger_motion["abduction"]
                elif action == "flexion_assist":
                    activation = 0.35 * finger_motion["flexion"]
                else:
                    activation = max(
                        finger_motion["flexion"],
                        finger_motion["abduction"],
                    )

            if self.model.actuator_ctrllimited[actuator_id]:
                low, high = self.model.actuator_ctrlrange[actuator_id]
                activation = low + np.clip(activation, 0.0, 1.0) * (high - low)
            activations[actuator_id] = activation

        self.data.ctrl[:] = activations
        return activations.copy()
