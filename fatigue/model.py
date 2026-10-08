import math

import mujoco
import numpy as np
from myosuite.envs.myo.fatigue import CumulativeFatigue


class MyoSuiteFatigue:
    """Apply MyoSuite's 3CC-r fatigue model to SmartGlove muscle controls."""

    def __init__(self, model: mujoco.MjModel, timestamp: float) -> None:
        self.model = model
        self.dt = float(model.opt.timestep)
        if not math.isfinite(self.dt) or self.dt <= 0.0:
            raise ValueError("Timestep MuJoCo harus finite dan lebih besar dari nol.")

        self.muscle_ids = np.flatnonzero(
            model.actuator_dyntype == mujoco.mjtDyn.mjDYN_MUSCLE
        )
        if self.muscle_ids.size == 0:
            raise ValueError("Model tidak memiliki aktuator otot untuk fatigue.")

        self.ctrl_ranges = model.actuator_ctrlrange[self.muscle_ids].copy()
        self.ctrl_limited = model.actuator_ctrllimited[self.muscle_ids].copy()
        self.fatigue = CumulativeFatigue(model, frame_skip=1)
        self.enabled = False
        self.last_time = timestamp
        self.remainder = 0.0
        self.last_target = np.zeros(self.muscle_ids.size, dtype=float)

    def set_enabled(self, enabled: bool, timestamp: float) -> None:
        if self.enabled == enabled:
            return
        self.enabled = enabled
        self.last_time = timestamp
        self.remainder = 0.0

    def update(self, target: np.ndarray, timestamp: float) -> np.ndarray:
        """Return direct controls or fatigue-limited controls for one frame."""
        if target.shape != (self.model.nu,):
            raise ValueError(
                f"target harus memiliki bentuk ({self.model.nu},), "
                f"bukan {target.shape}."
            )
        if not np.isfinite(target).all():
            raise ValueError("target harus hanya berisi nilai finite.")
        if not math.isfinite(timestamp) or timestamp < self.last_time:
            raise ValueError("timestamp harus finite dan tidak boleh mundur.")

        result = target.copy()
        muscle_target = self._to_normalized(target[self.muscle_ids])

        if self.enabled:
            elapsed = timestamp - self.last_time + self.remainder
            steps = int(math.floor((elapsed + self.dt * 1e-9) / self.dt))
            for _ in range(steps):
                self.fatigue.compute_act(self.last_target)
            self.remainder = max(0.0, elapsed - steps * self.dt)
            active = self.fatigue.MA
            result[self.muscle_ids] = self._from_normalized(active)
        else:
            result[self.muscle_ids] = target[self.muscle_ids]
            self.remainder = 0.0

        self.last_target = muscle_target
        self.last_time = timestamp
        return result

    def _to_normalized(self, controls: np.ndarray) -> np.ndarray:
        normalized = controls.copy()
        for index, limited in enumerate(self.ctrl_limited):
            if not limited:
                continue
            low, high = self.ctrl_ranges[index]
            if high <= low:
                raise ValueError(
                    f"Rentang kontrol aktuator otot tidak valid: {low}..{high}."
                )
            normalized[index] = (controls[index] - low) / (high - low)
        return np.clip(normalized, 0.0, 1.0)

    def _from_normalized(self, activations: np.ndarray) -> np.ndarray:
        controls = activations.copy()
        for index, limited in enumerate(self.ctrl_limited):
            if limited:
                low, high = self.ctrl_ranges[index]
                controls[index] = low + activations[index] * (high - low)
        return controls
