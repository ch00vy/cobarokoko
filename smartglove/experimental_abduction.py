"""Offline IK diagnostics and opt-in abduction refinement; production is unchanged."""
import numpy as np
import mujoco
from scipy.optimize import minimize_scalar

from smartglove.mapping import HandQposMapper
from smartglove.quaternion import normalize_quaternion, quaternion_error, quaternion_multiply


def fit_abduction(model, data, finger, target):
    """Minimize SO(3) residual with all other joint positions fixed.

    Search the bounded one-dimensional joint interval, including endpoints and
    the original pose. Keeping the original candidate prevents worsening the
    residual even if numerical minimization fails to find an improvement.
    """
    slot = 0 if finger['flexion_slots'][0] == 1 else 1
    joint = finger['joint_ids'][slot]
    index = finger['qpos'][slot]
    if not model.jnt_limited[joint]:
        raise ValueError('Experimental abduction requires a bounded joint')
    low, high = map(float, model.jnt_range[joint])
    original = float(data.qpos[index])

    def objective(value):
        data.qpos[index] = value
        mujoco.mj_forward(model, data)
        error = quaternion_error(target, normalize_quaternion(data.xquat[finger['body']]))
        return float(error @ error)

    grid = np.linspace(low, high, 17)
    candidates = [(original, objective(original))]
    candidates.extend((float(v), objective(v)) for v in grid)
    best = int(np.argmin([v[1] for v in candidates[1:]]))
    left, right = grid[max(0, best - 1)], grid[min(len(grid) - 1, best + 1)]
    if right > left:
        result = minimize_scalar(objective, bounds=(left, right), method='bounded',
                                 options={'xatol': 1e-9})
        if result.success:
            candidates.append((float(result.x), float(result.fun)))
    chosen, cost = min(candidates, key=lambda value: value[1])
    data.qpos[index] = chosen
    mujoco.mj_forward(model, data)
    return {'before_deg': float(np.degrees(np.sqrt(candidates[0][1]))),
            'after_deg': float(np.degrees(np.sqrt(cost))),
            'abduction_deg': float(np.degrees(chosen)),
            'at_limit': bool(min(abs(chosen-low), abs(chosen-high)) < 1e-5)}


class DiagnosticHandMapper(HandQposMapper):
    """Observe legacy IK and optionally refine abduction after final flexion.

    Legacy fitting remains as initialization and handles uncalibrated fingers.
    Only successfully grip-calibrated fingers are refined, so fallback IK and
    its distal coupling retain their original semantics. This isolates the
    effect of solving abduction with final flexion locked.
    """
    refine_abduction = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.diagnostics = []
        self._fit_residuals = {}

    def _fit_finger(self, finger_data, target_orientation, initial_variables):
        result = super()._fit_finger(finger_data, target_orientation, initial_variables)
        error = quaternion_error(target_orientation, normalize_quaternion(self.data.xquat[finger_data['body']]))
        self._fit_residuals[finger_data['body']] = float(np.degrees(np.linalg.norm(error)))
        return result

    def map_frame(self, sensor_quaternions):
        super().map_frame(sensor_quaternions)
        frame = {}
        for name, sensor in sensor_quaternions.items():
            finger = self.fingers[name]
            target = normalize_quaternion(quaternion_multiply(self.sensor_to_model[name], sensor))
            error = quaternion_error(target, normalize_quaternion(self.data.xquat[finger['body']]))
            after_overwrite = float(np.degrees(np.linalg.norm(error)))
            slot = 0 if name == 'THUMB' else 1
            joint, index = finger['joint_ids'][slot], finger['qpos'][slot]
            low, high = self.model.jnt_range[joint]
            entry = {'ik_before_overwrite_deg': self._fit_residuals[finger['body']],
                     'final_before_refinement_deg': after_overwrite,
                     'after_deg': after_overwrite,
                     'abduction_deg': float(np.degrees(self.data.qpos[index])),
                     'at_limit': bool(min(abs(self.data.qpos[index]-low), abs(self.data.qpos[index]-high)) < 1e-5)}
            if self.refine_abduction and self.flexion_calibration is not None and self.flexion_calibration.get(name) is not None:
                entry.update(fit_abduction(self.model, self.data, finger, target))
            frame[name] = entry
        self.diagnostics.append(frame)
        self.data.qvel[:] = 0
        return self.data.qpos.copy()


class FixedFlexionAbductionMapper(DiagnosticHandMapper):
    refine_abduction = True
