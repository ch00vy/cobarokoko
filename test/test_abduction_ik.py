"""Solver tests use known reachable poses, independent of recorded targets."""
import sys
from pathlib import Path

import mujoco
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from smartglove.experimental_abduction import fit_abduction


@pytest.fixture
def chain():
    model = mujoco.MjModel.from_xml_string('''<mujoco><compiler angle="radian"/>
    <worldbody><body><joint axis="0 1 0" range="-1 1"/><geom size=".02"/>
      <body><joint axis="1 0 0" range="0 1.5"/>
        <joint axis="0 0 1" range="-.26 .26"/><geom size=".02"/>
        <body pos="0 .05 0"><joint axis="1 0 0" range="0 1.5"/><geom size=".02"/>
          <body pos="0 .05 0" name="tip"><joint axis="1 0 0" range="0 1.5"/><geom size=".02"/></body>
        </body></body></body></worldbody></mujoco>''')
    data = mujoco.MjData(model)
    finger = {'qpos': [1, 2, 3, 4], 'joint_ids': [1, 2, 3, 4],
              'flexion_slots': (0, 2, 3), 'body': model.body('tip').id}
    data.qpos[:] = [.1, .3, .12, .4, .2]
    mujoco.mj_forward(model, data)
    target = data.xquat[finger['body']].copy()
    return model, data, finger, target


@pytest.mark.parametrize('sign', [1, -1])
def test_reachable_orientation_with_locked_flexion(chain, sign):
    model, data, finger, target = chain
    data.qpos[2] = -.2
    fixed = data.qpos[[0, 1, 3, 4]].copy()
    result = fit_abduction(model, data, finger, target * sign)
    assert result['after_deg'] < 1e-5
    assert data.qpos[2] == pytest.approx(.12, abs=1e-7)
    np.testing.assert_array_equal(data.qpos[[0, 1, 3, 4]], fixed)


def test_unreachable_target_chooses_limit_without_exceeding_it(chain):
    model, data, finger, _ = chain
    data.qpos[2] = .6
    mujoco.mj_forward(model, data)
    target = data.xquat[finger['body']].copy()
    data.qpos[2] = 0
    result = fit_abduction(model, data, finger, target)
    assert data.qpos[2] == pytest.approx(.26, abs=1e-7)
    assert result['at_limit']
    assert result['after_deg'] < result['before_deg']


def test_unrelated_orientation_error_cannot_worsen(chain):
    model, data, finger, _ = chain
    target = np.array([1., 0, 0, 0])
    fixed = data.qpos[[0, 1, 3, 4]].copy()
    result = fit_abduction(model, data, finger, target)
    assert result['after_deg'] <= result['before_deg'] + 1e-9
    np.testing.assert_array_equal(data.qpos[[0, 1, 3, 4]], fixed)
