import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from evaluate_hand_motion import palm_features, phase_profile
from experiment_hand_endpoints import bend
from scipy.spatial.transform import Rotation


def test_palm_normalization_invariant_to_rigid_motion_and_scale():
    wrist=np.zeros(3)
    bases=np.array([[.4,.5,0],[.3,1,0],[0,1.1,0],[-.2,1,0],[-.4,.8,0]])
    tips=bases+np.array([0,.7,.1])
    expected=palm_features((wrist,bases,tips))
    r=Rotation.from_rotvec([.5,-.7,.2]).as_matrix();offset=np.array([2,3,4])
    actual=palm_features(tuple(3.2*np.asarray(p)@r.T+offset for p in (wrist,bases,tips)))
    for a,b in zip(actual,expected):np.testing.assert_allclose(a,b,atol=1e-12)


def test_phase_segmentation_handles_different_session_timing():
    profiles=[]
    for duration in (8.,13.):
        t=np.linspace(0,duration,401)
        d=np.repeat((1.5+.5*np.cos(2*np.pi*4*t/duration))[:,None],5,axis=1)
        _,segments,p=phase_profile(t,d)
        assert len(segments)>=5
        profiles.append(p)
    for phase in ('opening','closing'):
        np.testing.assert_allclose(profiles[0][phase].mean(axis=0),profiles[1][phase].mean(axis=0),atol=.03)


def test_geometric_bend_straight_and_right_angle():
    assert abs(bend(np.array([0,0,0]),np.array([1,0,0]),np.array([2,0,0])))<1e-12
    assert abs(bend(np.array([0,0,0]),np.array([1,0,0]),np.array([1,1,0]))-np.pi/2)<1e-12
