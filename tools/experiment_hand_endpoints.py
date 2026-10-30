"""Offline endpoint experiment. Does not alter production defaults.
FBX-derived geometry priors are exploratory, not anatomical ground truth.
"""
import json
from pathlib import Path
import sys
from unittest.mock import patch
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'tools'))
from evaluate_hand_motion import dump, palm_features, phase_profile


def bend(a,b,c):
    u=b-a; v=c-b
    return np.arccos(np.clip(np.dot(u,v)/(np.linalg.norm(u)*np.linalg.norm(v)),-1,1))


def endpoint_priors(reference):
    result={}
    for digit,finger in enumerate(('INDEX','MIDDLE','RING','PINKY'),2):
        values=[]
        for frame in reference['frames']:
            p=[np.asarray(frame[f'RightFinger{digit}{part}']['world'])[:3,3] for part in ('Metacarpal','Proximal','Medial','Distal','Tip')]
            values.append([bend(*p[i:i+3]) for i in range(3)])
        # Open and closed excursion removes static skeleton bend offsets.
        values=np.asarray(values)
        result[finger]=np.percentile(values,95,axis=0)-np.percentile(values,5,axis=0)
    return result


def main():
    import gymnasium as gym
    import mujoco
    from smartglove import runtime
    from smartglove.mapping import HandQposMapper
    from smartglove.config import FINGER_JOINTS
    out=ROOT/'evaluation/baseline'
    ref=json.loads((out/'right_hand_forearm_reference.json').read_text())
    priors=endpoint_priors(ref)
    class EndpointMapper(HandQposMapper):
        def calibrate_grip(self,sensor_grip):
            skipped=super().calibrate_grip(sensor_grip)
            for finger,angles in priors.items():
                item=self.flexion_calibration.get(finger)
                if item is None: continue
                targets=item[2]
                for (qpos_id,(opened,closed,joint)),angle in zip(targets.items(),angles):
                    span=min(abs(closed-opened),float(angle))
                    targets[qpos_id]=(opened+0., opened+np.sign(closed-opened)*span,joint)
            return skipped
    path=ROOT/'recordings/uji2.jsonl'
    records=[json.loads(l) for l in path.read_text().splitlines()]
    stored=np.array([r['qpos'] for r in records if r.get('name')=='mapping_frame'])
    original=runtime.replay(path,show_viewer=False,realtime=False,mode='stored')
    resume=next(r['n'] for r in records if r.get('name')=='calib_grip_done' and r.get('ok'))
    replay_numbers=[r['n'] for r in records if r['type']=='frame' and r['n']>=resume and r.get('proc',True)]
    indices=[replay_numbers.index(r['n']) for r in records if r.get('name')=='mapping_frame']
    original.qpos=original.qpos[indices]
    delta=float(np.max(abs(original.qpos-stored)))
    if delta>1e-8: raise ValueError(f'Current runtime does not reproduce recording: {delta}')
    with patch.object(runtime,'HandQposMapper',EndpointMapper):
        candidate=runtime.replay(path,show_viewer=False,realtime=False,mode='stored')
    candidate.qpos=candidate.qpos[indices]
    env=gym.make('myoHandPoseFixed-v0'); m=env.unwrapped.mj_model; d=env.unwrapped.mj_data
    def bid(name): return mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,name)
    def sid(name): return mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_SITE,name)
    bases=[bid(n) for n in ('firstmc','proxph2','proxph3','proxph4','proxph5')]
    tips=[sid(n) for n in ('THtip','IFtip','MFtip','RFtip','LFtip')]
    distances=[]
    for row in candidate.qpos:
        d.qpos[:]=row; mujoco.mj_forward(m,d)
        _,v=palm_features((d.xpos[bid('lunate')],d.xpos[bases],d.site_xpos[tips]));distances.append(v)
    norm=json.loads((out/'normalized_motion.json').read_text())
    rt=np.array(norm['reference_t']); t=np.array(norm['mapped_t'])
    refdist=np.linalg.norm(norm['reference_tip_palm'],axis=2)
    _,_,rprofiles=phase_profile(rt,refdist)
    segments=json.loads((out/'comparison_metrics.json').read_text())['mapped_segments']
    distances=np.asarray(distances)
    profiles={}
    for phase in ('closing','opening'):
        profiles[phase]=np.asarray([np.column_stack([np.interp(np.linspace(t[s['start']],t[s['end']],101),t,distances[:,j]) for j in range(5)]) for s in segments if s['phase']==phase])
    metrics={'recording_replay_max_abs_qpos_error':delta,'candidate':'FBX geometric 5th-95th percentile flexion excursion per joint; thumb unchanged',
             'endpoint_priors_deg':{k:np.degrees(v).tolist() for k,v in priors.items()},'comparison':{},'warnings':candidate.warnings, 'alignment':'Frozen baseline transition windows for paired A/B evaluation; reference profiles fixed.',
             'limitation':'Priors and evaluation use same Studio clip: descriptive development result, not held-out validation. MCP axes and anatomical models differ.'}
    for phase,curves in profiles.items():
        if phase in rprofiles:
            metrics['comparison'][phase]={'transitions':len(curves),'rmse_hand_lengths':np.sqrt(np.mean((curves.mean(axis=0)-rprofiles[phase].mean(axis=0))**2,axis=0)).tolist()}
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(4,1,figsize=(10,9))
    for ax,finger in zip(axes,priors):
        spec=FINGER_JOINTS[finger]; name=spec['joints'][spec['flexion_slots'][2]]
        j=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,name); col=int(m.jnt_qposadr[j])
        ax.plot(t-t[0],np.degrees(stored[:,col]),label='baseline DIP')
        ax.plot(t-t[0],np.degrees(candidate.qpos[:,col]),label='experimental DIP');ax.set_title(finger);ax.legend()
    fig.tight_layout();fig.savefig(out/'endpoint_experiment.png');plt.close(fig)
    dump(out/'endpoint_experiment_metrics.json',metrics)
    dump(out/'endpoint_experiment_qpos.json',candidate.qpos.tolist())
    env.close();print(json.dumps(metrics,indent=2))

if __name__=='__main__':main()
