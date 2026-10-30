"""Reproducible offline baseline; source recordings and production mapping are untouched.
Run: .venv/Scripts/python tools/evaluate_hand_motion.py
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
from scipy.signal import find_peaks, savgol_filter
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
FINGERS = ('THUMB', 'INDEX', 'MIDDLE', 'RING', 'PINKY')


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def matrix(value):
    result = np.eye(4)
    result[:3, :] = np.asarray(value).T
    return result


def extract(source, output):
    import ufbx
    scene = ufbx.load_file(str(source))
    nodes = list(scene.nodes)  # Keep wrappers alive: ufbx 0.0.5 lifetime issue.
    stacks = list(scene.anim_stacks)
    if len(stacks) != 1:
        raise ValueError('Select an animation explicitly when multiple takes exist')
    stack = stacks[0]
    selected = [n for n in nodes if n.name in ('RightForeArm', 'RightHand') or n.name.startswith('RightFinger')]
    fps = scene.settings.frames_per_second
    times = np.arange(round((stack.time_end-stack.time_begin)*fps)+1)/fps+stack.time_begin
    frames = []
    # Evaluate FBX local transforms with pre/post rotations and pivots via ufbx.
    # Compose the complete ancestor chain, including the scene root.
    parents = {n.typed_id: (n.parent.typed_id if n.parent else None) for n in nodes}
    for t in times:
        local, world = {}, {}
        for n in nodes:
            tr = n.evaluate_transform(stack.anim, float(t))
            m = np.eye(4)
            m[:3,:3] = Rotation.from_quat(tuple(tr.rotation)).as_matrix() @ np.diag(tuple(tr.scale))
            m[:3,3] = tuple(tr.translation)
            local[n.typed_id] = m
        def resolve(i):
            if i not in world:
                parent = parents[i]
                world[i] = local[i] if parent is None else resolve(parent) @ local[i]
            return world[i]
        frames.append({n.name: {'world': resolve(n.typed_id).tolist(), 'local': local[n.typed_id].tolist()} for n in selected})
    dump(output, {'source': str(source), 'sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'evaluator': 'ufbx 0.0.5 evaluate_transform + full ancestor FK', 'fps': fps, 'unit_meters': scene.settings.unit_meters,
        'axes': str(scene.settings.axes), 'quaternion_order': 'xyzw',
        'note': 'Rest is FBX default transform, not an asserted anatomical neutral; matrices include scale.',
        'bones': {n.name: {'parent': n.parent.name if n.parent else None,
            'rest_world': matrix(n.node_to_world).tolist(), 'rest_local': matrix(n.node_to_parent).tolist()} for n in selected},
        'times': times.tolist(), 'frames': frames})
    # Native wrapper teardown can crash on Windows after successful evaluation.
    sys.stdout.flush()
    os._exit(0)


def palm_features(points):
    """Right hand frame: index-to-little lateral, wrist-to-middle longitudinal.
    Scale by wrist-to-middle MCP. Distances are independent of global coordinates.
    """
    wrist, bases, tips = points
    y = bases[2]-wrist
    scale = np.linalg.norm(y)
    y = y/scale
    x = bases[1]-bases[4]
    x = x-y*np.dot(x,y)
    x /= np.linalg.norm(x)
    z = np.cross(x,y)
    basis = np.column_stack((x,y,z))
    return (tips-wrist)@basis/scale, np.linalg.norm(tips-wrist, axis=1)/scale


def phase_profile(t, distances):
    closure = -np.mean(distances[:,1:], axis=1)
    low, high = np.percentile(closure, [5,95])
    if high-low < 1e-8:
        raise ValueError('No measurable open-close excursion')
    progress = (closure-low)/(high-low)
    dt = np.median(np.diff(t))
    window = max(5, int(.25/dt)//2*2+1)
    window = min(window, len(t)//2*2-1)
    smooth = savgol_filter(progress, window, 2)
    maxima, _ = find_peaks(smooth, prominence=.3, distance=max(1,int(.4/dt)))
    minima, _ = find_peaks(-smooth, prominence=.3, distance=max(1,int(.4/dt)))
    extrema = sorted([(int(i),'closed') for i in maxima]+[(int(i),'open') for i in minima])
    segments = []
    for (a,pa),(b,pb) in zip(extrema, extrema[1:]):
        if pa == pb or t[b]-t[a] < .15:
            continue
        segments.append({'start':a,'end':b,'phase':'closing' if pa=='open' else 'opening'})
    profiles = {}
    for phase in ('closing','opening'):
        curves = [np.column_stack([np.interp(np.linspace(t[s['start']],t[s['end']],101),t,distances[:,j]) for j in range(5)]) for s in segments if s['phase']==phase]
        if curves:
            profiles[phase] = np.asarray(curves)
    return progress, segments, profiles


def baseline(output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import gymnasium as gym
    import myosuite
    import mujoco
    from smartglove.config import FINGER_JOINTS
    output.mkdir(parents=True, exist_ok=True)
    refpath=output/'right_hand_forearm_reference.json'
    if not refpath.exists():
        subprocess.run([sys.executable, __file__, '--extract', str(refpath)], check=True)
    ref=json.loads(refpath.read_text())
    if ref['sha256'] != hashlib.sha256((ROOT/'studio/Clip_3.fbx').read_bytes()).hexdigest():
        raise ValueError('Cached FBX reference hash mismatch')
    records=[json.loads(line) for line in (ROOT/'recordings/uji2.jsonl').read_text().splitlines()]
    meta=records[0]
    events=[r for r in records if r.get('name')=='mapping_frame']
    raw=[r for r in records if r['type']=='frame']
    q=np.array([r['qpos'] for r in events]); t=np.array([r['t'] for r in events])
    if not np.isfinite(q).all() or np.any(np.diff(t)<=0):
        raise ValueError('Nonfinite qpos or nonmonotonic mapping timestamps')
    env=gym.make(meta['params']['env']); model=env.unwrapped.mj_model; data=env.unwrapped.mj_data
    names=[mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_JOINT,i) for i in range(model.njnt)]
    addresses=[int(x) for x in model.jnt_qposadr]
    if model.nq != q.shape[1]: raise ValueError('qpos length mismatch')
    for event in events:
        for name,value in event['joint_positions'].items():
            if abs(event['qpos'][addresses[names.index(name)]]-value)>1e-10:
                raise ValueError('Joint order mismatch')
    def body(name):
        i=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,name)
        if i<0: raise ValueError('Missing body '+name)
        return i
    def site(name):
        i=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_SITE,name)
        if i<0: raise ValueError('Missing site '+name)
        return i
    tips=[site(n) for n in ('THtip','IFtip','MFtip','RFtip','LFtip')]
    bases=[body(n) for n in ('firstmc','proxph2','proxph3','proxph4','proxph5')]
    wrist=body('lunate')
    md=[]; mp=[]
    for row in q:
        data.qpos[:]=row; mujoco.mj_forward(model,data)
        p,d=palm_features((data.xpos[wrist].copy(),data.xpos[bases].copy(),data.site_xpos[tips].copy()))
        md.append(d); mp.append(p)
    rd=[]; rp=[]
    for frame in ref['frames']:
        def pos(n): return np.array(frame[n]['world'])[:3,3]*ref['unit_meters']
        p,d=palm_features((pos('RightHand'),np.array([pos('RightFinger1Metacarpal')]+[pos(f'RightFinger{i}Proximal') for i in range(2,6)]),np.array([pos(f'RightFinger{i}Tip') for i in range(1,6)])))
        rd.append(d); rp.append(p)
    md=np.array(md); rd=np.array(rd); rt=np.array(ref['times'])
    pg,ms,mprof=phase_profile(t,md); rg,rs,rprof=phase_profile(rt,rd)
    joint_metrics={}
    for i,name in enumerate(names):
        v=q[:,addresses[i]]; lo,hi=model.jnt_range[i]
        velocity=np.diff(v)/np.diff(t)
        joint_metrics[name]={'rom_deg':float(np.degrees(np.ptp(v))), 'limit_fraction':float(np.mean((abs(v-lo)<1e-5)|(abs(v-hi)<1e-5))), 'lower_fraction':float(np.mean(abs(v-lo)<1e-5)), 'upper_fraction':float(np.mean(abs(v-hi)<1e-5)), 'velocity_rms_deg_s':float(np.degrees(np.sqrt(np.mean(velocity**2))))}
    coupling={}
    for finger,spec in FINGER_JOINTS.items():
        cols=[addresses[names.index(spec['joints'][i])] for i in spec['flexion_slots']]
        singular=np.linalg.svd(q[:,cols]-q[:,cols].mean(axis=0),compute_uv=False)
        coupling[finger]={'flexion_variance_first_component':float(singular[0]**2/np.sum(singular**2))}
    comparison={}
    fig,axes=plt.subplots(5,2,figsize=(11,14))
    for col,phase in enumerate(('closing','opening')):
        if phase not in mprof or phase not in rprof: continue
        a,b=mprof[phase],rprof[phase]
        comparison[phase]={'reference_transitions':len(b),'mapped_transitions':len(a),'distance_profile_rmse_hand_lengths':np.sqrt(np.mean((a.mean(axis=0)-b.mean(axis=0))**2,axis=0)).tolist()}
        for j,finger in enumerate(FINGERS):
            ax=axes[j,col]
            for curves,label in ((a,'MyoHand'),(b,'Studio')):
                mean=curves[:,:,j].mean(axis=0); std=curves[:,:,j].std(axis=0)
                ax.plot(np.linspace(0,1,101),mean,label=label); ax.fill_between(np.linspace(0,1,101),mean-std,mean+std,alpha=.2)
            ax.set_title(f'{finger} {phase}'); ax.set_ylabel('tip-wrist / palm length'); ax.legend()
    fig.tight_layout(); fig.savefig(output/'phase_comparison.png'); plt.close(fig)
    fig,ax=plt.subplots(2,1,figsize=(12,6))
    ax[0].plot(rt,rg); ax[0].set_title('Studio closure (robust normalized)')
    ax[1].plot(t-t[0],pg); ax[1].set_title('MyoHand closure (robust normalized)')
    for a in ax: a.set_xlabel('seconds')
    fig.tight_layout(); fig.savefig(output/'closure_baseline.png'); plt.close(fig)
    metrics={'counts':{'raw':len(raw),'mapped':len(events),'reference':len(rd)},'params':meta['params'],
        'mapping_dt_seconds':{'median':float(np.median(np.diff(t))),'max':float(np.max(np.diff(t)))},
        'qpos_joint_order':dict(zip(names,addresses)),'joints':joint_metrics,'coupling':coupling,
        'alignment':'Separate opening/closing transitions between prominent extrema; resample each transition to 101 elapsed-phase points; no absolute frame pairing.',
        'normalization':'Right-handed geometric palm basis; wrist origin, middle MCP distance scale. FBX cm to m. Thumb bases differ anatomically; thumb distance is exploratory.',
        'comparison':comparison,'mapped_segments':ms,'reference_segments':rs,
        'limitations':['Different sessions: profile RMSE is descriptive, not ground-truth tracking error.', 'Phase averaging does not measure live lag.', 'No actual dynamic state/EMG in recording.','No anatomical angle equality inferred from FBX local Euler axes.']}
    metrics['source_sha256'] = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'recordings/uji2.jsonl', ROOT/'studio/Clip_3.fbx')}
    metrics['validation'] = {'missing_mapping_frame_numbers': sorted(set(r['n'] for r in raw if r['n']>=events[0]['n'])-set(r['n'] for r in events)),
        'raw_quaternion_max_norm_error': max(abs(np.linalg.norm(sample['q'])-1) for r in raw for sample in r['s'].values()),
        'filtered_order': 'wxyz (verified from runtime; meta quat_order applies to sensor samples)'}
    matched={r['n']:r for r in raw}
    filter_difference={}
    for sensor in ('THUMB','INDEX','MIDDLE','RING','PINKY','FOREARM','HAND'):
        differences=[]
        for event in events:
            xyzw=np.asarray(matched[event['n']]['s'][sensor]['q'])
            wxyz=xyzw[[3,0,1,2]]
            filtered=np.asarray(event['filtered_quaternions'][sensor])
            dot=abs(np.dot(wxyz/np.linalg.norm(wxyz),filtered/np.linalg.norm(filtered)))
            differences.append(np.degrees(2*np.arccos(np.clip(dot,0,1))))
        filter_difference[sensor]={'median_geodesic_deg':float(np.median(differences)), 'p95_geodesic_deg':float(np.percentile(differences,95))}
    metrics['filter']={'raw_to_filtered':filter_difference, 'median_update_weight':float(-np.expm1(-2*np.pi*meta['params']['filter_cutoff_hz']*np.median(np.diff(t)))), 'note':'Weight near one provides little smoothing at this processed sample rate. Geodesic difference is not an estimated delay.'}
    metrics['calibration']={}
    for name in ('calib_open_done','calib_grip_done'):
        event=next(r for r in records if r.get('name')==name)
        samples=[r for r in raw if event['n']-30<r['n']<=event['n']]
        key='neutral' if name=='calib_open_done' else 'grip'
        stability={}
        for sensor,mean in event[key].items():
            qmean=np.asarray(mean)/np.linalg.norm(mean)
            dots=[abs(np.dot(qmean,np.asarray(r['s'][sensor]['q'])/np.linalg.norm(r['s'][sensor]['q']))) for r in samples]
            stability[sensor]=float(np.degrees(2*np.arccos(np.clip(dots,0,1))).max())
        metrics['calibration'][name]={'max_sample_deviation_deg':stability}
    uniform=np.linspace(t[0],t[-1],len(t))
    qi=np.column_stack([np.interp(uniform,t,v) for v in q.T])
    derivative=qi
    for order,label in enumerate(('velocity','acceleration','jerk'),1):
        derivative=np.gradient(derivative,uniform,axis=0)
        metrics[label+'_rms_rad_per_s_power_'+str(order)]=np.sqrt(np.mean(derivative**2,axis=0)).tolist()
    metrics['derivative_caution']='Linear resampling to median-rate uniform grid then finite differences; jerk is noise-sensitive, no biomechanical optimality claim.'
    dump(output/'comparison_metrics.json',metrics)
    dump(output/'normalized_motion.json',{'mapped_t':t.tolist(),'reference_t':rt.tolist(),'mapped_tip_palm':np.asarray(mp).tolist(),'reference_tip_palm':np.asarray(rp).tolist()})
    env.close()
    print(json.dumps({'counts':metrics['counts'],'comparison':comparison,'highest_limit_fraction':sorted(joint_metrics.items(),key=lambda item:-item[1]['limit_fraction'])[:6]},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--extract'); parser.add_argument('--output',default=str(ROOT/'evaluation/baseline'))
    args=parser.parse_args()
    if args.extract: extract(ROOT/'studio/Clip_3.fbx',Path(args.extract))
    else: baseline(Path(args.output))
