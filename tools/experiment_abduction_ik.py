"""Replay diagnostic baseline and fixed-flexion abduction candidate.

Run from project root: .venv/Scripts/python tools/experiment_abduction_ik.py
"""
import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluate_hand_motion import dump, palm_features, phase_profile
from smartglove.experimental_abduction import DiagnosticHandMapper, FixedFlexionAbductionMapper


def write_report(output, metrics):
    lines = ['# Diagnostik IK abduksi — 26 Oktober 2026', '',
             'Kandidat belum diterima sebagai default. Residual orientasi turun, tetapi clipping dan profil fingertip tidak membaik secara konsisten.', '',
             '## Eksperimen terkontrol', '',
             'Baseline memakai HandQposMapper asli dengan subclass observasi. Kandidat memakai hasil IK lama sebagai inisialisasi, menetapkan fleksi akhir hasil kalibrasi, lalu memperbaiki hanya abduksi dengan optimasi skalar bounded pada residual geodesik SO(3). Pencarian menyertakan grid, minimum lokal, endpoint, dan pose awal. Fleksi serta wrist dikunci. Jari tanpa kalibrasi grip tetap memakai fallback asli.',
             'Endpoint fleksi, filter, wrist, kalibrasi dan target sensor tetap seperti rekaman. Eksperimen endpoint FBX sebelumnya tidak digabung agar pengaruh abduksi dapat dipisahkan. Runtime produksi, CLI dan default tidak diubah.',
             f"Replay: {metrics['frames']} frame mapping; selisih maksimum baseline terhadap rekaman {metrics['baseline_max_qpos_error']:.1e} rad. Selisih fleksi/wrist kandidat terhadap baseline {metrics['fixed_flexion_wrist_max_error']:.1e} rad. Frame mentah 523 tanpa event mapping dikecualikan dengan nomor frame, bukan pemotongan array.", '',
             '## Pengaruh overwrite fleksi pada baseline', '',
             'Residual adalah sudut rotasi terpendek dari orientasi distal model menuju target orientasi sensor yang sudah dikalibrasi. Ini error terhadap target mapper, bukan error biomekanika terhadap FBX.', '',
             '| Jari | Setelah IK, sebelum overwrite | Setelah overwrite | Perubahan rata-rata | Frame memburuk |',
             '|---|---:|---:|---:|---:|']
    for finger, entry in metrics['fingers'].items():
        v = entry['overwrite']
        lines.append(f"| {finger} | {v['mean_before_deg']:.2f}° | {v['mean_after_deg']:.2f}° | {v['mean_increase_deg']:+.2f}° | {100*v['worsened_fraction']:.1f}% |")
    lines += ['', 'Overwrite tidak selalu memperburuk residual: jari manis justru membaik rata-rata. Jadi penyebabnya tidak dapat digeneralisasi sebagai satu bug yang sama pada semua jari.', '',
              '## Hasil kandidat', '',
              '| Jari | Residual baseline → kandidat | Di batas baseline → kandidat | RMS kecepatan abduksi baseline → kandidat |',
              '|---|---:|---:|---:|']
    for finger, entry in metrics['fingers'].items():
        a, b = entry['baseline'], entry['candidate']
        lines.append(f"| {finger} | {a['residual_mean_deg']:.2f}° → {b['residual_mean_deg']:.2f}° | {100*a['limit_fraction']:.1f}% → {100*b['limit_fraction']:.1f}% | {a['abduction_velocity_rms_deg_s']:.1f} → {b['abduction_velocity_rms_deg_s']:.1f} °/s |")
    lines += ['', 'Optimasi orientasi saja dapat memilih batas anatomis lebih sering. Jari manis menjadi contoh: residual turun sedikit tetapi occupancy batas naik. Kecepatan RMS adalah ukuran perubahan sepanjang seluruh sesi, bukan jitter pose diam yang sudah dipisahkan.', '',
              '## Profil fingertip terhadap Studio', '',
              'Jendela transisi baseline dibekukan; tiap transisi diresample ke 101 titik fase. Satuan RMSE adalah panjang telapak. Tidak ada pemasangan frame langsung antarrekaman.', '',
              '| Jari | Menutup baseline → kandidat | Membuka baseline → kandidat |', '|---|---:|---:|']
    for j, finger in enumerate(metrics['fingers']):
        a, b = metrics['phase_comparison']['baseline'], metrics['phase_comparison']['candidate']
        lines.append(f"| {finger} | {a['closing'][j]:.4f} → {b['closing'][j]:.4f} | {a['opening'][j]:.4f} → {b['opening'][j]:.4f} |")
    lines += ['', 'Ibu jari menunjukkan perbaikan besar pada metrik jarak, tetapi geometri thumb antarmodel berbeda dan jarak radial tidak mengukur arah gerak. Telunjuk memburuk di kedua fase. Empat jari lain tidak menunjukkan peningkatan seragam yang cukup untuk mempromosikan kandidat.', '',
              '## Grafik', '',
              '- residual_and_abduction.png: kolom kiri residual baseline, kandidat, dan baseline sebelum overwrite; kolom kanan sudut abduksi. Sumbu X detik dari frame mapping pertama; sumbu Y derajat.',
              '- phase_comparison.png: profil jarak tip-wrist/panjang telapak. Studio adalah referensi sesi berbeda; baseline dan kandidat dari replay sesi JSONL yang sama. Sumbu X fase transisi 0–1.', '',
              '## Implikasi teknis berikutnya', '',
              'Prioritas berikutnya adalah memvalidasi target orientasi jari dalam frame HAND dan kalibrasi sumbu/arah fleksi. Residual yang mendekati 180° pada sebagian frame menunjukkan ketidaksesuaian target dan pose akhir yang terlalu besar untuk diselesaikan satu joint abduksi. Angka ini tidak membuktikan salah urutan quaternion: baseline telah mereproduksi rekaman dan quaternion dinormalisasi. Perlu uji rotasi kaku bersama tangan/lengan dan pose sintetis dengan sumbu yang diketahui.',
              'Jangan menambahkan regularisasi atau mengurangi ROM semata-mata agar clipping terlihat turun sebelum targetnya tervalidasi. Setelah target konsisten, kandidat IK ini dapat diuji kembali dengan prior abduksi netral dan kontinuitas waktu secara terpisah.', '',
              '## Pengujian dan reproduksi', '',
              'Empat kasus tes solver lulus: target reachable dengan q dan -q, target di luar ROM, dan residual non-worsening dengan fleksi/wrist tetap. Tiga tes normalisasi/fase/geometri sebelumnya juga lulus. Replay seluruh 463 event memverifikasi baseline identik, fleksi/wrist identik, batas joint, dan residual final kandidat dibanding baseline.', '',
              '```powershell', '.\\.venv\\Scripts\\python.exe tools/experiment_abduction_ik.py', '.\\.venv\\Scripts\\python.exe -m pytest test/test_abduction_ik.py test/test_motion_evaluation.py -q -p no:cacheprovider', '```', '',
              'Keterbatasan: hanya satu sesi replay; target masih orientasi dunia; tidak mengevaluasi dinamika otot; pose thumb memerlukan validasi anatomis tambahan; mean profil antar-sesi tidak sama dengan tracking ground truth.']
    (output / 'analysis_report.md').write_text('\n'.join(lines), encoding='utf-8')


def main():
    import gymnasium as gym
    import mujoco
    from smartglove import runtime
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    source = ROOT / 'recordings/uji2.jsonl'
    baseline_dir = ROOT / 'evaluation/baseline'
    output = ROOT / 'evaluation/abduction_ik'
    output.mkdir(parents=True, exist_ok=True)
    records = [json.loads(line) for line in source.read_text().splitlines()]
    events = [r for r in records if r.get('name') == 'mapping_frame']
    stored = np.asarray([r['qpos'] for r in events])
    base_metrics = json.loads((baseline_dir / 'comparison_metrics.json').read_text())
    source_hashes = {key.replace('\\', '/'): value for key, value in base_metrics['source_sha256'].items()}
    if source_hashes['recordings/uji2.jsonl'] != hashlib.sha256(source.read_bytes()).hexdigest():
        raise ValueError('Baseline source hash mismatch')
    resume = next(r['n'] for r in records if r.get('name') == 'calib_grip_done' and r.get('ok'))
    numbers = [r['n'] for r in records if r['type'] == 'frame' and r['n'] >= resume and r.get('proc', True)]
    indices = [numbers.index(r['n']) for r in events]
    results, logs = {}, {}
    for name, cls in [('baseline', DiagnosticHandMapper), ('candidate', FixedFlexionAbductionMapper)]:
        instances = []

        def factory(*args, **kwargs):
            mapper = cls(*args, **kwargs)
            instances.append(mapper)
            return mapper

        with patch.object(runtime, 'HandQposMapper', factory):
            result = runtime.replay(source, show_viewer=False, realtime=False, mode='stored')
        if result.warnings:
            raise ValueError(f'Replay warnings require investigation: {result.warnings}')
        results[name] = result.qpos[indices]
        logs[name] = [instances[0].diagnostics[i] for i in indices]
    delta = float(np.max(abs(results['baseline'] - stored)))
    if delta > 1e-10:
        raise ValueError(f'Diagnostics changed baseline output: {delta}')
    env = gym.make(records[0]['params']['env'])
    try:
        model, data = env.unwrapped.mj_model, env.unwrapped.mj_data
        fingers = tuple(instances[0].fingers)
        abd_indices = [instances[0].fingers[f]['qpos'][0 if f == 'THUMB' else 1] for f in fingers]
        fixed = [i for i in range(model.nq) if i not in abd_indices]
        fixed_error = float(np.max(abs(results['candidate'][:, fixed] - stored[:, fixed])))
        if fixed_error > 1e-12:
            raise ValueError(f'Candidate changed flexion/wrist: {fixed_error}')
        for j in range(model.njnt):
            if model.jnt_limited[j]:
                values = results['candidate'][:, int(model.jnt_qposadr[j])]
                low, high = model.jnt_range[j]
                if not np.isfinite(values).all() or np.any(values < low-1e-10) or np.any(values > high+1e-10):
                    raise ValueError('Candidate violates joint limits')
        worst_increase = max(logs['candidate'][i][f]['after_deg']-logs['baseline'][i][f]['after_deg']
                             for i in range(len(events)) for f in fingers)
        if worst_increase > 1e-6:
            raise ValueError(f'Candidate worsened per-frame residual: {worst_increase}')
        t = np.asarray([r['t'] for r in events])
        relative_t = t - t[0]
        segments = base_metrics['mapped_segments']
        norm = json.loads((baseline_dir / 'normalized_motion.json').read_text())
        _, _, reference_profiles = phase_profile(np.asarray(norm['reference_t']), np.linalg.norm(norm['reference_tip_palm'], axis=2))
        bid = lambda n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n)
        sid = lambda n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, n)
        bases = [bid(n) for n in ('firstmc', 'proxph2', 'proxph3', 'proxph4', 'proxph5')]
        tips = [sid(n) for n in ('THtip', 'IFtip', 'MFtip', 'RFtip', 'LFtip')]
        metrics = {'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
                   'frames': len(events), 'baseline_max_qpos_error': delta,
                   'fixed_flexion_wrist_max_error': fixed_error, 'fingers': {}, 'phase_comparison': {},
                   'method': 'Legacy IK initializes candidate; bounded scalar SO(3) optimization refines only abduction with final calibrated flexion fixed. Original pose included among candidates.',
                   'limitations': ['Sensor orientation target remains world-based.', 'No temporal regularization; lower orientation residual does not ensure anatomical naturalness.', 'Different sessions; FBX profiles are descriptive, not ground truth.', 'Endpoint experiment is not combined with this candidate.']}
        fig, axes = plt.subplots(5, 2, figsize=(14, 16))
        for row, finger in enumerate(fingers):
            summary = {}
            for variant in results:
                values = logs[variant]
                residual = np.asarray([r[finger]['after_deg'] for r in values])
                abd = np.asarray([r[finger]['abduction_deg'] for r in values])
                summary[variant] = {'residual_mean_deg': float(np.mean(residual)), 'residual_p95_deg': float(np.percentile(residual, 95)),
                                    'limit_fraction': float(np.mean([r[finger]['at_limit'] for r in values])),
                                    'abduction_velocity_rms_deg_s': float(np.sqrt(np.mean((np.diff(abd) / np.diff(t))**2)))}
                axes[row, 0].plot(relative_t, residual, label=variant)
                axes[row, 1].plot(relative_t, abd, label=variant)
            pre = np.asarray([r[finger]['ik_before_overwrite_deg'] for r in logs['baseline']])
            post = np.asarray([r[finger]['final_before_refinement_deg'] for r in logs['baseline']])
            summary['overwrite'] = {'mean_before_deg': float(np.mean(pre)), 'mean_after_deg': float(np.mean(post)),
                                    'mean_increase_deg': float(np.mean(post-pre)), 'worsened_fraction': float(np.mean(post > pre + 1e-6))}
            metrics['fingers'][finger] = summary
            axes[row, 0].plot(relative_t, pre, label='legacy before overwrite', alpha=.5)
            for col in range(2):
                axes[row, col].set_title(f'{finger}: ' + ('orientation residual' if col == 0 else 'abduction'))
                axes[row, col].set_ylabel('degrees'); axes[row, col].set_xlabel('seconds'); axes[row, col].legend()
        fig.tight_layout(); fig.savefig(output / 'residual_and_abduction.png'); plt.close(fig)
        phase_fig, phase_axes = plt.subplots(5, 2, figsize=(12, 14))
        for variant, qpos in results.items():
            distances = []
            for q in qpos:
                data.qpos[:] = q
                mujoco.mj_forward(model, data)
                _, distance = palm_features((data.xpos[bid('lunate')], data.xpos[bases], data.site_xpos[tips]))
                distances.append(distance)
            distances = np.asarray(distances)
            metrics['phase_comparison'][variant] = {}
            for col, phase in enumerate(('closing', 'opening')):
                curves = np.asarray([np.column_stack([np.interp(np.linspace(t[s['start']], t[s['end']], 101), t, distances[:, j]) for j in range(5)]) for s in segments if s['phase'] == phase])
                mean = curves.mean(axis=0)
                reference = reference_profiles[phase].mean(axis=0)
                metrics['phase_comparison'][variant][phase] = np.sqrt(np.mean((mean-reference)**2, axis=0)).tolist()
                for j, finger in enumerate(fingers):
                    ax = phase_axes[j, col]
                    ax.plot(np.linspace(0, 1, 101), mean[:, j], label=variant)
                    if variant == 'baseline':
                        ax.plot(np.linspace(0, 1, 101), reference[:, j], label='Studio', linestyle='--')
                    ax.set_title(f'{finger} {phase}'); ax.set_ylabel('tip-wrist / palm length'); ax.set_xlabel('phase'); ax.legend()
        phase_fig.tight_layout(); phase_fig.savefig(output / 'phase_comparison.png'); plt.close(phase_fig)
        dump(output / 'comparison_metrics.json', metrics)
        dump(output / 'frame_diagnostics.json', {'frame_numbers': [r['n'] for r in events], 'times': t.tolist(), **logs})
        dump(output / 'candidate_qpos.json', results['candidate'].tolist())
        write_report(output, metrics)
        print(json.dumps(metrics, indent=2))
    finally:
        env.close()


if __name__ == '__main__':
    main()
