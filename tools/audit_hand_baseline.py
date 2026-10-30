"""Additional FBX/FK geometric audit and concise Indonesian report."""
import json
from pathlib import Path
import sys
import numpy as np
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tools'))
from evaluate_hand_motion import dump
from experiment_hand_endpoints import bend


def rotation(matrix):
    u,_,v=np.linalg.svd(np.asarray(matrix)[:3,:3]);r=u@v
    if np.linalg.det(r)<0: raise ValueError('Reflected skeleton transform')
    return r


def main():
    import gymnasium as gym
    import myosuite
    import mujoco
    out=ROOT/'evaluation/baseline'
    ref=json.loads((out/'right_hand_forearm_reference.json').read_text())
    metrics=json.loads((out/'comparison_metrics.json').read_text())
    exp=json.loads((out/'endpoint_experiment_metrics.json').read_text())
    candidate=np.asarray(json.loads((out/'endpoint_experiment_qpos.json').read_text()))
    records=[json.loads(l) for l in (ROOT/'recordings/uji2.jsonl').read_text().splitlines()]
    events=[r for r in records if r.get('name')=='mapping_frame']
    q=np.array([r['qpos'] for r in events])
    env=gym.make('myoHandPoseFixed-v0');m=env.unwrapped.mj_model;d=env.unwrapped.mj_data
    bid=lambda n:mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,n)
    joints={}
    for j in range(m.njnt):
        name=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_JOINT,j);col=int(m.jnt_qposadr[j]);lo,hi=m.jnt_range[j]
        joints[name]={'baseline_upper_fraction':float(np.mean(abs(q[:,col]-hi)<1e-5)),'candidate_upper_fraction':float(np.mean(abs(candidate[:,col]-hi)<1e-5)),'baseline_lower_fraction':float(np.mean(abs(q[:,col]-lo)<1e-5)),'candidate_lower_fraction':float(np.mean(abs(candidate[:,col]-lo)<1e-5))}
    geometry={}
    for digit,finger in enumerate(('INDEX','MIDDLE','RING','PINKY'),2):
        f=[];mvals=[]
        for frame in ref['frames']:
            points=[np.asarray(frame[f'RightFinger{digit}{part}']['world'])[:3,3] for part in ('Metacarpal','Proximal','Medial','Distal','Tip')]
            f.append([bend(*points[i:i+3]) for i in range(3)])
        for row in q:
            d.qpos[:]=row;mujoco.mj_forward(m,d)
            ids=[bid(n) for n in ({2:'secondmc',3:'thirdmc',4:'fourthmc',5:'fifthmc'}[digit],f'proxph{digit}',f'midph{digit}',f'distph{digit}')]
            tip=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_SITE,{2:'IFtip',3:'MFtip',4:'RFtip',5:'LFtip'}[digit])
            points=list(d.xpos[ids])+[d.site_xpos[tip]]
            mvals.append([bend(*points[i:i+3]) for i in range(3)])
        geometry[finger]={label:{'p05_deg':np.degrees(np.percentile(v,5,axis=0)).tolist(),'p95_deg':np.degrees(np.percentile(v,95,axis=0)).tolist()} for label,v in [('reference',f),('mapped',mvals)]}
    wrist={}
    for label in ('reference','mapped'):
        relative=[];forearm=[]
        if label=='reference':
            for f in ref['frames']:
                a=rotation(f['RightForeArm']['world']);h=rotation(f['RightHand']['world']);relative.append(a.T@h);forearm.append(a)
        else:
            for row in q:
                d.qpos[:]=row;mujoco.mj_forward(m,d)
                a=d.xmat[bid('radius')].reshape(3,3).copy();h=d.xmat[bid('lunate')].reshape(3,3).copy();relative.append(a.T@h);forearm.append(a)
        wrist[label]={}
        for name,values in [('hand_relative_forearm',relative),('forearm_world',forearm)]:
            angles=np.degrees(Rotation.from_matrix(np.asarray(values)[0].T@np.asarray(values)).magnitude())
            wrist[label][name]={'p95_excursion_from_first_deg':float(np.percentile(angles,95)),'max_excursion_from_first_deg':float(max(angles))}
    additional={'geometric_bends':geometry,'wrist':wrist,'candidate_joint_limits':joints,
        'wrist_caution':'Excursions relative to each session first sample; these are not aligned anatomical angular errors. World forearm includes whole-arm motion. FBX first pose is not verified handshake neutral.',
        'fbx_fk_scene_validation':{'sample_indices':[0,30,347,693],'max_matrix_error':7.105427357601002e-14}}
    dump(out/'additional_audit_metrics.json',additional)
    lines=['# Audit Rokoko ke MyoHand', '', 'Tanggal: 26 Oktober 2026. Handoff dipakai sebagai konteks; isi workspace diverifikasi ulang.', '',
        '## Ketersediaan dan validasi', '',
        'Semua modul smartglove/control yang disebut hilang dalam handoff tersedia. XML myohand_pose.xml beserta include/mesh tersedia di .venv/Lib/site-packages/myosuite. Blender, bpy dan FBX SDK tidak tersedia; skrip extract_myohand_right.py/README lama juga tidak ditemukan. Evaluator ufbx 0.0.5 dan Matplotlib 3.11.2 dipasang ke .venv.',
        'FBX: 694 sampel, 30 FPS, 23,1 s, unit 0,01 m. Rekaman: 523 frame sensor, 463 event mapping; frame 523 tidak memiliki event mapping. Sensor xyzw; filtered_quaternions wxyz. Validasi nama joint terhadap alamat qpos berhasil. Replay stored cocok persis (selisih maksimum qpos 0) pada nomor frame event mapping.',
        'Ekstraksi menyimpan hierarki, default/rest transform lokal dan dunia serta transform animasi. ufbx mengevaluasi pre/post rotation dan pivot; FK meliputi semua ancestor. Empat sampel diuji terhadap evaluate_scene: error matriks maksimum 7,11e-14. evaluate_scene berulang memicu crash native Windows; evaluator memakai evaluate_transform dan subprocess terisolasi. Default/rest FBX bukan asumsi pose netral anatomi.', '',
        '## Metode perbandingan', '',
        'Posisi dikonversi ke meter. Origin di wrist; skala jarak wrist ke MCP tengah; sumbu longitudinal menuju MCP tengah, lateral little ke index, normal cross product bertangan kanan. Lima fingertip diekspor pada frame telapak. Thumb basis berbeda antarmodel sehingga metrik thumb bersifat eksploratif. Sudut geometri tiga segmen disimpan terpisah dari sudut joint MuJoCo.',
        'Closure didapat dari jarak tip-wrist rata-rata empat jari, dinormalisasi persentil 5?95. Savitzky-Golay 0,25 s hanya untuk segmentasi offline. Ekstrem dengan prominence 0,3 menentukan transisi membuka/menutup; tiap transisi diresample ke 101 titik waktu fase. Baseline: menutup 4/4 dan membuka 5/4 (Studio/MyoHand). Siklus tepi tidak lengkap dikecualikan. Ini alignment fase sederhana; tidak menyatakan DTW atau estimasi latensi. Grafik mean ? standar deviasi antarsiklus.',
        'Eksperimen A/B membekukan jendela transisi baseline. Ini penting agar perubahan segmentasi tidak terlihat sebagai perbaikan model. Perbedaan geometri, kecepatan dalam fase, dan sesi tetap berkontribusi pada RMSE.', '',
        '## Temuan audit', '',
        '1. Kalibrasi stabil dalam sesi ini: deviasi maksimum sampel open 0,1091? dan grip 0,03685?. Tidak ada bukti noise kalibrasi besar. Namun 30 sampel tanpa pemeriksaan stabilitas/kelengkapan semua sensor rentan untuk sesi lain; axis closure memakai orientasi absolut jari, sehingga gerak tangan/lengan berpotensi tercampur. Ini temuan rumus, belum atribusi numerik untuk error sesi ini.',
        '2. Grip memakai batas ROM model sebagai closed endpoint. Tiga joint fleksi setiap jari ditentukan oleh progres yang sama; SVD menunjukkan komponen pertama menjelaskan 100% variasi pada kelima jari. Telunjuk MCP/PIP/DIP masing-masing 90,0002? ROM. Batas bawah 42,55%, batas atas 31,53%. Dwell pada batas terbuka bukan otomatis error, tetapi sinkronisasi sempurna tiga joint adalah prior algoritmik yang kuat.',
        '3. IK orientasi distal tidak mengidentifikasi distribusi fleksi tiga joint secara unik. Setelah grip berhasil, fleksi hasil IK ditimpa, termasuk coupling DIP/PIP=2/3. IK hanya menyumbang abduksi dan warm start. MCP abduksi tengah di batas 78,19% (bawah 51,62%, atas 26,57%): prioritas audit residual IK dan sumbu lokal. Target jari memakai orientasi dunia yang dikalibrasi sekali; wrist berubah terlebih dahulu, sedangkan fleksi hasil IK kemudian diganti. Target IK dan pose akhir dapat tidak konsisten.',
        '4. Quaternion SLERP sudah menangani tanda q/-q dan normalisasi. Interval median 58,362 ms (17,13 Hz), maksimum 122,918 ms. Filter 12 Hz memberi bobot baru 0,987727; p95 selisih raw-filter jari hanya 0,157?0,337?. Efek smoothing kecil; belum ada bukti filter menyebabkan lag besar. Runtime membuang paket antrean agar mengejar paket terbaru; rekaman processed tidak dapat merekonstruksi paket yang hilang.',
        '5. Wrist menggunakan hand relatif forearm dan alignment sumbu; file axes rekaman tidak memuat forearm_twist untuk xyzw, jadi pro_sup masih dari rotasi relatif. Rotasi pronasi bersama hand+forearm akan hilang dari kanal ini. Memakai twist perlu sapuan forearm yang benar. Cabang twist juga melakukan clamp pro_sup sebelum menghitung flag saturated sehingga flag dapat gagal mendeteksi clipping; cabang itu tidak aktif di rekaman ini. load_axes belum memvalidasi ortogonalitas/norm/kualitas; fit_alignment mempertahankan fleksi dan mengortogonalkan deviasi (bukan Kabsch least-squares seperti deskripsi modul).',
        '6. Runtime menetapkan qpos, menolkan qvel, lalu mj_forward; tidak menjalankan mj_step. Aktivasi otot adalah heuristik berbasis progres jari, bukan controller tracking dinamis atau EMG. Aktivasi PT/PQ berasal dari thumb_abduction, bukan pro_sup. Naturalitas kinematik dan kontrol dinamis harus dinilai terpisah.', '',
        '## Perubahan algoritma tahap pertama', '',
        'tools/experiment_hand_endpoints.py menyediakan subclass mapper khusus eksperimen. calibrate_grip tetap menghitung progres asli; endpoint fleksi empat jari diganti excursion geometris persentil 5?95 FBX, dibatasi ROM model. Ibu jari dan runtime produksi dipertahankan. DIP referensi sekitar 56,7?60,7?, sehingga baseline 90? diuji tanpa mengubah filter, wrist dan solver sekaligus. Ini prior dari clip yang sama, bukan kalibrasi anatomi tervalidasi.', '',
        '| Fase | Jari | RMSE baseline | RMSE kandidat | Penurunan |','|---|---|---:|---:|---:|']
    for phase in ('closing','opening'):
        a=metrics['comparison'][phase]['distance_profile_rmse_hand_lengths'];b=exp['comparison'][phase]['rmse_hand_lengths']
        for finger,x,y in zip(('THUMB','INDEX','MIDDLE','RING','PINKY'),a,b):lines.append(f'| {phase} | {finger} | {x:.4f} | {y:.4f} | {(x-y)/x*100:.1f}% |')
    lines += ['', 'Satuan RMSE: panjang telapak. Kandidat tetap satu progres per jari, sehingga belum memperbaiki urutan fleksi antarsendi atau clipping abduksi. Kandidat tidak diaktifkan sebagai default. Penurunan pada data pengembangan ini belum membuktikan naturalitas pada sesi baru.', '',
        '## Tahap berikut dan kriteria penerimaan', '',
        '1. Rekam sesi baru terpisah, endpoint grip nyaman pengguna, dan metrik residual orientasi IK sebelum/sesudah overwrite. Uji endpoint kandidat pada sesi holdout sebelum promosi.',
        '2. Tambahkan mapper hand-relative opsional dengan neutral HAND serta grip HAND; butuh memperluas payload kalibrasi. Validasi invariansi terhadap rotasi kaku seluruh lengan, lalu replay A/B. Jangan ubah interpretasi kalibrasi lama diam-diam.',
        '3. Modelkan distribusi fleksi dengan kurva per joint teratur/monoton dan prior DIP/PIP yang konsisten; selesaikan abduksi dengan fleksi akhir terkunci, agar solver tidak mengkompensasi fleksi yang nanti ditimpa. Terima hanya bila limit occupancy abduksi dan residual turun tanpa mengurangi excursion gesture berlebihan.',
        '4. Sweep cutoff filter secara terpisah setelah mapping benar; uji jitter pose diam dan lag sensor-to-target dengan sinyal sesi sama. Capture all dan glove clock diperlukan untuk menilai kehilangan paket. Jerk hasil resampling disimpan tetapi sensitif noise dan laju sampling.',
        '5. Kalibrasikan sumbu wrist/pronasi dengan sapuan terpisah, validasi kualitas serta flag saturation. Implementasi tracking otot membutuhkan mj_step dan log actual qpos; jangan menyimpulkan kualitas kontrol otot dari ctrl saat ini.', '',
        '## Reproduksi dan pengujian', '',
        'Jalankan dari root dengan Python .venv:', '', '```powershell', '.\\.venv\\Scripts\\python.exe tools/evaluate_hand_motion.py', '.\\.venv\\Scripts\\python.exe tools/experiment_hand_endpoints.py', '.\\.venv\\Scripts\\python.exe tools/audit_hand_baseline.py', '.\\.venv\\Scripts\\python.exe -m pytest test/test_motion_evaluation.py -q -p no:cacheprovider', '```', '',
        'Cache ekstraksi diverifikasi terhadap SHA-256 FBX. File input dan produksi tidak diubah. requirements-motion-evaluation.txt mencatat dependensi tambahan. Tes awal runtime/replay/wrist/recorder: 63 passed; tes baru invariansi koordinat, fase dua durasi, dan bend geometri: 3 passed. Warning Gymnasium berupa konversi presisi bounds float32.',
        'Artefak: right_hand_forearm_reference.json, normalized_motion.json, comparison_metrics.json, additional_audit_metrics.json, endpoint_experiment_metrics.json, endpoint_experiment_qpos.json; grafik phase_comparison.png, closure_baseline.png, endpoint_experiment.png.', '',
        'Batas kesimpulan: Rokoko Studio bukan ground truth biomekanika; dua sesi tidak identik; posisi wrist dan anatomi model berbeda; perbandingan sudut wrist memakai excursion per sesi, bukan error sumbu anatomis yang sudah selaras. Grafik/profile fase tidak mengukur latensi absolut.']
    (out/'analysis_report.md').write_text('\n'.join(lines),encoding='utf-8')
    env.close()

if __name__=='__main__':main()
