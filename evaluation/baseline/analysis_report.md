# Audit Rokoko ke MyoHand

Tanggal: 26 Oktober 2026. Handoff dipakai sebagai konteks; isi workspace diverifikasi ulang.

## Ketersediaan dan validasi

Semua modul smartglove/control yang disebut hilang dalam handoff tersedia. XML myohand_pose.xml beserta include/mesh tersedia di .venv/Lib/site-packages/myosuite. Blender, bpy dan FBX SDK tidak tersedia; skrip extract_myohand_right.py/README lama juga tidak ditemukan. Evaluator ufbx 0.0.5 dan Matplotlib 3.11.2 dipasang ke .venv.
FBX: 694 sampel, 30 FPS, 23,1 s, unit 0,01 m. Rekaman: 523 frame sensor, 463 event mapping; frame 523 tidak memiliki event mapping. Sensor xyzw; filtered_quaternions wxyz. Validasi nama joint terhadap alamat qpos berhasil. Replay stored cocok persis (selisih maksimum qpos 0) pada nomor frame event mapping.
Ekstraksi menyimpan hierarki, default/rest transform lokal dan dunia serta transform animasi. ufbx mengevaluasi pre/post rotation dan pivot; FK meliputi semua ancestor. Empat sampel diuji terhadap evaluate_scene: error matriks maksimum 7,11e-14. evaluate_scene berulang memicu crash native Windows; evaluator memakai evaluate_transform dan subprocess terisolasi. Default/rest FBX bukan asumsi pose netral anatomi.

## Metode perbandingan

Posisi dikonversi ke meter. Origin di wrist; skala jarak wrist ke MCP tengah; sumbu longitudinal menuju MCP tengah, lateral little ke index, normal cross product bertangan kanan. Lima fingertip diekspor pada frame telapak. Thumb basis berbeda antarmodel sehingga metrik thumb bersifat eksploratif. Sudut geometri tiga segmen disimpan terpisah dari sudut joint MuJoCo.
Closure didapat dari jarak tip-wrist rata-rata empat jari, dinormalisasi persentil 5?95. Savitzky-Golay 0,25 s hanya untuk segmentasi offline. Ekstrem dengan prominence 0,3 menentukan transisi membuka/menutup; tiap transisi diresample ke 101 titik waktu fase. Baseline: menutup 4/4 dan membuka 5/4 (Studio/MyoHand). Siklus tepi tidak lengkap dikecualikan. Ini alignment fase sederhana; tidak menyatakan DTW atau estimasi latensi. Grafik mean ? standar deviasi antarsiklus.
Eksperimen A/B membekukan jendela transisi baseline. Ini penting agar perubahan segmentasi tidak terlihat sebagai perbaikan model. Perbedaan geometri, kecepatan dalam fase, dan sesi tetap berkontribusi pada RMSE.

## Temuan audit

1. Kalibrasi stabil dalam sesi ini: deviasi maksimum sampel open 0,1091? dan grip 0,03685?. Tidak ada bukti noise kalibrasi besar. Namun 30 sampel tanpa pemeriksaan stabilitas/kelengkapan semua sensor rentan untuk sesi lain; axis closure memakai orientasi absolut jari, sehingga gerak tangan/lengan berpotensi tercampur. Ini temuan rumus, belum atribusi numerik untuk error sesi ini.
2. Grip memakai batas ROM model sebagai closed endpoint. Tiga joint fleksi setiap jari ditentukan oleh progres yang sama; SVD menunjukkan komponen pertama menjelaskan 100% variasi pada kelima jari. Telunjuk MCP/PIP/DIP masing-masing 90,0002? ROM. Batas bawah 42,55%, batas atas 31,53%. Dwell pada batas terbuka bukan otomatis error, tetapi sinkronisasi sempurna tiga joint adalah prior algoritmik yang kuat.
3. IK orientasi distal tidak mengidentifikasi distribusi fleksi tiga joint secara unik. Setelah grip berhasil, fleksi hasil IK ditimpa, termasuk coupling DIP/PIP=2/3. IK hanya menyumbang abduksi dan warm start. MCP abduksi tengah di batas 78,19% (bawah 51,62%, atas 26,57%): prioritas audit residual IK dan sumbu lokal. Target jari memakai orientasi dunia yang dikalibrasi sekali; wrist berubah terlebih dahulu, sedangkan fleksi hasil IK kemudian diganti. Target IK dan pose akhir dapat tidak konsisten.
4. Quaternion SLERP sudah menangani tanda q/-q dan normalisasi. Interval median 58,362 ms (17,13 Hz), maksimum 122,918 ms. Filter 12 Hz memberi bobot baru 0,987727; p95 selisih raw-filter jari hanya 0,157?0,337?. Efek smoothing kecil; belum ada bukti filter menyebabkan lag besar. Runtime membuang paket antrean agar mengejar paket terbaru; rekaman processed tidak dapat merekonstruksi paket yang hilang.
5. Wrist menggunakan hand relatif forearm dan alignment sumbu; file axes rekaman tidak memuat forearm_twist untuk xyzw, jadi pro_sup masih dari rotasi relatif. Rotasi pronasi bersama hand+forearm akan hilang dari kanal ini. Memakai twist perlu sapuan forearm yang benar. Cabang twist juga melakukan clamp pro_sup sebelum menghitung flag saturated sehingga flag dapat gagal mendeteksi clipping; cabang itu tidak aktif di rekaman ini. load_axes belum memvalidasi ortogonalitas/norm/kualitas; fit_alignment mempertahankan fleksi dan mengortogonalkan deviasi (bukan Kabsch least-squares seperti deskripsi modul).
6. Runtime menetapkan qpos, menolkan qvel, lalu mj_forward; tidak menjalankan mj_step. Aktivasi otot adalah heuristik berbasis progres jari, bukan controller tracking dinamis atau EMG. Aktivasi PT/PQ berasal dari thumb_abduction, bukan pro_sup. Naturalitas kinematik dan kontrol dinamis harus dinilai terpisah.

## Perubahan algoritma tahap pertama

tools/experiment_hand_endpoints.py menyediakan subclass mapper khusus eksperimen. calibrate_grip tetap menghitung progres asli; endpoint fleksi empat jari diganti excursion geometris persentil 5?95 FBX, dibatasi ROM model. Ibu jari dan runtime produksi dipertahankan. DIP referensi sekitar 56,7?60,7?, sehingga baseline 90? diuji tanpa mengubah filter, wrist dan solver sekaligus. Ini prior dari clip yang sama, bukan kalibrasi anatomi tervalidasi.

| Fase | Jari | RMSE baseline | RMSE kandidat | Penurunan |
|---|---|---:|---:|---:|
| closing | THUMB | 0.2896 | 0.2896 | 0.0% |
| closing | INDEX | 0.3880 | 0.3751 | 3.3% |
| closing | MIDDLE | 0.3999 | 0.3854 | 3.6% |
| closing | RING | 0.3797 | 0.3650 | 3.9% |
| closing | PINKY | 0.3217 | 0.3085 | 4.1% |
| opening | THUMB | 0.2702 | 0.2702 | -0.0% |
| opening | INDEX | 0.2623 | 0.2191 | 16.4% |
| opening | MIDDLE | 0.2356 | 0.2131 | 9.5% |
| opening | RING | 0.2270 | 0.2013 | 11.3% |
| opening | PINKY | 0.2020 | 0.1813 | 10.2% |

Satuan RMSE: panjang telapak. Kandidat tetap satu progres per jari, sehingga belum memperbaiki urutan fleksi antarsendi atau clipping abduksi. Kandidat tidak diaktifkan sebagai default. Penurunan pada data pengembangan ini belum membuktikan naturalitas pada sesi baru.

## Tahap berikut dan kriteria penerimaan

1. Rekam sesi baru terpisah, endpoint grip nyaman pengguna, dan metrik residual orientasi IK sebelum/sesudah overwrite. Uji endpoint kandidat pada sesi holdout sebelum promosi.
2. Tambahkan mapper hand-relative opsional dengan neutral HAND serta grip HAND; butuh memperluas payload kalibrasi. Validasi invariansi terhadap rotasi kaku seluruh lengan, lalu replay A/B. Jangan ubah interpretasi kalibrasi lama diam-diam.
3. Modelkan distribusi fleksi dengan kurva per joint teratur/monoton dan prior DIP/PIP yang konsisten; selesaikan abduksi dengan fleksi akhir terkunci, agar solver tidak mengkompensasi fleksi yang nanti ditimpa. Terima hanya bila limit occupancy abduksi dan residual turun tanpa mengurangi excursion gesture berlebihan.
4. Sweep cutoff filter secara terpisah setelah mapping benar; uji jitter pose diam dan lag sensor-to-target dengan sinyal sesi sama. Capture all dan glove clock diperlukan untuk menilai kehilangan paket. Jerk hasil resampling disimpan tetapi sensitif noise dan laju sampling.
5. Kalibrasikan sumbu wrist/pronasi dengan sapuan terpisah, validasi kualitas serta flag saturation. Implementasi tracking otot membutuhkan mj_step dan log actual qpos; jangan menyimpulkan kualitas kontrol otot dari ctrl saat ini.

## Reproduksi dan pengujian

Jalankan dari root dengan Python .venv:

```powershell
.\.venv\Scripts\python.exe tools/evaluate_hand_motion.py
.\.venv\Scripts\python.exe tools/experiment_hand_endpoints.py
.\.venv\Scripts\python.exe tools/audit_hand_baseline.py
.\.venv\Scripts\python.exe -m pytest test/test_motion_evaluation.py -q -p no:cacheprovider
```

Cache ekstraksi diverifikasi terhadap SHA-256 FBX. File input dan produksi tidak diubah. requirements-motion-evaluation.txt mencatat dependensi tambahan. Tes awal runtime/replay/wrist/recorder: 63 passed; tes baru invariansi koordinat, fase dua durasi, dan bend geometri: 3 passed. Warning Gymnasium berupa konversi presisi bounds float32.
Artefak: right_hand_forearm_reference.json, normalized_motion.json, comparison_metrics.json, additional_audit_metrics.json, endpoint_experiment_metrics.json, endpoint_experiment_qpos.json; grafik phase_comparison.png, closure_baseline.png, endpoint_experiment.png.

Batas kesimpulan: Rokoko Studio bukan ground truth biomekanika; dua sesi tidak identik; posisi wrist dan anatomi model berbeda; perbandingan sudut wrist memakai excursion per sesi, bukan error sumbu anatomis yang sudah selaras. Grafik/profile fase tidak mengukur latensi absolut.