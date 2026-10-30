# Diagnostik IK abduksi — 26 Oktober 2026

Kandidat belum diterima sebagai default. Residual orientasi turun, tetapi clipping dan profil fingertip tidak membaik secara konsisten.

## Eksperimen terkontrol

Baseline memakai HandQposMapper asli dengan subclass observasi. Kandidat memakai hasil IK lama sebagai inisialisasi, menetapkan fleksi akhir hasil kalibrasi, lalu memperbaiki hanya abduksi dengan optimasi skalar bounded pada residual geodesik SO(3). Pencarian menyertakan grid, minimum lokal, endpoint, dan pose awal. Fleksi serta wrist dikunci. Jari tanpa kalibrasi grip tetap memakai fallback asli.
Endpoint fleksi, filter, wrist, kalibrasi dan target sensor tetap seperti rekaman. Eksperimen endpoint FBX sebelumnya tidak digabung agar pengaruh abduksi dapat dipisahkan. Runtime produksi, CLI dan default tidak diubah.
Replay: 463 frame mapping; selisih maksimum baseline terhadap rekaman 0.0e+00 rad. Selisih fleksi/wrist kandidat terhadap baseline 0.0e+00 rad. Frame mentah 523 tanpa event mapping dikecualikan dengan nomor frame, bukan pemotongan array.

## Pengaruh overwrite fleksi pada baseline

Residual adalah sudut rotasi terpendek dari orientasi distal model menuju target orientasi sensor yang sudah dikalibrasi. Ini error terhadap target mapper, bukan error biomekanika terhadap FBX.

| Jari | Setelah IK, sebelum overwrite | Setelah overwrite | Perubahan rata-rata | Frame memburuk |
|---|---:|---:|---:|---:|
| THUMB | 19.61° | 100.47° | +80.86° | 100.0% |
| INDEX | 78.66° | 89.34° | +10.69° | 45.6% |
| MIDDLE | 54.37° | 87.58° | +33.21° | 80.6% |
| RING | 94.60° | 82.46° | -12.14° | 20.7% |
| PINKY | 63.23° | 80.91° | +17.69° | 93.5% |

Overwrite tidak selalu memperburuk residual: jari manis justru membaik rata-rata. Jadi penyebabnya tidak dapat digeneralisasi sebagai satu bug yang sama pada semua jari.

## Hasil kandidat

| Jari | Residual baseline → kandidat | Di batas baseline → kandidat | RMS kecepatan abduksi baseline → kandidat |
|---|---:|---:|---:|
| THUMB | 100.47° → 77.40° | 69.8% → 56.8% | 94.3 → 44.5 °/s |
| INDEX | 89.34° → 85.01° | 56.2% → 53.3% | 19.8 → 29.6 °/s |
| MIDDLE | 87.58° → 79.13° | 78.2% → 76.0% | 75.4 → 76.3 °/s |
| RING | 82.46° → 80.72° | 39.3% → 53.6% | 121.9 → 90.0 °/s |
| PINKY | 80.91° → 80.37° | 56.8% → 55.9% | 43.9 → 22.5 °/s |

Optimasi orientasi saja dapat memilih batas anatomis lebih sering. Jari manis menjadi contoh: residual turun sedikit tetapi occupancy batas naik. Kecepatan RMS adalah ukuran perubahan sepanjang seluruh sesi, bukan jitter pose diam yang sudah dipisahkan.

## Profil fingertip terhadap Studio

Jendela transisi baseline dibekukan; tiap transisi diresample ke 101 titik fase. Satuan RMSE adalah panjang telapak. Tidak ada pemasangan frame langsung antarrekaman.

| Jari | Menutup baseline → kandidat | Membuka baseline → kandidat |
|---|---:|---:|
| THUMB | 0.2896 → 0.1056 | 0.2702 → 0.0682 |
| INDEX | 0.3880 → 0.3984 | 0.2623 → 0.2829 |
| MIDDLE | 0.3999 → 0.4004 | 0.2356 → 0.2342 |
| RING | 0.3797 → 0.3788 | 0.2270 → 0.2269 |
| PINKY | 0.3217 → 0.3197 | 0.2020 → 0.2013 |

Ibu jari menunjukkan perbaikan besar pada metrik jarak, tetapi geometri thumb antarmodel berbeda dan jarak radial tidak mengukur arah gerak. Telunjuk memburuk di kedua fase. Empat jari lain tidak menunjukkan peningkatan seragam yang cukup untuk mempromosikan kandidat.

## Grafik

- residual_and_abduction.png: kolom kiri residual baseline, kandidat, dan baseline sebelum overwrite; kolom kanan sudut abduksi. Sumbu X detik dari frame mapping pertama; sumbu Y derajat.
- phase_comparison.png: profil jarak tip-wrist/panjang telapak. Studio adalah referensi sesi berbeda; baseline dan kandidat dari replay sesi JSONL yang sama. Sumbu X fase transisi 0–1.

## Implikasi teknis berikutnya

Prioritas berikutnya adalah memvalidasi target orientasi jari dalam frame HAND dan kalibrasi sumbu/arah fleksi. Residual yang mendekati 180° pada sebagian frame menunjukkan ketidaksesuaian target dan pose akhir yang terlalu besar untuk diselesaikan satu joint abduksi. Angka ini tidak membuktikan salah urutan quaternion: baseline telah mereproduksi rekaman dan quaternion dinormalisasi. Perlu uji rotasi kaku bersama tangan/lengan dan pose sintetis dengan sumbu yang diketahui.
Jangan menambahkan regularisasi atau mengurangi ROM semata-mata agar clipping terlihat turun sebelum targetnya tervalidasi. Setelah target konsisten, kandidat IK ini dapat diuji kembali dengan prior abduksi netral dan kontinuitas waktu secara terpisah.

## Pengujian dan reproduksi

Empat kasus tes solver lulus: target reachable dengan q dan -q, target di luar ROM, dan residual non-worsening dengan fleksi/wrist tetap. Tiga tes normalisasi/fase/geometri sebelumnya juga lulus. Replay seluruh 463 event memverifikasi baseline identik, fleksi/wrist identik, batas joint, dan residual final kandidat dibanding baseline.

```powershell
.\.venv\Scripts\python.exe tools/experiment_abduction_ik.py
.\.venv\Scripts\python.exe -m pytest test/test_abduction_ik.py test/test_motion_evaluation.py -q -p no:cacheprovider
```

Keterbatasan: hanya satu sesi replay; target masih orientasi dunia; tidak mengevaluasi dinamika otot; pose thumb memerlukan validasi anatomis tambahan; mean profil antar-sesi tidak sama dengan tracking ground truth.