# Alur proyek dan diagnosis pergelangan

## Pembaruan: putaran aktif tanpa sapuan pronasi tambahan

Mapper kini mengaktifkan estimasi sumbu putar secara default jika sumbu fleksi/deviasi tersedia. Saat kalibrasi terbuka, sumbu pro_sup model dipindahkan ke frame sensor HAND melalui alignment.T, lalu ke frame sensor FOREARM melalui orientasi relatif netral. Twist FOREARM terhadap sumbu ini menggerakkan pro_sup, sedangkan fleksi/deviasi tetap dihitung dari HAND relatif FOREARM. Ini memakai konfigurasi xyzw yang sudah tersedia tanpa memasukkan rekaman putar berkualitas rendah.

GUI menampilkan `PS estimated` untuk estimasi ini. Entri forearm_twist hasil sapuan khusus, jika tersedia, tetap diprioritaskan. Pemanggil dapat memilih perilaku relatif lama dengan `infer_forearm_twist=False`. Tanpa sumbu fleksi/deviasi maupun forearm_twist, fallback tetap relatif dengan peringatan.

Mulai ulang `main.py`, lakukan kalibrasi terbuka dengan pose berjabat tangan dan lengan bertumpu, lalu kalibrasi genggam. Putar telapak ke atas/bawah sambil menjaga arah lengan tetap. Perhatikan angka PS di GUI dan putaran radius/tangan di viewer. Estimasi bergantung pada kesesuaian pemasangan sensor dengan sapuan fleksi/deviasi lama; hasil nyata belum diuji dengan glove. Kalibrasi khusus di bawah tetap dianjurkan untuk memperbaiki akurasi bila pemasangan berubah. Temuan konfigurasi tanpa kanal twist pada bagian berikut menjelaskan kondisi sebelum pembaruan ini.

## Alur aplikasi

- `main.py` membuka `GUI/smartglove_gui.py`. GUI menerima UDP, melakukan kalibrasi terbuka/genggam, memfilter quaternion, memetakan wrist lalu jari, menghitung kontrol otot dan fatigue. State qpos/ctrl dikirim ke proses `GUI/full_viewer_worker.py`.
- `cli.py` (juga melalui `smartglove_mujoco.py`) menjalankan live atau replay JSONL melalui `smartglove/runtime.py`. Runtime dan GUI menggunakan mapper yang sama, tetapi loop GUI terpisah dari Pipeline runtime.
- `smartglove_realtime.py` memvalidasi paket 624 byte: lima jari, FOREARM, HAND. Posisi sensor belum digunakan untuk menggerakkan model; pemetaan menggunakan quaternion. GUI menggunakan urutan xyzw; CLI bisa memilih xyzw/wxyz.
- `smartglove/calibration.py` mengelola kalibrasi runtime/replay. Pose terbuka menentukan referensi sensor/model; genggaman menentukan rentang fleksi jari. Kalibrasi genggam tidak mengukur sumbu putar pergelangan.
- `smartglove/quaternion.py` menyediakan operasi quaternion dan low-pass filter. `wrist_alignment.py` memuat sumbu sapuan serta menyelaraskannya dengan model.
- `wrist_mapping.py` memetakan pro_sup, deviation, flexion. Fleksi/deviasi berasal dari orientasi HAND relatif terhadap FOREARM. Jika tersedia, forearm_twist mengukur putaran sensor FOREARM dari pose netral.
- `mapping.py` memakai IK orientasi ujung jari, coupling distal dan rentang genggaman. Karena sensor jari dipetakan dari orientasi absolut, gerak bersama lengan dapat memengaruhi estimasi jari; ini perlu uji terpisah saat wrist sudah terkalibrasi.
- `control/smartglove_muscle_control.py` menghitung aktivasi dari pose, bukan EMG. `fatigue/model.py` memodifikasi kontrol otot pada GUI. Viewer menulis qpos lalu mj_forward; gerak saat ini dikendalikan secara kinematik, bukan hasil integrasi otot dengan mj_step.
- `recorder.py` dan `frame.py` menyimpan/membaca sensor, hasil mapping, waktu dan event. `capture_finger/` menangkap paket mentah. `tools/` dan `evaluation/` berisi kalibrasi serta evaluasi offline, `inspect/` inspeksi model, `test/` pengujian.
- `experimental_abduction.py` adalah mapper eksperimen offline/opt-in; runtime dan GUI tidak mengimpornya. Mengubah berkas ini saja tidak mengubah gerak viewer utama.

## Temuan dari kode dan data lokal

Konfigurasi aktif `smartglove/wrist_axes.json` tidak memiliki forearm_twist, baik untuk xyzw maupun wxyz. Entri pro_sup pada wxyz bukan forearm_twist: keduanya berasal dari frame/pengukuran berbeda dan tidak boleh sekadar diganti namanya. Putaran bersama HAND dan FOREARM menghilang pada rotasi relatif; tes sintetis mereproduksi masalah ini dan membuktikan kanal twist dapat memulihkannya.

Sapuan xyzw kanan_kiri_baru dan atas_bawah_baru mereproduksi sumbu fleksi/deviasi yang tersimpan (kemurnian 98,5% / 97,6%, sudut antar sumbu 93,2 derajat). Ini belum membuktikan arah tanda sesuai tangan pengguna saat ini. Pose pemasangan sensor harus sama dengan saat merekam sapuan.

| Rekaman putar xyzw | Kemurnian sumbu FOREARM | Rentang total | Selisih sumbu dunia FOREARM/HAND |
| --- | ---: | ---: | ---: |
| putar | 74,8% | 33,9 derajat | 66,8 derajat |
| putar_baru | 50,3% | 41,8 derajat | 14,5 derajat |
| putar_new | 90,5% | 25,1 derajat | 78,4 derajat |

Tidak ada yang memenuhi seluruh ambang pada alat kalibrasi: kemurnian minimal 90%, rentang minimal 40 derajat, dan selisih sumbu dunia maksimal 25 derajat. Kandidat `evaluation/wrist_axes_candidate.json` hanya diagnostik; tidak diaktifkan.

Model MyoHand memiliki batas pro_sup sekitar -90..+90, deviation -10..+25, flexion -45..+45 derajat. Clamp dapat membuat gerak tampak berhenti. Sudut status ditampilkan relatif terhadap pose netral model; batas absolut dan offset netral perlu dibedakan. Tes sintetis memvalidasi fleksi/deviasi serta isolasi saturasi, tetapi validasi gerak nyata masih memerlukan glove.

## Kalibrasi untuk mengaktifkan putaran

1. Rekam ulang tiga sapuan dengan pemasangan sensor yang sama. Awali setiap rekaman dengan sedikitnya 10 paket pose netral berjabat tangan: jari lurus, ibu jari di atas, tangan segaris lengan bawah.
2. Fleksi/ekstensi: tekuk telapak menuju/menjauhi sisi telapak. Deviasi: gerakkan ke sisi ibu jari/kelingking. Tumpukan lengan bawah dan hindari mencampur gerakan.
3. Pronasi/supinasi: siku dan arah panjang lengan tetap; putar telapak ke bawah/atas. Pastikan sensor manset ikut berputar. Ambil sapuan yang cukup panjang, bukan hanya sekitar 2,4 detik dari 200 paket jika belum mencakup pose netral dan kedua arah.
4. Jalankan dari root, ganti direktori dengan hasil rekaman baru:

```powershell
.\.venv\Scripts\python.exe tools/calibrate_wrist_axes.py --flexion DIR_FLEKSI --deviation DIR_DEVIASI --pronation DIR_PUTAR --order xyzw --out evaluation/wrist_axes_new.json
```

5. Periksa kualitas, lalu gunakan hasil yang lolos sebagai konfigurasi `smartglove/wrist_axes.json` (pertahankan entri order lain bila diperlukan). Alat saat ini tetap menulis hasil meskipun kualitas rendah; tulisan `[OK]` hanya berarti file tersimpan.
6. Mulai ulang aplikasi dan kalibrasi pose terbuka/genggam. Uji tiap gerakan secara terpisah. Jika arah terbalik, hitung ulang dengan --flip-flexion, --flip-deviation atau --flip-pronation sesuai gerakan yang salah. Pastikan jari tidak ikut berubah saat wrist digerakkan dengan jari tetap lurus.

Dengan hanya FOREARM dan HAND, twist tidak dapat membedakan semua gerakan seluruh lengan dari pronasi anatomis. Validasi awal harus dengan arah lengan tetap. Gerakan lengan bebas memerlukan referensi tambahan atau estimasi orientasi lengan yang terpisah.
