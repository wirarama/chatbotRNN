# iot_slm_app — struktur modul

`iot_app_v3.py` sebelumnya adalah satu file ±1800 baris. Sekarang file itu
hanya jadi **entry point** tipis (layout halaman + alur routing chat), dan
seluruh logikanya dipecah ke paket `iot_slm_app/` ini, satu modul per
tanggung jawab. Tujuannya supaya, misalnya, mengubah arsitektur SLM/GRU
tidak perlu menggulir file 1800 baris — cukup buka `model.py`.

## Peta ketergantungan (agar tidak ada circular import)

```
config.py  (tidak bergantung pada modul lain)
   ├── model.py       (SLM/GRU + forward pass)
   │      └── engine.py   (memakai forecast_window dari model.py)
   │             └── loader.py  (menggabungkan model.py + engine.py)
   ├── nlp_queries.py (deteksi intent + narasi ranking/cluster/rate)
   │      └── parser.py   (parser tanggal, memanggil nlp_queries.py)
   ├── narration.py   (narasi query temporal biasa)
   ├── charts.py       (semua figure Plotly)
   ├── ui_components.py (stat card, badge notifikasi, tabel per jam)
   ├── rules.py        (mesin rule notifikasi)
   └── live_feed.py    (ingestion streaming live — fitur baru, lihat di bawah)

iot_app_v3.py mengimpor semuanya dan merangkainya di main().
```

## Isi & fungsi tiap file

**`config.py`** — Konfigurasi statis: resolusi `--model`, semua konstanta
(`SENSOR_KEYS`, `SENSOR_UNITS`, kelas notifikasi Decision-Tree, deskripsi
cluster, palet warna chart), dan `configure_page()` yang memanggil
`st.set_page_config(...)` + menyuntik CSS tema putih dengan teks kontras
tinggi. **Edit di sini** untuk mengubah warna/tema, menambah sensor baru,
atau mengubah teks pesan notifikasi Decision-Tree.

**`model.py`** — **INI FILE UNTUK MEMODIFIKASI SLM/RNN.** Berisi kelas
`IoTSLM` (GRU-RNN: proyeksi → GRU bertumpuk → self-attention → decoder
forecast + head anomali) dan `forecast_window()` (satu forward pass
inferensi). Arsitektur di sini **harus identik** dengan yang dipakai
`iot_train.py`, karena bobot terlatih (`slm_weights.pt`) dimuat ke sini
lewat `state_dict`. Tidak ada kode chat, tanggal, atau UI di file ini —
supaya perubahan pada RNN tidak berisiko merusak bagian lain.

**`engine.py`** — Kelas `Engine`: pembungkus data mentah + fitur + label
cluster + Decision-Tree, dengan method `query(start, end, label)` yang
menghasilkan semua statistik/cluster/notifikasi/forecast untuk satu
rentang tanggal (inilah yang menjawab setiap pertanyaan chat), plus
`series()`/`cluster_series()` untuk kebutuhan chart.

**`loader.py`** — `load_models(model_dir)`: satu-satunya fungsi yang
membaca file-file hasil `iot_train.py` dari disk (pickle/JSON) dan
merakit `IoTSLM` + `Engine`. Dipisah dari `model.py`/`engine.py` semata
untuk menghindari circular import (Engine butuh `forecast_window` dari
model.py, loader butuh keduanya).

**`nlp_queries.py`** — Tiga "mesin chatbot mini": query **ranking**
(suhu tertinggi/terendah), **cluster** (list/detail), dan **rate-of-change**
(kenaikan/penurunan tercepat/terlambat). Masing-masing punya fungsi
`detect_*()` untuk mengenali maksud pertanyaan dan `*_query()` untuk
menyusun narasi jawabannya.

**`parser.py`** — Kelas `Parser`: mengubah frasa waktu ("hari ini",
"minggu lalu", dst — Indonesia & Inggris) menjadi rentang tanggal, dan
menjadi titik akses tunggal ke tiga mesin NLP di atas.

**`narration.py`** — `narrate()`: narasi balasan chat standar untuk
pertanyaan temporal biasa ("hari ini", "bandingkan hari ini dan
kemarin"), beserta kata bantu `_td`/`_hd`/`_tw` (deskripsi suhu,
kelembaban, tren).

**`charts.py`** — Semua figure Plotly: `make_charts()` (6 panel
perbandingan), `training_chart()` (kurva training MSE), dan
`live_stream_chart()` — **fitur baru:** grafik garis 2x2 (satu panel per
sensor) untuk tab "📡 Live Stream", lihat bagian Live Streaming di bawah.

**`ui_components.py`** — Komponen Streamlit kecil: `stat_cards()`, `notif_badge()`,
`hourly_table()`.

**`rules.py`** — Mesin rule notifikasi buatan pengguna dari chat. Lihat
bagian di bawah.

**`live_feed.py`** — **Fitur baru: ingestion streaming live.** Membaca
DB SQLite bersama yang ditulis oleh `iot_stream_sender.py` (proses
terpisah, lihat bagian di bawah) dan menambahkan pembacaan baru ke
`Engine` yang sedang berjalan lewat `Engine.append_raw()` /
`Engine.append_window()` (didefinisikan di `engine.py`). Setelah ini
terpasang, `rules.py` dan semua kueri chat "otomatis" melihat data baru
tanpa perubahan kode di modul lain — keduanya hanya pernah membaca
`engine.data`/`engine.X`/`engine.metas`/`engine.cl`, array yang sama
yang di-append oleh modul ini.

## Fitur baru: Rule Notifikasi

Ketik langsung di kotak chat, contoh persis dari permintaan awal:

```
buat rule jika suhu diatas 32C dan humidity diatas 70% maka notif saya
```

Kata kunci yang dikenali (Indonesia & Inggris, tidak case-sensitive):

| Kata kunci | Arti |
|---|---|
| `diatas` / `di atas` / `above` | nilai harus **melebihi** ambang |
| `dibawah` / `di bawah` / `below` | nilai harus **di bawah** ambang |
| `antara X dan Y` / `between X and Y` | nilai harus berada **dalam rentang** |
| `peningkatan` / `kenaikan` sebelum nama sensor | mengubah kondisi jadi **rate-of-change**: kenaikan nilai dibanding ~1 jam lalu |
| `penurunan` sebelum nama sensor | sama seperti di atas tapi untuk **penurunan** |
| `dan` / `atau` antar kondisi | AND / OR (default: AND) |

Rule tersimpan **permanen** di `notification_rules.json` (di root
repo, sejajar dengan `iot_app_v3.py`) — bukan hanya di session Streamlit,
jadi tetap ada setelah aplikasi di-restart.

Setiap kali halaman di-render ulang, `check_rules()` mengecek semua rule
yang aktif terhadap **pembacaan sensor terbaru** yang tersedia di
`engine.data` (baris terakhir dataset yang dimuat). Notifikasi hanya
dikirim ke chat saat kondisi **baru saja menjadi benar** (edge-triggered),
supaya kondisi yang terus-menerus benar tidak membanjiri chat dengan
pesan berulang.

Secara default `iot_app_v3.py` masih memutar ulang dataset historis
tetap dari `iot_train.py`. Kalau **Live Streaming** diaktifkan dari
sidebar (lihat bagian di bawah), `check_rules()` membaca "sampel
terakhir di `engine.data`" yang sudah termasuk pembacaan live terbaru —
titik sambung yang sama persis yang didesain sejak awal untuk fitur ini.

**Menghapus atau menonaktifkan rule dilakukan di tab "Rules"** pada
panel visualisasi (bukan lewat chat) — setiap rule punya kotak centang
Enabled dan tombol Delete. Jika pengguna mengetik sesuatu seperti "hapus
rule ..." di chat, bot akan mengarahkan mereka ke tab Rules, bukan
menghapus apa pun secara langsung.

## Fitur baru: Live Streaming

Sebelumnya `iot_app_v3.py` murni *offline-replay*: dataset yang dimuat
dari `iot_model/` tetap sama sejak halaman pertama dibuka sampai
di-restart. Sekarang tersedia jalur untuk data sensor yang benar-benar
mengalir, lewat dua bagian:

1. **`iot_stream_sender.py`** (di root repo, **proses Python terpisah**,
   dijalankan sendiri lewat `python iot_stream_sender.py`) — mensimulasikan
   pembacaan sensor realistis (pola diurnal suhu/kelembaban/cahaya, siklus
   tekanan mingguan, injeksi fault ~3% — meniru persis `iot_train.py`'s
   `simulate_data()`) dan menulis tiap pembacaan baru ke database SQLite
   bersama (default `./iot_stream.db`).
2. **`iot_slm_app/live_feed.py`** — dipanggil dari dalam `iot_app_v3.py`
   ketika toggle **"Enable live streaming"** di sidebar dinyalakan.
   Membaca baris baru dari DB SQLite tadi dan memanggil
   `Engine.append_raw()` (supaya rule notifikasi langsung bereaksi) lalu,
   begitu cukup riwayat terkumpul untuk satu window penuh (`seq_len`,
   biasanya 24 sampel), `Engine.append_window()` (supaya kueri chat
   seperti "hari ini" dan semua chart/tabel ikut memasukkan data live).

Cara pakai singkat:

```
# Terminal 1 — jalankan sumber data live
python iot_stream_sender.py --continue-model --burst 24

# Terminal 2 — jalankan app seperti biasa, lalu nyalakan
# "Enable live streaming" di sidebar
streamlit run iot_app_v3.py
```

`--continue-model` membuat timeline sender bersambung persis dari akhir
dataset training (`ref_date` di `iot_model/config.json`), dan
`--burst 24` langsung mengisi satu window penuh supaya app tidak perlu
menunggu ~24×interval menit dunia-nyata sebelum window live pertama
muncul. Tanpa argumen, sender mulai dari waktu sekarang dan tetap
memakai `interval_min` dari model yang sama (kalau `iot_model/`
ditemukan) — sidebar akan memperingatkan kalau interval sender ternyata
tidak cocok dengan interval model (mempengaruhi akurasi rule
rate-of-change seperti "peningkatan suhu diatas 2C").

Karena SQLite dengan mode WAL dipakai sebagai titik sambung (bukan
MQTT/HTTP), mengganti sumber data nyata di kemudian hari (mis. ESP32
sungguhan) hanya berarti mengganti apa yang menulis ke DB itu —
`live_feed.py`, `engine.py`, dan semua modul di atasnya tidak perlu
disentuh.

### Sumber data live kedua: `iot_stream_form.py` (form manual)

Selain `iot_stream_sender.py` (otomatis, pola realistis), ada juga
**`iot_stream_form.py`** — form web Flask polos (satu file, tanpa
JS framework) untuk mengirim pembacaan **acak dalam rentang yang Anda
tentukan sendiri per sensor**. Berguna untuk sengaja mendorong satu
sensor keluar batas dan melihat rule notifikasi / Decision-Tree
bereaksi, tanpa menunggu simulator memunculkan fault secara acak.

Menggunakan ulang `open_db()`/`insert_reading()`/`set_meta()` milik
`iot_stream_sender.py` (diimpor langsung, bukan ditulis ulang), jadi
skema tabelnya persis sama — bisa gonta-ganti antara sender dan form
ini ke DB yang sama, timeline tetap nyambung (keduanya selalu
melanjutkan dari timestamp baris terakhir yang sudah ada).

Dua mode pengiriman di form:
- **Kirim jumlah tertentu** — N record acak langsung sekaligus (burst).
- **Mode Start/Stop** — thread background di proses Flask yang terus
  mengirim satu record acak setiap `tick_seconds` detik sampai tombol
  Stop ditekan; status (jumlah terkirim, record terakhir) di-refresh
  otomatis tiap 2 detik di halamannya lewat `fetch()` biasa.

```
pip install flask   # kalau belum ada
python iot_stream_form.py            # default port 5050
# buka http://127.0.0.1:5050/ di browser
```

Proses terpisah juga dari `iot_stream_sender.py` dan `iot_app_v3.py` —
bisa jalan bertiga sekaligus, masing-masing di terminalnya sendiri.

### Tab "📡 Live Stream" — grafik + garis batas rule

Tab baru di panel visualisasi (sejajar dengan Charts/Statistics/dst.)
menampilkan grafik garis 2x2 — satu panel per sensor — untuk data yang
**benar-benar sudah mengalir lewat streaming** (bukan seluruh dataset
historis 60 hari; batasnya `engine._live_base_len`, diset sekali oleh
`live_feed.poll_and_ingest()` saat streaming pertama kali dinyalakan).
Menampilkan sampai `DEFAULT_LIVE_CHART_POINTS` (default 200) pembacaan
live terbaru per sensor, dengan pembacaan bertanda `is_anomaly` ditandai
× merah.

Setiap kondisi **level** (diatas/dibawah/antara) dari setiap rule yang
sedang **enabled** digambar sebagai garis horizontal putus-putus/titik-titik
di panel sensor yang bersangkutan — persis ambang yang sedang dipantau
rule tersebut. Rule yang di-nonaktifkan (`enabled: false`) tidak
digambar sama sekali.

Kondisi **rate-of-change** (peningkatan/penurunan) sengaja **dipisah**
ke grafik BATANG tersendiri di bawahnya — "Laju Perubahan (Rate of
Change)", dari `charts.py`'s `rate_of_change_chart()` — karena ambang
sebuah rule rate-of-change bukan nilai level pada sensor mentah,
melainkan besar perubahan dibanding ~60 menit sebelumnya (persis
kuantitas yang dihitung `rules.py`'s `_condition_value_at()`), jadi
skalanya berbeda dan tidak masuk akal digambar sebagai garis datar di
atas grafik garis sensor mentah. Satu batang per pembacaan live =
`nilai_sekarang - nilai_~60_menit_lalu`; warna panas = naik, warna
dingin = turun. Setiap kondisi rate-of-change dari rule yang **enabled**
menggambar garis ambang di panel sensornya (diterjemahkan ke domain
delta mentah ini oleh `_raw_delta_condition()`, supaya posisinya tepat
sama dengan titik rule itu benar-benar terpicu — lihat docstring fungsi
tersebut untuk detail pembalikan tanda pada mode "penurunan").

## Kapan mengubah file yang mana (ringkasan cepat)

- Mengubah arsitektur GRU/RNN atau cara forecast dihitung → **`model.py`**
- Menambah sensor baru → **`config.py`** (lalu `iot_train.py` juga harus disesuaikan)
- Mengubah tampilan/warna/tema → **`config.py`** (CSS di `configure_page()`)
- Menambah frasa waktu baru ("2 minggu lalu") → **`parser.py`**
- Menambah cara bertanya baru (mis. kueri anomali) → **`nlp_queries.py`**
- Mengubah gaya bahasa balasan chat standar → **`narration.py`**
- Menambah/mengubah grafik → **`charts.py`**
- Mengubah tampilan/garis batas grafik Live Stream, atau jumlah titik
  yang ditampilkan → **`charts.py`**'s `live_stream_chart()` (default
  jumlah titik ada di `config.py`'s `DEFAULT_LIVE_CHART_POINTS`)
- Mengubah kata kunci atau perilaku rule notifikasi → **`rules.py`**
- Mengubah cara data live disimulasikan/dikirim otomatis → **`iot_stream_sender.py`**
- Mengubah form input acak manual (rentang default, tampilan) → **`iot_stream_form.py`**
- Mengubah cara data live di-polling/diproses jadi window → **`live_feed.py`**
  (dan `Engine.append_raw()`/`Engine.append_window()` di `engine.py` kalau
  bentuk window/fitur ikut berubah)
- Mengubah alur chat / tab / sidebar → **`iot_app_v3.py`**
