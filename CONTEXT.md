# CONTEXT.md — Panduan Eksekusi AI Agent: Article Named-Entity Recognition (NER) Service

Dokumen ini merupakan spesifikasi konteks dan instruksi kerja menyeluruh bagi AI Agent untuk membangun, melatih, mengevaluasi, menguji, dan mendokumentasikan proyek **Article Named-Entity Recognition (NER) Service** sesuai dengan seluruh ketentuan pada modul studi kasus.

## 1. Gambaran Proyek dan Hasil Akhir yang Akan Dicapai

Tim pemantau berita (*news-monitoring team*) membutuhkan layanan web prototipe yang mampu mengidentifikasi entitas bernama (*named entities*) dari teks artikel berita. Pengguna memasukkan judul artikel (*title*) dan isi artikel (*body*) melalui panel antarmuka sederhana di *browser*, kemudian layanan menggabungkan kedua teks tersebut, menjalankan *pipeline* NER, dan menampilkan daftar entitas yang ditemukan beserta kategori dan posisi karakternya.

Melalui pengerjaan modul ini, AI Agent wajib menghasilkan capaian akhir (*deliverables*) berikut:

1. **Laporan Profiling Data Sampel**: Dokumentasi statistik pemuatan file CSV dari direktori `use_cases/pnm_study_case/sample/` yang mencakup jumlah baris, baris duplikat, nilai kosong (*missing titles/bodies*), distribusi bahasa, dan hasil inspeksi kolom `entities`.
2. **Dataset Beranotasi & Pembagian Data Reproducible**: Anotasi level *span* berformat mesin (JSONL, spaCy `DocBin`, atau BIO/IOB2) dengan *split* deterministik berbasis identitas artikel (**70% *training***, **15% *validation***, **15% *held-out test***) serta minimal **10 contoh evaluasi pendek** di `data/evaluation_examples.json` yang telah divalidasi.
3. **Pipeline Model NER Terlatih/Teradaptasi**: Perbandingan antara model *baseline* (*pretrained* atau *non-trained baseline*) dengan model akhir yang telah dilatih (*fine-tuned*), diadaptasi, atau dikalibrasi menggunakan data *training* dari sampel yang tersedia (solusi berbasis aturan/*rules-only* atau sekadar menjalankan model *pretrained* tanpa adaptasi tidak memenuhi syarat).
4. **Laporan Evaluasi & Error Analysis**: Metrik **Precision, Recall, dan F1-score** (keseluruhan secara *exact-span* maupun per label) yang dilaporkan terpisah untuk *validation set* dan *test set*, disertai analisis minimal **3 kesalahan (*false positives* atau *false negatives*)**.
5. **Layanan Backend API & Panel Antarmuka Web**: Backend dengan *endpoint* `POST /extract-entities` (menggunakan Python Flask atau FastAPI) yang terhubung ke artefak model hasil pelatihan, dilengkapi panel antarmuka *browser* yang memiliki indikator *loading*, validasi input, dan tabel hasil ekstraksi.
6. **Pengujian Otomatis (Automated Tests)**: Minimal **5 pengujian otomatis** yang dapat dijalankan ulang (*reproducible*) untuk memverifikasi perilaku API dan *pipeline* pada berbagai kondisi input normal maupun kasus tepi (*edge cases*).
7. **Dokumentasi Lengkap (`README.md`) & Jawaban 8 Pertanyaan Analisis**: Panduan instalasi, cara menjalankan aplikasi dan pengujian, diagram/deskripsi arsitektur, contoh *request/response*, keterbatasan sistem, serta jawaban mendalam atas 8 pertanyaan konseptual dan etika.

## 2. Aturan Arsitektur, Kontrak API, dan Struktur Proyek

### Struktur Direktori Wajib

Susun proyek di dalam repositori dengan struktur direktori berikut:

```text
project/
├── README.md
├── requirements.txt          # atau package.json sesuai stack yang disetujui
├── src/                      # kode sumber backend API, panel UI, preprocessing, training, dan evaluasi
├── tests/                    # minimal 5 automated tests (pytest / unittest)
├── data/
│   └── evaluation_examples.json
└── screenshots/              # opsional, disertakan jika diminta instruktur
```

**Larangan & Kepatuhan Repositori:**
- **Jangan meng-*commit* salinan kedua** dari file CSV artikel asli yang ada di `use_cases/pnm_study_case/sample/`. Baca langsung dari lokasi sumber atau gunakan *path* yang dapat dikonfigurasi.
- **Jangan meng-*commit* kredensial atau *secrets***. Seluruh konfigurasi sensitif wajib disuplai melalui *environment variables*.
- **Jangan melakukan *hard-code*** pada daftar entitas untuk contoh demonstrasi. Semua hasil ekstraksi harus murni berasal dari prediksi *pipeline* model terlatih/teradaptasi.
- Catat secara eksplisit versi seluruh *library* di `requirements.txt` dan nama/versi model yang digunakan.

### Kontrak Input dan Output API (`POST /extract-entities`)

| Aspek | Spesifikasi Teknis |
| :--- | :--- |
| **Endpoint & Method** | `POST /extract-entities` |
| **Content-Type** | `application/json` |
| **Format Request Body** | `{"title": "Example article title", "body": "The full article text goes here."}` |
| **Aturan Validasi Input** | `title` dan `body` wajib bertipe `string`. Tolak dengan HTTP status `400 Bad Request` dan pesan *error* JSON yang jelas apabila: (1) JSON tidak valid/*malformed*, (2) `title` atau `body` kosong/hanya berisi spasi, (3) tipe data bukan *string*, atau (4) panjang gabungan teks melebihi batas maksimum yang didokumentasikan (misalnya 50.000 karakter). |
| **Penggabungan Teks** | Gabungkan `title` dan `body` secara konsisten dan aman (misalnya `combined_text = f"{title.strip()}\n\n{body.strip()}"` atau pemisah terdokumentasi lainnya) sebelum dianalisis oleh model, sehingga perhitungan *offset* karakter `start` dan `end` mengacu secara presisi pada indeks karakter di `combined_text`. |
| **Determinisme** | Untuk pasangan input dan versi model yang sama, *output* yang dikembalikan wajib 100% deterministik. |

**Contoh Respons Valid (`HTTP 200 OK`):**

```json
{
  "entities": [
    {
      "text": "Ada Lovelace",
      "label": "PERSON",
      "start": 0,
      "end": 12
    },
    {
      "text": "London",
      "label": "GPE",
      "start": 28,
      "end": 34
    }
  ],
  "meta": {
    "entity_count": 2,
    "model": "documented-model-name"
  }
}
```

**Ketentuan Field Entitas:**
- `text`: potongan teks persis sesuai yang terdeteksi pada teks gabungan (`combined_text[start:end] == text`).
- `label`: kategori entitas yang didukung dan dipetakan dengan jelas (minimal `PERSON`, `ORG`, `GPE`/`LOC`; tambahkan `DATE`, `EVENT`, atau kategori relevan lainnya).
- `start`: indeks karakter awal berbasis nol (*zero-based character offset*) pada teks yang dianalisis.
- `end`: indeks karakter akhir eksklusif (*exclusive character offset*) pada teks yang dianalisis.

## 3. Tahapan Pengerjaan Langkah demi Langkah untuk AI Agent

### Tahap 1: Eksplorasi dan Profiling Data Sampel
1. Temukan dan muat file CSV di direktori `use_cases/pnm_study_case/sample/` menggunakan pemisah titik koma (`;`, `sep=";"`) dengan mempertahankan teks bertanda kutip (*quoted text*) secara utuh.
2. Periksa keberadaan dan tipe kolom utama: `title`, `body`, `language`, `entities`, `original_id`, `source_name`, dan `date_published`.
3. Buat skrip *profiling* yang menghitung dan mencatat metrik berikut untuk dimasukkan ke dalam laporan/`README.md`:
   - Total baris yang berhasil dimuat dari seluruh file CSV yang relevan.
   - Jumlah baris duplikat atau *near-duplicate* (berdasarkan `original_id` maupun kesamaan isi `title` + `body`).
   - Jumlah baris dengan `title` kosong/hilang dan `body` kosong/hilang.
   - Distribusi bahasa (`language`).
4. Inspeksi isi, format, dan asal-usul kolom `entities`:
   - Periksa apakah kolom tersebut kosong, tidak lengkap, atau tidak memiliki label rentang karakter (*span-labeled*).
   - Gunakan kolom `entities` hanya sebagai bahan eksplorasi awal atau *weak supervision* (bantuan pra-anotasi), **bukan** sebagai *ground truth* langsung tanpa verifikasi dan validasi.

### Tahap 2: Skema Anotasi, Pembuatan Set Evaluasi, dan Pembagian Data
1. **Tetapkan Skema Label**: Definisikan kategori entitas yang didukung oleh model dan ditampilkan ke pengguna (minimal mencakup orang/`PERSON`, organisasi/`ORG`, dan lokasi/entitas geopolitik/`LOC` atau `GPE`, ditambah `DATE` atau `EVENT` bila sesuai). Dokumentasikan aturan batas *span* (misalnya apakah gelar atau kata depan ikut dimasukkan).
2. **Buat Anotasi Level Span**:
   - Gunakan bantuan pra-anotasi (dari kolom `entities` yang dipetakan ke indeks karakter, kamus entitas, atau *pretrained model*) lalu lakukan kurasi/tinjauan agar label dan *character offset* (`start`, `end`) akurat.
   - Simpan anotasi dalam format yang dapat dibaca mesin (JSONL, spaCy `DocBin`, atau BIO/IOB2) dengan menyertakan `original_id` dan referensi indeks baris sumber.
   - Catat pedoman anotasi, kasus ambigu/*disagreements*, dan koreksi yang dilakukan terhadap label awal.
3. **Susun `data/evaluation_examples.json` (Minimal 10 Artikel/Kutipan)**:
   - Pilih minimal 10 artikel pendek atau kutipan artikel dari data sampel yang telah dikurasi secara cermat.
   - Pastikan set tersebut mencakup kombinasi: (a) entitas orang, organisasi, tempat, tanggal, dan kategori lain yang didukung, (b) entitas yang muncul lebih dari satu kali dalam satu teks, (c) nama ambigu dan nama yang mengandung tanda baca, (d) artikel yang sama sekali tidak mengandung *named entities*, serta (e) minimal satu contoh yang menantang/sulit bagi model yang dipilih.
4. **Bagi Dataset (*Data Splitting*)**:
   - Lakukan pembagian berdasarkan **identitas artikel (`original_id` / kelompok duplikat)**, bukan berdasarkan kalimat atau baris mentah, agar teks dari artikel yang sama tidak bocor (*data leakage*) ke *split* yang berbeda.
   - Tangani artikel duplikat atau *near-duplicate* dengan menghapusnya terlebih dahulu atau menempatkannya ke dalam satu *split* yang sama disertai penjelasan tertulis.
   - Gunakan *random seed* deterministik yang terdokumentasi (misalnya `SEED = 42`) dengan proporsi **70% *training***, **15% *validation***, dan **15% *held-out test***.
   - Kunci *held-out test set* agar tidak pernah digunakan selama pemilihan *hyperparameter* maupun penyesuaian aturan anotasi.

### Tahap 3: Pengembangan Model Baseline dan Pelatihan/Adaptasi Pipeline NER
1. **Bangun Baseline**:
   - Implementasikan *baseline* menggunakan model NER *pretrained* bawaan (misalnya spaCy pretrained model atau HuggingFace Transformers token classification model) tanpa adaptasi khusus pada data sampel, atau *baseline* sederhana yang terdokumentasi.
   - Evaluasi *baseline* pada *validation set* dan *test set* sebagai tolok ukur pembanding.
2. **Latih atau Adaptasi Model pada Data Training**:
   - Lakukan *fine-tuning* atau adaptasi *pipeline* NER menggunakan anotasi pada *training set* (untuk ukuran dataset kecil, *transfer learning*, *few-shot adaptation*, atau kalibrasi model dengan komponen terlatih diperbolehkan; tidak wajib melatih dari nol).
   - Gunakan *validation set* untuk menentukan *preprocessing*, pemetaan label (*label mappings*), *hyperparameters* (*learning rate*, *dropout*, *batch size*, jumlah *epoch*), serta *stopping criteria* (*early stopping*).
3. **Simpan Artefak & Reproduksibilitas**:
   - Simpan model terbaik hasil pelatihan ke dalam direktori model lokal atau sediakan satu perintah pelatihan (*training command*) yang *reproducible*.
   - Catat jumlah dokumen dan jumlah *span* entitas per *split* (*train*, *validation*, *test*), distribusi/ketimpangan kelas (*class imbalance*), konfigurasi model, *random seed*, dan durasi waktu pelatihan.

### Tahap 4: Evaluasi Komprehensif dan Analisis Kesalahan
1. Evaluasi model akhir yang telah dilatih pada *validation set* dan *held-out test set* **secara terpisah**.
2. Hitung dan laporkan metrik evaluasi berikut:
   - **Exact-span Precision, Recall, dan F1-score** secara keseluruhan (*overall micro/macro*).
   - **Metrik per kategori label** (*per-label Precision, Recall, F1-score*).
   - Nyatakan secara eksplisit skema pencocokan yang digunakan (*exact-span matching*, *partial-span matching*, atau *entity-type-and-span matching*).
   - Tabel perbandingan performa antara **Model Baseline vs Model Terlatih/Teradaptasi**.
   - Pernyataan eksplisit mengenai apakah contoh pada *test set* memengaruhi proses anotasi atau *tuning* (pastikan jawabannya tidak memengaruhi karena *test set* diisolasi sejak awal).
3. Lakukan **Analisis Kesalahan (*Error Analysis*)**:
   - Ambil minimal **3 contoh kesalahan nyata dari *test set*** (*false positives*, *false negatives*, atau kesalahan batas *span*/kategori label).
   - Jelaskan kutipan teksnya, label prediksi vs label *ground truth*, penyebab teknis terjadinya kesalahan tersebut, dan cara mengatasinya.

### Tahap 5: Implementasi Backend API dan Panel Antarmuka Web
1. **Backend Endpoint (`src/`)**:
   - Bangun aplikasi web menggunakan Flask atau FastAPI yang memuat artefak model NER hasil pelatihan/adaptasi dari Tahap 3.
   - Sediakan *endpoint* `POST /extract-entities` yang menerima JSON `{"title": "...", "body": "..."}`.
   - Terapkan validasi ketat: periksa struktur JSON, pastikan `title` dan `body` bertipe *string* dan tidak kosong setelah di-*trim*, serta periksa batas panjang maksimum karakter. Kembalikan status HTTP `400` dengan pesan JSON yang deskriptif bila validasi gagal.
   - Gabungkan `title` dan `body`, jalankan inferensi model secara deterministik, dan kembalikan daftar `entities` (`text`, `label`, `start`, `end`) beserta `meta` (`entity_count`, `model`).
2. **Panel Antarmuka Pengguna (Frontend UI)**:
   - Sediakan halaman web yang dapat diakses melalui *browser* (misalnya pada `GET /`) berisi kolom input untuk `title`, area teks untuk `body`, dan tombol *submit*.
   - Tambahkan validasi di sisi panel (*client-side validation*) sebelum mengirim permintaan ke *backend*.
   - Tampilkan indikator status *loading* atau *processing* yang terlihat jelas saat menunggu respons API.
   - Tampilkan hasil entitas yang ditemukan dalam format yang mudah dibaca (minimal berupa tabel yang memuat teks entitas, tipe/label entitas, dan posisi *offset*).
   - Pastikan panel menangani respons *error* dari *backend* dengan anggun (menampilkan pesan *error* yang ramah pengguna tanpa membuat antarmuka *crash*).

### Tahap 6: Pembuatan Pengujian Otomatis (Automated Tests)
Buat *test suite* otomatis di dalam folder `tests/` (misalnya menggunakan `pytest`) yang berisi **minimal 5 pengujian otomatis** dan mencakup seluruh skenario uji berikut:
1. **Input normal**: Artikel standar dengan `title` dan `body` yang menghasilkan daftar entitas valid beserta struktur JSON dan *offset* yang presisi (`combined_text[start:end] == text`).
2. **Input kosong & JSON tidak valid**: Permintaan dengan `title` kosong, `body` kosong, tipe non-*string*, atau *payload* JSON rusak harus ditolak dengan status HTTP `400`.
3. **Tanda baca & nama ambigu**: Artikel yang memuat entitas dengan tanda baca (seperti singkatan, tanda hubung, atau apostrof) serta nama ambigu.
4. **Entitas berulang (*repeated entities*)**: Artikel di mana entitas yang sama disebut beberapa kali pada lokasi berbeda harus mendeteksi kemunculan masing-masing dengan *offset* `start` dan `end` yang berbeda dan tepat.
5. **Angka, tanggal, dan teks non-ASCII**: Artikel yang memuat format angka, tanggal, serta karakter non-ASCII/internasional (misalnya aksen atau simbol mata uang/karakter Unicode) tanpa menyebabkan kesalahan perhitungan *character offset*.
6. **Teks sangat panjang (*long text*)**: Pengujian untuk teks panjang yang masih dalam batas valid maupun teks yang melampaui batas maksimum yang didokumentasikan (harus ditolak dengan HTTP `400`).

### Tahap 7: Penyusunan Dokumentasi (`README.md`) dan Jawaban 8 Pertanyaan Analisis
Susun `README.md` yang bersih, terstruktur, dan memuat seluruh bagian wajib berikut:
1. **Deskripsi Singkat & Diagram Arsitektur**: Gambaran alur data dari panel UI -> validasi API -> penggabungan `title` & `body` -> *pipeline* NER terlatih -> respons JSON.
2. **Instruksi Setup, Pelatihan, dan Menjalankan Layanan**: Perintah lengkap mulai dari pembuatan *virtual environment*, instalasi dependensi (`pip install -r requirements.txt`), perintah *reproducible* untuk menjalankan *data profiling/splitting/training*, cara menyalakan server lokal, hingga cara menjalankan `pytest`.
3. **Penggunaan API**: Contoh konkret *request* (`curl` / JSON) dan *response* sukses maupun *error* (`HTTP 400`), termasuk penjelasan batas panjang input dan cara penggabungan `title` serta `body`.
4. **Laporan Data, Model, & Hasil Evaluasi**:
   - Statistik hasil *profiling* CSV di `use_cases/pnm_study_case/sample/`.
   - Alasan pemilihan model, daftar versi dependensi/model, skema anotasi, dan rincian *data split* (70/15/15 beserta *seed* dan jumlah dokumen/*span*).
   - Tabel metrik evaluasi terpisah untuk *validation* dan *held-out test* (*overall* dan *per-label* Precision, Recall, F1), perbandingan *baseline* vs *trained model*, serta pembahasan minimal 3 *error analysis* dari *test set*.
5. **Jawaban Lengkap atas 8 Pertanyaan Studi Kasus**:
   - **Pertanyaan 1 (*What makes a text span a named entity in this problem?*)**: Jelaskan definisi *named entity* dalam konteks pemantauan berita (referensi dunia nyata yang spesifik seperti tokoh, organisasi/institusi, lokasi geografis/geopolitik, tanggal/waktu, atau peristiwa bernama) beserta aturan batas *span* yang membedakannya dari kata benda umum (*common nouns*).
   - **Pertanyaan 2 (*Why should title and body be combined before analysis? What information could be lost if they are processed independently?*)**: Jelaskan bagaimana judul sering memuat entitas utama atau konteks topik yang dirujuk kembali di isi berita (koreferensi/disambiguasi konteks), serta bagaimana penggabungan menyederhanakan satu kali inferensi terpadu dan menjaga konsistensi konteks model.
   - **Pertanyaan 3 (*What are the supported labels of the chosen model, and how do they map to the categories shown to users?*)**: Sajikan tabel pemetaan dari label internal model ke label *output* layanan beserta contohnya.
   - **Pertanyaan 4 (*How does tokenization, punctuation, capitalization, or language affect the result?*)**: Bahas dampak subword/word tokenization terhadap batas *span*, pengaruh tanda baca yang menempel pada kata, ketergantungan model terhadap huruf kapital (*casing* pada judul huruf besar semua vs huruf kecil), serta perbedaan performa lintas bahasa pada dataset multibahasa.
   - **Pertanyaan 5 (*What is the difference between an entity's text, label, and character offsets?*)**: Jelaskan perbedaan antara *surface string* (`text`), klasifikasi semantik (`label`), dan koordinat posisi indeks karakter (`start`, `end`) yang membedakan beberapa kemunculan teks yang sama di lokasi berbeda dalam satu artikel.
   - **Pertanyaan 6 (*What happens when the same entity appears multiple times with different spelling or capitalization?*)**: Jelaskan bahwa NER standar mengekstrak setiap kemunculan sebagai *span* terpisah apa adanya sesuai teks permukaan (*surface form*) tanpa menyatukannya secara otomatis kecuali ditambahkan tahap *entity linking* atau normalisasi kanonik.
   - **Pertanyaan 7 (*Which errors are most important for a news-monitoring user, and how could the system be improved?*)**: Analisis dampak *false negatives* (berita penting terlewat dari pantauan klien) vs *false positives* (kebisingan/*alert fatigue*), lalu usulkan perbaikan konkret (seperti penambahan data anotasi domain spesifik, normalisasi/alias entitas, dukungan model multibahasa, dan *active learning*).
   - **Pertanyaan 8 (*What privacy, copyright, bias, or responsible-use concerns apply to processing article text?*)**: Bahas aspek PII/privasi individu dalam berita, hak cipta penerbit media atas penyimpanan/redistribusi teks artikel, bias representasi model terhadap nama non-Barat/lokal, serta tanggung jawab penggunaan otomatis agar tidak menghasilkan atribusi palsu yang merugikan.
6. **Keterbatasan Sistem (*Known Limitations*)**: Dokumentasikan batasan panjang teks, bahasa yang didukung, sensitivitas terhadap kesalahan ketik/kapitalisasi, dan batasan cakupan domain.

## 4. Tantangan Tambahan Opsional (Extension Challenges)

Jika waktu dan sumber daya memungkinkan setelah seluruh kebutuhan utama selesai dan teruji, AI Agent dapat mengimplementasikan satu atau lebih fitur nilai tambah berikut:
- Menyorot (*highlight*) rentang entitas secara visual langsung di dalam tampilan teks artikel pada panel UI.
- Mendukung beberapa bahasa sekaligus dan membandingkan performa model antarbahasa.
- Melakukan normalisasi entitas yang ekuivalen (misalnya menyatukan variasi nama/alias) sambil tetap mempertahankan teks asli (`text`).
- Menambahkan skor kepercayaan (*confidence scores*) pada setiap entitas beserta penjelasan maknanya.
- Membungkus layanan ke dalam kontainer (`Dockerfile` dan `docker-compose.yml`).
- Membandingkan dua arsitektur/model NER yang berbeda pada set evaluasi yang sama.
- Menambahkan *endpoint batch* (misalnya `POST /extract-entities/batch`) untuk memproses banyak artikel sekaligus.

## 5. Checklist Verifikasi Akhir (Definition of Done)

Sebelum menyatakan pekerjaan selesai, AI Agent wajib memverifikasi bahwa seluruh poin di bawah ini telah terpenuhi:
- [ ] Aplikasi dapat dijalankan secara lokal menggunakan perintah yang tertulis di `README.md` tanpa perlu mengubah kode secara manual.
- [ ] Pengguna dapat membuka panel di *browser*, mengisi `title` dan `body`, menekan tombol *submit*, melihat indikator *loading*, dan melihat daftar entitas hasil ekstraksi.
- [ ] Model yang digunakan oleh *endpoint* API adalah model yang telah dilatih atau diadaptasi menggunakan data sampel dari `use_cases/pnm_study_case/sample/`, bukan sekadar *hard-coded* atau *rules-only*.
- [ ] File CSV asli tidak diduplikasi ke dalam *commit* repositori, dan tidak ada *secret* yang di-*commit*.
- [ ] `data/evaluation_examples.json` tersedia dan memuat minimal 10 contoh evaluasi yang tervalidasi.
- [ ] Seluruh pengujian otomatis (minimal 5 pengujian) di folder `tests/` berjalan sukses (`PASS`).
- [ ] `README.md` memuat statistik data, rincian *split* (70/15/15), hasil evaluasi terpisah (*validation* vs *test*), perbandingan *baseline* vs *trained*, minimal 3 *error analysis*, contoh *request/response* API, dan jawaban atas 8 pertanyaan studi kasus.
