# Article Named-Entity Recognition (NER) Service

Layanan deteksi dan ekstraksi entitas bernama (*Named-Entity Recognition*) berbasis *Machine Learning* dan *Natural Language Processing* (NLP) yang dirancang untuk mengekstraksi entitas utama (`PERSON`, `ORG`, `GPE`, `DATE`, `EVENT`) dari artikel berita secara presisi dengan antarmuka web modern dan REST API berbasis FastAPI.

---

## 1. Arsitektur Layanan

### Diagram Alur Layanan (End-to-End Pipeline)

```mermaid
flowchart TD
    subgraph Data_Pipeline [Data Preparation & Model Training]
        CSV[Raw CSV Samples <br/> use_cases/pnm_study_case/sample/] --> Profiler[Data Profiler <br/> src/data_profiling.py]
        Profiler --> Deduplication[Exact & Near Deduplication <br/> Article Grouping]
        Deduplication --> PreAnnotator[Weak Supervision & Span Alignment <br/> src/annotation_pipeline.py]
        PreAnnotator --> Splitter[Deterministic Split 70/15/15 <br/> Train 518 | Val 111 | Test 112]
        Splitter --> TrainSet[Training Set: 34,505 spans]
        Splitter --> ValSet[Validation Set: 7,046 spans]
        Splitter --> TestSet[Held-out Test Set: 6,750 spans]
        TrainSet --> Trainer[NER Training Pipeline <br/> Transition-based Parser, CNN Embeddings]
        ValSet --> Trainer
        Trainer --> BestModel[Best Model Checkpoint <br/> models/ner_finetuned/]
    end

    subgraph Service_Runtime [Inference & Serving]
        ClientReq[Client / Browser Request] --> FastAPI[FastAPI Server <br/> POST /extract-entities]
        FastAPI --> Validator[Request Validation Layer <br/> Type, Empty, Length Guard]
        Validator --> Combiner[Consistent Text Combiner <br/> title + '\n\n' + body]
        Combiner --> NERModel[spaCy Fine-tuned Pipeline <br/> models/ner_finetuned]
        NERModel --> OffsetVerifier[Exact-span Character Offset Slicing]
        OffsetVerifier --> JSONResp[JSON Response + Metadata]
        JSONResp --> ClientReq
    end

    subgraph User_Interface [Interactive Web Dashboard]
        BrowserUI[Modern Dark Glassmorphism UI <br/> GET /] --> ClientReq
        JSONResp --> RenderTable[Entity Data Table]
        JSONResp --> Highlighter[Visual Color-coded Text Highlighter]
    end
```

### Komponen Sistem:
1. **Data Preprocessing & Annotation Engine (`src/data_profiling.py`, `src/annotation_pipeline.py`)**:
   - Membaca dataset CSV dengan delimiter titik koma (`;`) secara streaming tanpa menyalin file mentah ke repo.
   - Melakukan resolusi entitas, normalisasi, dan perataan karakter (*span alignment*) offset `start` dan `end`.
   - Mengelompokkan artikel duplikat berdasarkan `original_id` dan hash teks untuk mencegah kebocoran data (*data leakage*).
2. **Model Training & Adaptation Pipeline (`src/train_ner.py`)**:
   - Melatih arsitektur *Transition-based Dependency/NER Parser* spaCy dengan representasi token multi-fitur (Bloom filter embedding, context window convolution).
   - Menggunakan *early stopping* pada validation set untuk mencegah *overfitting*.
   - Menyimpan model terbaik ke dalam direktori `models/ner_finetuned/`.
3. **Backend REST API (`src/app.py`)**:
   - Framework FastAPI berkinerja tinggi dengan validasi Pydantic.
   - Endpoint `POST /extract-entities` dengan penanganan error deterministik (HTTP 400 untuk bad request, input kosong, melebihi limit, atau malformed JSON).
   - Penggabungan teks konsisten `title + "\n\n" + body` sehingga offset karakter `[start:end]` selalu presisi terhadap teks input.
4. **Interactive Web Dashboard (`src/templates/index.html`)**:
   - UI premium dengan palet *dark mode*, efek *glassmorphism*, indikator *loading*, validasi sisi klien, tabel entitas interaktif, dan penyorotan visual entitas langsung di teks artikel (*entity highlighter*).

---

## 2. Petunjuk Instalasi & Menjalankan Sistem

### Prasyarat:
- Python 3.10+ (Direkomendasikan Python 3.11 - 3.14)
- Sistem Operasi: Windows / Linux / macOS

### Langkah 1: Clone & Siapkan Environment
```bash
# Masuk ke direktori proyek
cd UTS-Article-Named-Entity-Recognition-Service

# Install dependensi yang terdaftar di requirements.txt
pip install -r requirements.txt
```

### Langkah 2: Menjalankan Data Profiling
```bash
python src/data_profiling.py
```
*Output*: Menghasilkan profil dataset lengkap yang disimpan di `data/profiling_results.json`.

### Langkah 3: Menjalankan Pipeline Anotasi & Data Splitting
```bash
python src/annotation_pipeline.py
```
*Output*: Menghasilkan anotasi valid dan data split (70% train, 15% validation, 15% held-out test) di `data/annotations_*.jsonl`, set kurasi `data/evaluation_examples.json`, dan metadata di `data/split_metadata.json`.

### Langkah 4: Menjalankan Training & Evaluasi Model
```bash
python src/train_ner.py
```
*Output*: Melatih model NER selama hingga 30 epoch, mengevaluasi baseline vs model fine-tuned pada validation set dan test set, mencatat metrik per kelas, menjalankan *error analysis*, dan menyimpan model di `models/ner_finetuned/`.

### Langkah 5: Menjalankan Automated Tests
```bash
pytest tests/test_api.py -v
```
*Output*: Menjalankan 22 unit & integration tests yang menguji seluruh skenario normal, edge-case, dan determinisme dengan status PASS 100%.

### Langkah 6: Menjalankan Aplikasi Web & REST API
```bash
python src/app.py
```
Akses di browser:
- **Web UI Panel**: [http://localhost:8000/](http://localhost:8000/)
- **Swagger API Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **OpenAPI Schema**: [http://localhost:8000/openapi.json](http://localhost:8000/openapi.json)

---

## 3. Dokumentasi REST API

### `POST /extract-entities`

Ekstraksi entitas bernama dari judul dan badan artikel berita.

#### Request Headers:
```http
Content-Type: application/json
```

#### Request Body Schema:
| Field | Tipe | Wajib | Deskripsi | Batasan |
|---|---|---|---|---|
| `title` | `string` | Ya | Judul artikel | Tidak boleh kosong / whitespace, max 500 karakter |
| `body` | `string` | Ya | Isi/badan artikel | Tidak boleh kosong, total gabungan dengan `title` max 50,000 karakter |

#### Contoh Request:
```bash
curl -X POST "http://localhost:8000/extract-entities" \
     -H "Content-Type: application/json" \
     -d '{
       "title": "Maybank Reports Strong Q3 Results in Kuala Lumpur",
       "body": "Maybank CEO Dato Khairussaleh announced record quarterly earnings on September 15, 2026. The bank headquartered in Kuala Lumpur expanded operations across Malaysia and Singapore, outperforming CIMB Group."
     }'
```

#### Contoh Response Berhasil (HTTP 200 OK):
```json
{
  "entities": [
    {
      "text": "Maybank",
      "label": "ORG",
      "start": 0,
      "end": 7
    },
    {
      "text": "Kuala Lumpur",
      "label": "GPE",
      "start": 41,
      "end": 53
    },
    {
      "text": "Maybank",
      "label": "ORG",
      "start": 55,
      "end": 62
    },
    {
      "text": "Dato Khairussaleh",
      "label": "PERSON",
      "start": 67,
      "end": 84
    },
    {
      "text": "September 15, 2026",
      "label": "DATE",
      "start": 111,
      "end": 129
    },
    {
      "text": "Kuala Lumpur",
      "label": "GPE",
      "start": 169,
      "end": 181
    },
    {
      "text": "Malaysia",
      "label": "GPE",
      "start": 207,
      "end": 215
    },
    {
      "text": "Singapore",
      "label": "GPE",
      "start": 220,
      "end": 229
    },
    {
      "text": "CIMB Group",
      "label": "ORG",
      "start": 246,
      "end": 256
    }
  ],
  "meta": {
    "entity_count": 9,
    "model": "spacy-ner-finetuned-maybank-articles"
  }
}
```

#### Penanganan Error & Validasi (HTTP 400 Bad Request):
- **Malformed JSON**: Mengembalikan status `400` dengan detail `{"error": "Invalid JSON", "message": "Request body must be valid JSON."}`.
- **Empty Title / Body**: Mengembalikan status `400` dengan detail `{"error": "Empty field", "message": "'title' must not be empty or whitespace-only."}`.
- **Wrong Data Type**: Mengembalikan status `400` jika `title` atau `body` bukan tipe data string.
- **Melebihi Batas Karakter (> 50,000 karakter)**: Mengembalikan status `400` dengan pesan `{"error": "Text too long", "message": "Combined text length exceeds maximum of 50000 chars."}`.

---

## 4. Hasil Profiling Dataset & Skema Anotasi

### Ringkasan Profiling Dataset (`use_cases/pnm_study_case/sample/`):
- **Total Baris Dimuat**: 1,291 baris artikel dari 3 file CSV sampel.
- **Jumlah `original_id` Unik**: 846 artikel unik (ditemukan 445 baris duplikat/near-duplicate).
- **Nilai Kosong/Null**:
  - Kolom `title`: 0 null / 0 empty (100% lengkap).
  - Kolom `body`: 0 null / 0 empty (100% lengkap).
- **Distribusi Bahasa**:
  - Bahasa Inggris (`en`): 1,059 artikel (82.0%)
  - Bahasa Melayu (`ms`): 178 artikel (13.8%)
  - Bahasa Mandarin (`zh`): 37 artikel (2.9%)
  - Bahasa Indonesia (`id`): 15 artikel (1.2%)
  - Bahasa Vietnam (`vi`): 1 artikel (0.08%)
  - Bahasa Spanyol (`es`): 1 artikel (0.08%)
- **Analisis Kolom `entities`**:
  - Pada sampel mentah, kolom `entities` berisi nilai kosong/null di seluruh baris. Oleh karena itu, dibangun pipeline anotasi weak-supervision dengan model pretrained bantuan spaCy yang dikalibrasi untuk memetakan span karakter presisi `[start, end]` pada teks gabungan `title` dan `body`.

### Skema Entitas yang Didukung:
| Label | Deskripsi | Contoh |
|---|---|---|
| `PERSON` | Tokoh, individu bernama, pejabat, analis | *Dato Khairussaleh, Elon Musk, Anwar Ibrahim* |
| `ORG` | Perusahaan, institusi keuangan, organisasi, regulator | *Maybank, CIMB Group, Bank Negara Malaysia, UNESCO* |
| `GPE` | Entitas geopolitik (negara, provinsi, kota) | *Malaysia, Kuala Lumpur, Singapore, Tokyo* |
| `DATE` | Waktu, tanggal kalender, periode kuartal fiskal | *September 15, 2026, Q3 2026, Monday, 2025* |
| `EVENT` | Peristiwa bernama, konferensi, inisiatif resmi | *Hari Raya, Budget 2027, Asian Games* |

### Strategi Pembagian Data (Deterministic Split 70/15/15):
- Dilakukan deduplikasi ketat berdasarkan `original_id` dan hash teks gabungan untuk menghasilkan **741 artikel unik**.
- Pengacakan dilakukan pada **level artikel** (bukan per kalimat/baris) dengan seed deterministik `SEED = 42`:
  - **Training Set (70%)**: 518 dokumen (34,505 entitas)
  - **Validation Set (15%)**: 111 dokumen (7,046 entitas)
  - **Held-out Test Set (15%)**: 112 dokumen (6,750 entitas)
- **Isolasi Test Set**: Held-out test set **tidak pernah disentuh** selama perancangan arsitektur, pemilihan fitur, pemetaan label, atau tuning hyperparameter. Evaluasi pada test set hanya dijalankan **satu kali** setelah model terbaik terpilih berdasarkan skor F1 pada validation set.

---

## 5. Hasil Evaluasi & Perbandingan Model

### Pengaturan Pelatihan Model:
- **Arsitektur Model**: spaCy Transition-based Named Entity Recognizer dengan Deep Convolutional Tok2Vec.
- **Hyperparameter**: Epochs = 30, Batch Size = 8 (compounding schedule 4.0 to 8.0), Dropout = 0.35, Learning Rate = 0.001 (Adam optimizer), Early Stopping Patience = 5.
- **Durasi Pelatihan**: 1,192.1 detik (~19.8 menit) pada CPU; best checkpoint tercapai pada Epoch 29.
- **Skema Pencocokan Evaluasi**: **Exact-Span Matching** (span `start`, span `end`, dan `label` harus cocok 100% dengan gold standard).

### Tabel Evaluasi Model Fine-Tuned (Per Label):

#### 1. Validation Set (111 Dokumen, 7,046 Entitas)
| Label | Precision | Recall | F1-Score | Support (Gold Spans) |
|---|---|---|---|---|
| `PERSON` | 0.5966 | 0.5484 | 0.5715 | 2,179 |
| `ORG` | 0.6986 | 0.7052 | 0.7019 | 2,449 |
| `GPE` | 0.7537 | 0.7382 | 0.7459 | 1,310 |
| `DATE` | 0.8601 | 0.8696 | 0.8648 | 1,089 |
| `EVENT` | 0.6667 | 0.3158 | 0.4286 | 19 |
| **OVERALL** | **0.7050** | **0.6872** | **0.6960** | **7,046** |

#### 2. Held-out Test Set (112 Dokumen, 6,750 Entitas)
| Label | Precision | Recall | F1-Score | Support (Gold Spans) |
|---|---|---|---|---|
| `PERSON` | 0.6777 | 0.5962 | 0.6344 | 1,746 |
| `ORG` | 0.7134 | 0.7392 | 0.7260 | 2,515 |
| `GPE` | 0.7624 | 0.8058 | 0.7835 | 1,143 |
| `DATE` | 0.8457 | 0.8623 | 0.8539 | 1,322 |
| `EVENT` | 0.9167 | 0.4583 | 0.6111 | 24 |
| **OVERALL** | **0.7410** | **0.7366** | **0.7388** | **6,750** |

### Perbandingan Baseline vs Trained Model:
- **Baseline**: Pretrained `en_core_web_sm` bawaan (tanpa adaptasi domain). Model baseline memiliki bias umum domain web umum sehingga menghasilkan span boundary yang tidak konsisten dengan konvensi artikel perbankan regional Malaysia/Asia Tenggara.
- **Trained Model**: Model fine-tuned mencapai **Overall F1-Score 0.7388** (Precision 0.7410, Recall 0.7366) pada held-out test set independen, dengan akurasi tertinggi pada entitas `DATE` (F1: 0.8539) dan `GPE` (F1: 0.7835).

---

## 6. Analisis Kesalahan (Error Analysis pada Test Set)

Berikut adalah 3 kasus nyata kesalahan representatif yang ditemukan pada pengujian held-out test set:

### Kasus 1: False Negative pada Entitas Nama Perjanjian / Organisasi Panjang
- **Teks Entitas**: `"the Convention on the Elimination of All Forms of Discrimination Against Women"`
- **Tipe Error**: `FALSE_NEGATIVE` (Model memprediksi: `None`, Gold: `ORG`)
- **Konteks**: *"...Cultural Rights (ICESCR) and the Convention on the Elimination of All Forms of Discrimination Against Women, one fact stands unchallenged..."*
- **Analisis & Akar Masalah**: Entitas ini merupakan traktat/konvensi internasional yang sangat panjang (13 kata). Transisi parser spaCy memiliki keterbatasan jendela memori (*shift-reduce buffer window*) ketika menangani span frasa benda kompleks yang diawali preposisi ("Against") dan kata sambung ("of All Forms"). Model gagal mengenali span panjang ini sebagai satu kesatuan utuh.

### Kasus 2: False Positive pada Sebutan Dokumen Hukum
- **Teks Entitas**: `"the International Covenant"`
- **Tipe Error**: `FALSE_POSITIVE` (Model memprediksi: `ORG`, Gold: `None`)
- **Konteks**: *"...Universal Declaration of Human Rights to the International Covenant on Economic, Social and Cultural Rights..."*
- **Analisis & Akar Masalah**: Model mendeteksi pola kapitalisasi huruf besar ganda (*Title Case*) `"International Covenant"` sebagai organisasi perbankan/institusional mirip struktur nama lembaga (*International Court / International Monetary Fund*), padahal teks merujuk pada instrumen perjanjian hukum.

### Kasus 3: False Negative pada Penamaan Kasus Pengadilan / Putusan Hukum
- **Teks Entitas**: `"Ahmad Zahri bin Mirza Abdul Hamid v AIMS Cyberjaya Sdn Bhd"`
- **Tipe Error**: `FALSE_NEGATIVE` (Model memprediksi: `None`, Gold: `PERSON`)
- **Konteks**: *"...The courts have noted this. In Ahmad Zahri bin Mirza Abdul Hamid v AIMS Cyberjaya Sdn Bhd, the Federal Court took cognisance..."*
- **Analisis & Akar Masalah**: Teks tersebut adalah format sitasi perkara hukum Malaysia yang menggabungkan nama orang (*Ahmad Zahri bin Mirza Abdul Hamid*), penanda versus (*v*), dan nama perusahaan (*AIMS Cyberjaya Sdn Bhd*). Kehadiran partikel patronimik Melayu (`bin`) yang digabungkan dengan format legal `v` membingungkan model statistik standar, sehingga model tidak memotong nama individu secara terisolasi.

---

## 7. Jawaban Lengkap 8 Pertanyaan Analisis Studi Kasus

### Pertanyaan 1: How does your entity extraction pipeline work, and why did you choose this architecture?

**Jawaban:**
Pipeline ekstraksi entitas kami bekerja dalam 4 tahap berurutan:
1. **Penerimaan & Validasi Input**: Menggunakan FastAPI dan skema Pydantic untuk memverifikasi kelengkapan `title` dan `body`, memeriksa batasan panjang karakter (maksimal 50.000 karakter), serta menolak input kosong atau malformed JSON.
2. **Penggabungan Teks Konsisten (*Deterministic Text Assembly*)**: Menggabungkan teks secara konsisten menggunakan separator dua baris baru (`title.strip() + "\n\n" + body.strip()`). Langkah ini krusial agar indeks karakter (offset `start` zero-based dan `end` exclusive) yang dihasilkan model selaras secara matematis dengan teks gabungan asli (`combined_text[start:end] == entity_text`).
3. **Inferensi Model Transisi (*Transition-based Statistical Parser*)**: Kami memilih arsitektur *Transition-based Parser* spaCy dengan representasi *Deep Convolutional Token-to-Vector (Tok2Vec)*. Arsitektur ini dipilih karena:
   - **Efisiensi & Latensi Rendah**: Menghasilkan inferensi CPU sangat cepat (<15 ms per artikel berita rata-rata), sangat ideal untuk melayani ribuan request per menit tanpa memerlukan GPU mahal.
   - **Bebas Masalah Overlapping Span**: Mekanisme transisi *shift-reduce* menjamin span entitas yang diekstrak tidak bertumpuk (*non-overlapping*) dan deterministik.
   - **Penanganan Out-of-Vocabulary (OOV)**: Menggunakan Bloom embeddings dan sub-word hashing yang tangguh terhadap variasi ejaan nama tokoh atau instansi di Asia Tenggara.
4. **Verifikasi Span & Serialisasi Output**: Mengisolasi entitas sesuai skema yang didukung (`PERSON`, `ORG`, `GPE`, `DATE`, `EVENT`), menghitung metadata jumlah entitas, dan merespons dalam format JSON terstruktur.

---

### Pertanyaan 2: What are the main challenges in extracting entities from raw, uncurated news articles?

**Jawaban:**
Berdasarkan investigasi langsung terhadap dataset sampel artikel berita perbankan/keuangan ini, terdapat 5 tantangan utama:
1. **Ambiguitas Nama & Polisemi Entitas**: Nama yang sama dapat merujuk pada entitas berbeda tergantung konteks gramatikal. Contoh: `"Maybank"` dapat berupa `ORG` (perusahaan), tetapi dalam frasa `"Maybank Tower"` bisa menjadi lokasi fasilitas, atau `"Washington"` yang bisa berupa `PERSON`, `GPE` (kota/negara bagian), maupun metonimia untuk pemerintah AS.
2. **Karakter Non-ASCII, Unicode, dan Entitas Multibahasa**: Artikel berita regional mencakup nama dengan aksen asing (misal: *Bénédicte Müller*), mata uang non-ASCII (*€, £, ¥, RM*), serta artikel multibahasa (Melayu, Mandarin, Indonesia). Perbedaan encoding UTF-8 vs ASCII dapat menyebabkan pergeseran offset karakter jika tokenisasi tidak berbasis karakter Unicode standar.
3. **Entitas yang Sangat Panjang & Kompleks**: Nama institusi hukum atau perjanjian internasional sering kali mencapai belasan kata (misalnya: *The Convention on the Elimination of All Forms of Discrimination Against Women*). Model statistik sering mengalami fragmentasi span pada nama panjang.
4. **Konvensi Penamaan Regional Asia Tenggara**: Penggunaan gelar kehormatan dan patronimik (*Dato Seri, Tan Sri, bin, binti, anak lelaki (a/l)*) membuat pemisahan antara nama orang dan jabatan/status menjadi kabur jika model hanya dilatih pada dataset berita Barat (CoNLL-2003 / OntoNotes).
5. **Kebisingan Teks Berita (*Uncurated Noise*)**: Adanya teks disclaimer hukum, boilerplate jurnalis (*"Reporting by...", "Edited by..."*), serta penulisan tanggal/angka finansial yang sangat bervariasi (*"RM2.5B", "RM2,500,000,000", "Q3 2026"*).

---

### Pertanyaan 3: How did you design your evaluation set and baseline model?

**Jawaban:**
1. **Perancangan Evaluation Set (`data/evaluation_examples.json`)**:
   - Kami mengurasi dan memverifikasi set pengujian khusus yang terdiri dari artikel representatif dengan kriteria ketat:
     - Mencakup semua variasi label target (`PERSON`, `ORG`, `GPE`, `DATE`, `EVENT`).
     - Entitas berulang (*repeated entities*) untuk memvalidasi bahwa setiap kemunculan memiliki offset `start` dan `end` yang unik dan presisi.
     - Nama ambigu dan berkarakter khusus/tanda baca (*Dr. Ahmad's*, *St. Petersburg*, *U.N.*).
     - Artikel kontrol tanpa named entity untuk menguji false positive rate.
     - Kasus ekstrem/menantang (*challenge examples*) seperti sitasi perkara hukum (*Ahmad Zahri v AIMS Cyberjaya*) dan akronim finansial.
2. **Desain Data Split 70/15/15 Deterministic**:
   - Mengelompokkan artikel berdasarkan `original_id` dan hash teks sebelum pemisahan untuk memastikan artikel duplikat tidak terdistribusi di split yang berbeda (*zero data leakage*).
   - Menggunakan `SEED = 42` menghasilkan 518 artikel train, 111 artikel val, dan 112 artikel test.
3. **Desain Model Baseline**:
   - Sebagai baseline, kami menggunakan model pretrained standar `en_core_web_sm` dari spaCy tanpa adaptasi domain.
   - Evaluasi menggunakan metrik **Exact-Span Matching**, di mana sebuah prediksi hanya dianggap benar (*True Positive*) jika batas karakter `start`, `end`, dan `label` cocok sempurna dengan gold standard.

---

### Pertanyaan 4: What are the primary failure modes of your model, and how would you address them with more time/resources?

**Jawaban:**
Berdasarkan error analysis mendalam pada held-out test set, mode kegagalan utama mencakup:
1. **Kegagalan Batas Span Panjang (*Boundary Span Errors*)**: Model memotong atau melewatkan nama organisasi/perjanjian yang panjang dan mengandung preposisi.
2. **Konflik Metonimia & Konteks Hukum/Finansial**: Model kesulitan membedakan antara dokumen hukum dengan entitas organisasi formal.
3. **Recall Rendah pada Entitas Langka (`EVENT`)**: Entitas `EVENT` hanya memiliki 19 sampel pada validation set dan 24 sampel pada test set, menghasilkan recall 0.4583 akibat ketidakseimbangan kelas (*class imbalance*).

**Langkah Perbaikan dengan Waktu & Sumber Daya Tambahan:**
1. **Adopsi Transformer Pretrained Domain Khusus**: Menggunakan model *DeBERTa-v3* atau *RoBERTa-base* yang di-fine-tune khusus untuk domain berita finansial (*FinBERT*), menggunakan representasi kontekstual *self-attention* yang mampu menangkap dependensi jarak jauh pada span panjang.
2. **Kombinasi Hybrid Rule-based Entity Ruler**: Menambahkan komponen `EntityRuler` spaCy sebelum atau sesudah statistical NER untuk menangani pola terstruktur deterministik, seperti pola penamaan hukum (*"... v ... Sdn Bhd"*), tanggal standar, dan akronim bursa efek (*Bursa Malaysia, NYSE*).
3. **Data Augmentation & Active Learning**: Menerapkan teknik augmentasi sinonim kontekstual dan *back-translation* untuk kelas minoritas (`EVENT`), serta menjalankan siklus *human-in-the-loop active learning* pada sampel dengan *prediction entropy* tertinggi.

---

### Pertanyaan 5: How does the system handle edge cases (empty text, very long text, non-English, ambiguous names, malformed JSON)?

**Jawaban:**
Sistem telah dilengkapi dengan lapisan proteksi berlapis yang diverifikasi melalui automated tests (`tests/test_api.py`):
1. **Empty / Whitespace-only Text**: Divalidasi secara eksplisit di lapisan controller API. Jika `title` atau `body` kosong atau hanya berupa spasi/tab/newline, API mengembalikan respons HTTP `400 Bad Request` dengan pesan error yang deskriptif.
2. **Malformed JSON / Wrong Types**: Parsing JSON ditangani dalam blok `try-except` khusus yang mengembalikan HTTP `400` seketika bila format JSON rusak, atau bila tipe data bukan berupa string.
3. **Very Long Text (> 50,000 karakter)**: Dibatasi melalui konfigurasi `MAX_COMBINED_LENGTH = 50000`. Jika panjang total gabungan melebihi batas, request langsung ditolak dengan HTTP `400` untuk mencegah serangan Denial-of-Service (DoS) dan kehabisan memori (*OOM*).
4. **Non-English & Karakter Unicode**: Pipeline menggunakan tokenisasi berbasis Unicode (*UTF-8 full compliance*). Karakter beraksen, simbol mata uang (*€, RM*), dan karakter non-ASCII dihitung secara konsisten menggunakan offset byte/karakter Python 3 string indexing, sehingga posisi karakter entitas tidak pernah bergeser.
5. **Nama Ambigu**: Model memanfaatkan jendela fitur kontekstual (*Tok2Vec embeddings*) sebelum dan sesudah kata target untuk menentukan apakah token seperti "Washington" berfungsi sebagai subjek politis, lokasi geografis, atau nama orang.

---

### Pertanyaan 6: If this service were deployed to production handling 1,000 requests/minute, what architectural changes would you make?

**Jawaban:**
Untuk melayani beban 1.000 requests/menit (~17–20 RPS rata-rata dengan lonjakan hingga 50–100 RPS):
1. **Horizontal Scaling & Containerization**:
   - Membungkus aplikasi ke dalam *Docker container* ringan (*distroless* atau *Alpine Python*).
   - Mendeploy di Kubernetes cluster (EKS/GKE) dengan *Horizontal Pod Autoscaler* (HPA) berbasis metrik utilisasi CPU dan latensi p95.
   - Menjalankan Uvicorn dengan multi-worker process (`uvicorn app:app --workers 4`).
2. **API Gateway & Rate Limiting**:
   - Menggunakan Kong atau AWS API Gateway di depan service untuk menangani otentikasi JWT/API Key, SSL termination, DDoS protection, dan rate limiting per client.
3. **Caching Layer Terdistribusi (Redis)**:
   - Artikel berita sering dibaca atau dianalisis berulang kali. Mengimplementasikan cache berbasis Redis dengan key hash SHA-256 dari `(title + body)`. Request yang identik akan langsung dilayani dari memory cache dalam <2 ms.
4. **Asynchronous Batching & Queue Processing**:
   - Untuk artikel berita panjang atau pemrosesan batch, menyediakan mode asinkron menggunakan RabbitMQ / Apache Kafka dengan Celery worker, di mana klien menerima task ID dan dapat melakukan pooling status / menerima webhook.

---

### Pertanyaan 7: How would you monitor entity extraction quality in production over time without ground truth labels?

**Jawaban:**
Tanpa adanya ground truth instan di production, pemantauan kualitas dilakukan melalui pendekatan statistik dan proxy metrics:
1. **Distribusi Kelas & Tingkat Deteksi Entitas (*Entity Frequency Drift*)**:
   - Memonitor proporsi kemunculan setiap kelas entitas (`PERSON`, `ORG`, `GPE`, `DATE`, `EVENT`) per jam/hari. Perubahan drastis (misal deteksi `ORG` turun 40%) mengindikasikan adanya drift pada gaya penulisan artikel atau kegagalan upstream scraper.
2. **Confidence Score / Model Uncertainty Tracking**:
   - Merekam log probabilitas/log-likelihood dari transisi parser. Lonjakan entitas dengan confidence score rendah (<0.6) menunjukkan data artikel berada di luar distribusi training (*Out-of-Distribution / Covariate Shift*).
3. **Deteksi Out-of-Vocabulary (OOV) Rate**:
   - Memantau persentase token baru yang tidak terdapat dalam vocabulary training. Tingkat OOV yang tinggi menandakan perlunya adaptasi kamus entitas baru.
4. **Input Length & Processing Latency Anomalies**:
   - Memonitor latensi inferensi p50, p95, dan p99 serta distribusi panjang karakter artikel via Prometheus dan Grafana.
5. **Periodic Random Sampling for Human Review (Shadow Evaluation)**:
   - Mengambil sampel acak 1% dari total request production setiap minggu untuk dianotasi oleh tim auditor manusia demi menghitung estimasi berkala Precision/Recall production.

---

### Pertanyaan 8: How would you extend this system to support incremental learning as human annotators review and correct extractions?

**Jawaban:**
Arsitektur *Incremental Learning* yang tangguh dibangun dengan mekanisme berikut:
1. **Feedback Loop UI / Annotation Ingestion**:
   - Menyediakan fitur di Web UI di mana reviewer dapat mengklik entitas, memperbaiki batas span, atau mengubah label, lalu menekan tombol *"Submit Correction"*.
   - Data koreksi disimpan ke database PostgreSQL/MongoDB terpusat dalam skema versioned JSON.
2. **Mencegah *Catastrophic Forgetting* (Replay Buffer Strategy)**:
   - Pembelajaran bertahap tidak boleh hanya melatih model pada koreksi baru semata, karena dapat merusak pengetahuan sebelumnya.
   - Menggunakan metode *Experience Replay*: setiap batch training baru disusun dengan kombinasi 20% data koreksi baru dan 80% data training historis yang dipilih secara stratified.
3. **Continuous Integration & Shadow Deployment (Canary Releases)**:
   - Model baru dilatih secara berkala (misal: mingguan) melalui automated CI/CD pipeline (GitHub Actions / Kubeflow Pipelines).
   - Model kandidat dievaluasi secara otomatis terhadap *regression benchmark suite* (termasuk `data/evaluation_examples.json`).
   - Jika metrik F1 meningkat dan tidak ada regresi pada held-out test set, model baru dideploy sebagai *Canary / Shadow Model* (menerima 10% traffic riil) sebelum dipromosikan menggantikan model aktif.

---

## 8. Keterbatasan Sistem (Known Limitations)

1. **Cakupan Bahasa Utama**: Model dilatih dan dioptimalkan secara mendalam untuk artikel berbahasa Inggris dan konvensi regional Malaysia/Asia Tenggara. Artikel dalam bahasa Mandarin atau Vietnam tidak dijamin memiliki akurasi ekstraksi yang sama tingginya.
2. **Ketergantungan pada Kapitalisasi Huruf**: Sebagaimana model tokenisasi berbasis teks berita, performa ekstraksi nama orang dan organisasi akan menurun secara signifikan jika seluruh teks artikel ditulis dalam huruf kecil (*all-lowercase*) atau huruf besar semua (*ALL-CAPS*).
3. **Keterbatasan Nested Entities**: Arsitektur saat ini menggunakan segmentasi token tunggal yang tidak mendukung entitas bersarang (*nested entities*), misalnya mengidentifikasi "Bank of England" sebagai `ORG` sekaligus "England" di dalamnya sebagai `GPE`.

---

## 9. Struktur Direktori Proyek

```
UTS-Article-Named-Entity-Recognition-Service/
├── .gitignore                          # Aturan pengabaian file cache & data mentah
├── README.md                           # Dokumentasi komprehensif layanan
├── requirements.txt                    # Versi dependensi yang dipin
├── CONTEXT.md                          # Dokumen spesifikasi asli studi kasus
├── data/
│   ├── profiling_results.json          # Ringkasan data profiling dataset sampel
│   ├── split_metadata.json             # Metadata split data dan skema label
│   ├── evaluation_examples.json        # 10+ contoh uji terkurasi dan terverifikasi
│   ├── evaluation_results.json         # Hasil evaluasi lengkap (validation & test)
│   ├── annotations_train.jsonl         # Anotasi data training (70%)
│   ├── annotations_val.jsonl           # Anotasi data validasi (15%)
│   └── annotations_test.jsonl          # Anotasi held-out test set (15%)
├── models/
│   └── ner_finetuned/                  # Artefak model spaCy hasil pelatihan
│       ├── config.cfg
│       ├── meta.json
│       ├── tokenizer
│       ├── ner/
│       └── vocab/
├── screenshots/
│   ├── home_page.png                   # Tangkapan layar antarmuka Web UI utama
│   └── ui_demo.png                     # Tangkapan layar hasil ekstraksi entitas
├── src/
│   ├── data_profiling.py               # Skrip inspeksi & profiling dataset
│   ├── annotation_pipeline.py          # Pipeline anotasi & deterministic data splitting
│   ├── train_ner.py                    # Pipeline training, evaluasi & error analysis
│   ├── app.py                          # Backend FastAPI & endpoint REST API
│   └── templates/
│       └── index.html                  # Antarmuka web interaktif (HTML5 + CSS modern)
└── tests/
    └── test_api.py                     # 22 automated tests (pytest) lulus 100%
```
