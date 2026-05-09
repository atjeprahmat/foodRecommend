# SPKC Menu Makanan Sehat dan Ekonomis

Aplikasi ini adalah prototype dashboard Streamlit untuk **Sistem Pendukung Keputusan Cerdas Rekomendasi Menu Makanan Sehat dan Ekonomis**. Aplikasi menggabungkan pendekatan **model driven** menggunakan AHP sederhana dan TOPSIS, serta pendekatan **data driven** menggunakan K-Means Clustering.

## Dataset

Sumber data utama yang disarankan:

https://www.kaggle.com/datasets/ahsanneural/global-food-and-nutrition-database-2026

Karena dataset Kaggle tidak selalu boleh disertakan langsung dalam repository, project ini menyediakan `sample_data.csv` agar aplikasi bisa dijalankan dan diuji. Untuk analisis penuh, unduh dataset dari Kaggle, ekstrak file CSV, lalu upload melalui sidebar aplikasi.

## Fitur

- Upload dataset CSV dari Kaggle.
- Instruksi unduh dataset jika pengguna belum upload dataset.
- Preview dataset mentah.
- Preprocessing otomatis:
  - standarisasi nama kolom menjadi lowercase dan snake_case;
  - deteksi kolom nama makanan dan kategori makanan;
  - deteksi kolom nutrisi utama seperti `calories`, `protein`, `fat`, `sugar`, `fiber`, `sodium`, dan `health_score`;
  - imputasi missing value;
  - penghapusan duplikasi;
  - simulasi kolom `price` dalam Rupiah jika belum tersedia;
  - normalisasi data numerik untuk TOPSIS dan K-Means.
- Ringkasan dataset, statistik deskriptif, dan distribusi kategori.
- Visualisasi EDA: distribusi nutrisi dan heatmap korelasi.
- Filter interaktif berdasarkan kategori, harga, kalori, protein, gula, sodium, dan health score.
- Ranking rekomendasi menggunakan TOPSIS.
- Segmentasi makanan menggunakan K-Means Clustering.

## Metode

### AHP Sederhana

AHP digunakan sebagai mekanisme input bobot kriteria. Pengguna memberi skor prioritas 1 sampai 9 untuk setiap kriteria. Skor tersebut dinormalisasi menjadi bobot dengan total 1.

### TOPSIS

TOPSIS menghitung kedekatan setiap makanan terhadap solusi ideal. Kriteria yang digunakan:

Benefit criteria:

- `protein`
- `fiber`
- `health_score`

Cost criteria:

- `price`
- `calories`
- `fat`
- `sugar`
- `sodium`

Semakin tinggi skor TOPSIS, semakin direkomendasikan makanan tersebut.

### K-Means Clustering

K-Means menggunakan fitur numerik yang relevan: `price`, `calories`, `protein`, `fat`, `sugar`, `fiber`, `sodium`, dan `health_score`. Jumlah cluster default adalah 4. Cluster diberi label interpretatif:

- Sehat dan Ekonomis
- Sehat tapi Mahal
- Murah tapi Kurang Sehat
- Kurang Direkomendasikan

## Kolom Tambahan

Jika dataset memiliki kolom tambahan, aplikasi akan mencoba mempertahankan kolom relevan berikut:

- `nutri_score`: konteks kualitas nutrisi atau grade nutrisi jika tersedia.
- `nova_group`: indikator tingkat pemrosesan makanan.
- `saturated_fat`: konteks tambahan risiko lemak jenuh.
- `carbohydrate`: konteks makronutrien tambahan.
- `cholesterol`: konteks tambahan untuk makanan hewani atau makanan tinggi kolesterol.

Kolom tersebut tidak masuk kriteria TOPSIS utama agar model tetap sederhana dan sesuai requirement, tetapi tetap ditampilkan pada data hasil preprocessing untuk analisis lanjutan.

## Struktur File

- `app.py`: source code utama aplikasi Streamlit.
- `requirements.txt`: daftar dependency Python.
- `README.md`: dokumentasi project.
- `sample_data.csv`: data contoh untuk menjalankan prototype tanpa dataset Kaggle.

## Cara Menjalankan

```bash
pip install -r requirements.txt
streamlit run app.py
```

Buka URL lokal yang ditampilkan Streamlit, biasanya:

```text
http://localhost:8501
```

## Cara Menggunakan Aplikasi

1. Jalankan aplikasi dengan `streamlit run app.py`.
2. Upload CSV Kaggle melalui sidebar, atau gunakan `sample_data.csv` bawaan.
3. Periksa preview dataset dan ringkasan preprocessing.
4. Atur filter makanan sesuai kebutuhan.
5. Atur bobot kriteria di sidebar.
6. Baca hasil pada tab **Hasil Rekomendasi**:
   - 10 rekomendasi makanan terbaik;
   - tabel ranking TOPSIS;
   - grafik skor TOPSIS;
   - scatter plot cluster;
   - ringkasan interpretasi.

## Catatan

Harga pada dataset nutrisi biasanya tidak tersedia. Jika kolom harga tidak ditemukan, aplikasi membuat `price` simulasi berbasis kategori makanan dan nama makanan. Nilai ini bersifat realistis untuk prototype, tetapi bukan harga pasar aktual.

Rekomendasi aplikasi bersifat edukatif dan tidak menggantikan saran dokter, ahli gizi, atau pemeriksaan label produk bagi pengguna dengan alergi atau kondisi medis tertentu.
