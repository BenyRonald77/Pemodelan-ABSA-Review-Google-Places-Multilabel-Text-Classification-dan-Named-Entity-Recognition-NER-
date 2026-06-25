# Kelp4 - ABSA Review Google Places

## Multilabel Text Classification dan Named Entity Recognition

Proyek ini merupakan proyek UAS mata kuliah **Natural Language Processing / Pengolahan Bahasa Alami**. Proyek ini adalah kelanjutan dari proyek UTS Aspect-Based Sentiment Analysis (ABSA) berbasis review Google Places.

Sistem yang dibangun terdiri dari dua model utama:

1. **Multilabel Text Classification ABSA**
   Model ini digunakan untuk memprediksi satu atau lebih label aspek dan sentimen pada level review.

2. **Named Entity Recognition ABSA**
   Model ini digunakan untuk mengenali span/aspek pada teks review menggunakan skema BIO.

Aplikasi deployment dibuat menggunakan **Streamlit** dan memiliki tiga tab utama:

1. Prediksi Multilabel ABSA
2. Prediksi NER ABSA
3. Hubungan Multilabel dan NER

---

## 1. Tujuan Proyek

Tujuan proyek ini adalah membangun sistem ABSA yang mampu:

1. Membaca dan memvalidasi dataset hasil anotasi ABSA.
2. Melakukan pemodelan Multilabel Text Classification pada level review.
3. Melakukan pemodelan Named Entity Recognition pada level token/span.
4. Mengevaluasi performa model menggunakan metrik yang sesuai.
5. Melakukan analisis kesalahan model.
6. Menyimpan model terbaik.
7. Membuat aplikasi Streamlit untuk demo prediksi review baru.
8. Menjelaskan hubungan antara output multilabel dan output NER.

---

## 2. Dataset

Dataset yang digunakan berasal dari hasil proyek UTS kelompok 4. Dataset utama berada dalam format JSONL dan berisi teks review, label multilabel, token, serta span hasil anotasi.

File dataset utama:

```text
Kelp4_dataset_anotasi.jsonl
```

Kolom penting pada dataset:

| Kolom    | Keterangan                |
| -------- | ------------------------- |
| `text`   | Teks review pelanggan     |
| `accept` | Label multilabel ABSA     |
| `tokens` | Token hasil anotasi       |
| `spans`  | Span/entity hasil anotasi |
| `answer` | Status anotasi            |

Jumlah data setelah validasi dan pembersihan:

| Keterangan                   | Jumlah |
| ---------------------------- | -----: |
| Data awal                    |   2118 |
| Data valid setelah filtering |   2103 |
| Data train                   |   1682 |
| Data validation              |    210 |
| Data test                    |    211 |

---

## 3. Skema Label ABSA

Label multilabel yang digunakan terdiri dari 13 label:

```text
PRODUCT_POSITIVE
PRODUCT_NEGATIVE
PRODUCT_NEUTRAL
PRICE_POSITIVE
PRICE_NEGATIVE
PRICE_NEUTRAL
PLACE_POSITIVE
PLACE_NEGATIVE
PLACE_NEUTRAL
PROMOTION_POSITIVE
PROMOTION_NEGATIVE
PROMOTION_NEUTRAL
OUT_OF_TOPIC
```

Untuk NER, label span dikonversi menjadi tag BIO:

```text
B-PRODUCT
I-PRODUCT
B-PRICE
I-PRICE
B-PLACE
I-PLACE
B-PROMOTION
I-PROMOTION
O
```

---

## 4. Struktur Folder

Struktur folder proyek:

```text
project/
├── dataset/
│   ├── Kelp4_dataset_1.csv
│   ├── Kelp4_dataset_2.csv
│   ├── Kelp4_dataset_anotasi.jsonl
│   ├── Kelp4_multilabel_train.csv
│   ├── Kelp4_multilabel_val.csv
│   ├── Kelp4_multilabel_test.csv
│   ├── Kelp4_ner_train.jsonl
│   ├── Kelp4_ner_val.jsonl
│   └── Kelp4_ner_test.jsonl
│
├── notebook/
│   ├── Kelp4_multilabel_modeling.ipynb
│   └── Kelp4_ner_modeling.ipynb
│
├── models/
│   ├── Kelp4_best_multilabel_model.pkl
│   ├── Kelp4_best_ner_model.keras
│   ├── Kelp4_multilabel_labels.pkl
│   ├── Kelp4_multilabel_thresholds.pkl
│   ├── Kelp4_ner_word2idx.pkl
│   ├── Kelp4_ner_idx2tag.pkl
│   └── Kelp4_ner_config.pkl
│
├── streamlit/
│   └── Kelp4_app.py
│
├── report/
│   └── Kelp4_laporan_uas.pdf
│
├── code_pdf/
│   └── Kelp4_code_export.pdf
│
├── Kelp4_anggota.txt
├── README.md
└── PENJELASAN.md
```

---

## 5. Metode Multilabel Text Classification

Tahapan pada proyek Multilabel Text Classification:

1. Membaca dataset anotasi JSONL.
2. Memvalidasi kolom penting.
3. Menghapus data yang tidak valid.
4. Mengubah label menjadi format multilabel biner.
5. Melakukan preprocessing teks.
6. Melakukan EDA dan visualisasi label.
7. Membagi data menjadi train, validation, dan test.
8. Membandingkan representasi teks.
9. Melatih beberapa model.
10. Mengevaluasi model.
11. Melakukan error analysis.
12. Menyimpan model terbaik.

Representasi teks yang digunakan:

1. TF-IDF
2. Bag of Words

Model yang dibandingkan:

| No | Model                                         |
| -: | --------------------------------------------- |
|  1 | TF-IDF + OneVsRest Logistic Regression        |
|  2 | Bag of Words + OneVsRest Logistic Regression  |
|  3 | TF-IDF + Classifier Chain Logistic Regression |
|  4 | TF-IDF + OneVsRest Linear SVM                 |
|  5 | Tuned TF-IDF + OneVsRest Logistic Regression  |

Model terbaik:

```text
Tuned TF-IDF + OneVsRest Logistic Regression
```

---

## 6. Hasil Evaluasi Multilabel

Hasil evaluasi model terbaik pada test set:

| Metrik          |  Nilai |
| --------------- | -----: |
| Micro-F1        | 0.7021 |
| Macro-F1        | 0.5036 |
| Weighted-F1     | 0.7224 |
| Hamming Loss    | 0.0941 |
| Subset Accuracy | 0.3318 |

Interpretasi:

* Weighted-F1 sebesar 0.7224 menunjukkan bahwa model cukup baik dalam memprediksi label dominan.
* Macro-F1 lebih rendah karena performa model belum merata pada seluruh label.
* Subset Accuracy lebih rendah karena pada multilabel classification prediksi hanya dianggap benar apabila seluruh label sama persis dengan ground truth.
* Dataset memiliki label imbalance, terutama karena label `PRODUCT_POSITIVE` jauh lebih dominan dibanding label minoritas seperti `PRICE_NEUTRAL`, `PLACE_NEUTRAL`, dan `PROMOTION_NEUTRAL`.

---

## 7. Metode Named Entity Recognition

Tahapan pada proyek NER:

1. Membaca dataset anotasi JSONL.
2. Memvalidasi token dan span.
3. Mengonversi span menjadi tag BIO.
4. Membagi data menjadi train, validation, dan test.
5. Membuat vocabulary token.
6. Membuat tag mapping.
7. Melakukan padding dan truncation.
8. Melatih model BiLSTM.
9. Mengevaluasi model.
10. Melakukan error analysis.
11. Menyimpan model dan mapping.

Model yang digunakan:

```text
BiLSTM
```

Arsitektur model:

1. Embedding layer
2. Bidirectional LSTM
3. Dropout
4. TimeDistributed Dense dengan aktivasi softmax

---

## 8. Hasil Evaluasi NER

Hasil evaluasi model NER pada test set:

| Metrik      |  Nilai |
| ----------- | -----: |
| Accuracy    | 0.6651 |
| Macro-F1    | 0.4240 |
| Weighted-F1 | 0.6365 |

Interpretasi:

* Model cukup baik dalam mengenali tag dominan seperti `I-PRODUCT`, `I-PLACE`, dan `O`.
* Model masih lemah pada tag minoritas seperti `B-PRICE`, `B-PROMOTION`, dan `I-PROMOTION`.
* Kesalahan paling banyak adalah Entity vs O Error, yaitu model salah membedakan token entitas dan token non-entitas.

---

## 9. Deployment Streamlit

Aplikasi Streamlit dibuat pada file:

```text
streamlit/Kelp4_app.py
```

Aplikasi memiliki tiga tab:

1. **Prediksi Multilabel ABSA**
   Menampilkan input review, label prediksi, confidence score, threshold, dan interpretasi hasil.

2. **Prediksi NER ABSA**
   Menampilkan token, predicted tag, entitas/aspek yang diekstraksi, dan highlight span.

3. **Hubungan Multilabel dan NER**
   Menjelaskan hubungan antara label aspek-sentimen pada level review dan span/aspek pada level token.

---

## 10. Post-processing Deployment

Pada aplikasi Streamlit, ditambahkan post-processing sederhana berdasarkan error analysis.

Post-processing ini tidak menggantikan model utama. Model utama tetap digunakan untuk menghasilkan confidence score dan prediksi awal. Post-processing hanya membantu merapikan output yang jelas kontradiktif.

Contoh:

```text
Baksonya tidak enak → PRODUCT_NEGATIVE
Harga mahal → PRICE_NEGATIVE
Harga murah → PRICE_POSITIVE
```

Tujuan post-processing adalah mengurangi kesalahan umum pada negasi dan label yang kontradiktif.

---

## 11. Cara Menjalankan Proyek

### 11.1 Membuat Environment

Contoh environment yang digunakan:

```text
Python 3.11
TensorFlow
scikit-learn
pandas
numpy
matplotlib
joblib
streamlit
```

Install package utama:

```bash
pip install pandas numpy scikit-learn matplotlib tensorflow streamlit joblib
```

---

### 11.2 Menjalankan Notebook

Jalankan notebook secara berurutan:

```text
notebook/Kelp4_multilabel_modeling.ipynb
notebook/Kelp4_ner_modeling.ipynb
```

Notebook pertama akan menghasilkan model multilabel dan file pendukung. Notebook kedua akan menghasilkan model NER dan file mapping.

---

### 11.3 Menjalankan Streamlit

Masuk ke folder project, lalu jalankan:

```bash
streamlit run streamlit/Kelp4_app.py
```

Atau menggunakan environment Python tertentu:

```bash
python -m streamlit run streamlit/Kelp4_app.py
```

---

## 12. Contoh Input

Contoh input multilabel:

```text
Baksonya tidak enak
```

Output yang diharapkan:

```text
PRODUCT_NEGATIVE
```

Contoh input harga:

```text
Harga mahal
```

Output yang diharapkan:

```text
PRICE_NEGATIVE
```

Contoh input gabungan:

```text
Baksonya enak, tempatnya luas, tapi harganya mahal.
```

Output yang diharapkan:

```text
PRODUCT_POSITIVE
PLACE_POSITIVE
PRICE_NEGATIVE
```

---

## 13. Keterbatasan

Beberapa keterbatasan proyek:

1. Dataset memiliki label imbalance.
2. Label minoritas sulit diprediksi.
3. Model multilabel masih dapat salah pada kalimat negasi yang kompleks.
4. Model NER masih lemah pada aspek price dan promotion.
5. Tokenisasi NER pada aplikasi Streamlit masih sederhana.
6. Post-processing hanya menangani beberapa pola kesalahan umum.

---
[Isi link video demo]
```
