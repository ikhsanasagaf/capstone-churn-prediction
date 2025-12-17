# Telco Customer Churn Prediction 📊

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-Deployment-red)
![Scikit-Learn](https://img.shields.io/badge/Library-Scikit--Learn-orange)

## 📝 Deskripsi Proyek

Proyek ini merupakan **Capstone Project** untuk Ujian Akhir Semester (UAS) mata kuliah **Bengkel Koding Data Science** di **Universitas Dian Nuswantoro (UDINUS)**.

**Latar Belakang:**
Dalam industri telekomunikasi, *churn* pelanggan (berhenti berlangganan) adalah masalah kritis yang berdampak langsung pada pendapatan perusahaan. Mempertahankan pelanggan lama jauh lebih efisien daripada mencari pelanggan baru. Oleh karena itu, kemampuan untuk memprediksi pelanggan mana yang berpotensi melakukan *churn* sangat penting agar perusahaan dapat mengambil tindakan pencegahan yang tepat.

**Tujuan:**
Membangun model *Machine Learning* untuk memprediksi apakah seorang pelanggan akan berhenti berlangganan (*Yes*) atau tetap berlangganan (*No*) berdasarkan data historis dan demografis, serta men-deploy model tersebut ke dalam aplikasi web interaktif.

## 📂 Dataset

Dataset yang digunakan adalah **Telco Customer Churn** yang bersumber dari Kaggle. Dataset ini terdiri dari **7.043 baris** data pelanggan dengan **21 fitur**.

* **Sumber Data:** [Kaggle - Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
* **Target Variabel:** `Churn` (Yes/No)

### Fitur Utama:
| Kategori | Kolom |
| :--- | :--- |
| **Demografis** | `gender`, `SeniorCitizen`, `Partner`, `Dependents` |
| **Layanan** | `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies` |
| **Akun** | `tenure`, `Contract`, `PaperlessBilling`, `PaymentMethod`, `MonthlyCharges`, `TotalCharges` |

## 🛠️ Metodologi

Proyek ini dikerjakan melalui beberapa tahapan sistematis untuk memastikan validitas model:

1. **Exploratory Data Analysis (EDA):**
* Pengecekan statistik deskriptif dan tipe data.
* Identifikasi *missing values* (khususnya penanganan data kosong pada `TotalCharges`).
* Visualisasi distribusi target (`Churn`) yang menunjukkan ketidakseimbangan data (Imbalanced Data).
* Analisis korelasi antar fitur numerik.
2. **Preprocessing Data:**
* **Data Cleaning:** Menangani nilai kosong dan memastikan konsistensi tipe data.
* **Encoding:** Menggunakan *One-Hot Encoding* untuk fitur kategorikal nominal (seperti `PaymentMethod`, `InternetService`) agar dapat diproses oleh algoritma numerik.
* **Scaling:** Menerapkan *StandardScaler* pada fitur numerik untuk meningkatkan performa model.
* **Splitting:** Membagi data menjadi Training (80%) dan Testing (20%).
3. **Pemodelan (Modeling):**
Dilakukan dalam 3 skenario perbandingan untuk melihat dampak preprocessing dan optimasi:
* **Tahap 1:** Direct Modeling.
* **Tahap 2:** Modeling setelah Preprocessing.
* **Tahap 3:** Hyperparameter Tuning menggunakan `GridSearchCV`.
* **Algoritma yang digunakan:** Logistic Regression, Random Forest, dan Voting Classifier (Soft Voting: LR + RF + KNN).
4. **Evaluasi:**
Menggunakan confusion matrix yang berfokus pada **Recall** dan **F1-Score** (kelas Churn).
5. **Deployment:**
Model terbaik (**Logistic Regression Tuned**) disimpan dalam format `.pkl` dan diintegrasikan ke dalam aplikasi web interaktif menggunakan **Streamlit**.

## 💻 Teknologi yang Digunakan

* **Bahasa Pemrograman:** Python
* **Data Manipulation:** Pandas, NumPy
* **Visualisasi:** Matplotlib, Seaborn
* **Machine Learning:** Scikit-Learn
* **Deployment:** Streamlit Cloud

## 📊 Hasil Evaluasi Model

Berikut adalah perbandingan performa model setelah dilakukan **Hyperparameter Tuning**.

| Model | Accuracy | Precision (Churn) | Recall (Churn) | F1-Score (Churn) |
| --- | --- | --- | --- | --- |
| **Logistic Regression (Tuned)** | **83%** | **0.63** | **0.59** | **0.61** |
| Random Forest (Tuned) | 81% | 0.60 | 0.53 | 0.56 |
| Voting Classifier (Tuned) | 82% | 0.61 | 0.56 | 0.58 |

### 🏆 Model Terpilih

Model **Logistic Regression (Tuned)** dipilih sebagai model final untuk deployment karena:

1. Memiliki **F1-Score kelas Churn tertinggi (0.61)** dan **Recall tertinggi (0.59)** dibandingkan model lainnya.
2. Mampu mendeteksi pelanggan berisiko (Churn) dengan lebih baik.

## 🌐 Link Deployment

Aplikasi prediksi churn dapat diakses secara publik melalui tautan berikut:
**[Link ke Aplikasi Streamlit Anda]**

## 👤 Author

**Muhammad Ikhsan Asagaf**
* NIM: A11.2022.14255
* Program Studi: Teknik Informatika
* Universitas Dian Nuswantoro

---
*Proyek ini dibuat untuk memenuhi tugas Ujian Akhir Semester Bengkel Koding Data Science.*
