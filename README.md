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

Proyek ini dikerjakan melalui beberapa tahapan sistematis:

1.  **Exploratory Data Analysis (EDA):**
    * Pengecekan statistik deskriptif dan tipe data.
    * Identifikasi *missing values* (terutama pada `TotalCharges`).
    * Visualisasi distribusi target (`Churn`) untuk melihat keseimbangan kelas.
    * Analisis korelasi antar fitur numerik.

2.  **Preprocessing Data:**
    * Cleaning data (menangani nilai kosong, duplikasi, dan outlier).
    * Encoding fitur kategorikal (One-Hot Encoding / Label Encoding).
    * Scaling fitur numerik.
    * Train-Test Split.

3.  **Pemodelan (Modeling):**
    Dilakukan dalam 3 skenario (Direct, Preprocessed, Hyperparameter Tuning)

4.  **Evaluasi:**
    Menggunakan metrik: Accuracy, Precision, Recall, F1-Score, dan Confusion Matrix untuk memilih model terbaik.

5.  **Deployment:**
    Model terbaik disimpan (format `.pkl` atau `.joblib`) dan diintegrasikan ke dalam aplikasi web menggunakan **Streamlit**.

## 💻 Teknologi yang Digunakan

* **Bahasa Pemrograman:** Python
* **Data Manipulation:** Pandas, NumPy
* **Visualisasi:** Matplotlib, Seaborn
* **Machine Learning:** Scikit-Learn, XGBoost, Imbalanced-learn (SMOTE)
* **Deployment:** Streamlit Cloud

## 📊 Hasil Evaluasi Model

*(Bagian ini akan diisi setelah proses training selesai. Contoh format:)*

| Model | Accuracy | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| Logistic Regression | 80% | 0.xx | 0.xx | 0.xx |
| **Random Forest (Tuned)** | **82%** | **0.xx** | **0.xx** | **0.xx** |
| Voting Classifier | 81% | 0.xx | 0.xx | 0.xx |

*Model terbaik yang dipilih untuk deployment adalah: [Nama Model]*

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
