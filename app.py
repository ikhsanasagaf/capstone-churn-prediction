import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(
    page_title="Churn Prediction by Ikhsan",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS untuk mempercantik
st.markdown("""
<style>
    .metric-card {
        background-color: #F8F9FA;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #007BFF; /* Sesuaikan dengan primaryColor */
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }          
    .status-box {
        padding: 30px;
        border-radius: 20px;
        text-align: center;
        margin-bottom: 20px;
    }
    .box-churn {
        background-color: #fdecea;
        border: 3px solid #e74c3c; 
        color: #c0392b; 
    }
    .box-loyal {
        background-color: #eafaf1;
        border: 3px solid #2ecc71; 
        color: #27ae60; 
    }
    .status-title {
        font-size: 3rem;
        font-weight: bold;
        margin: 0;
    }
    .status-subtitle {
        font-size: 1.5rem;
        font-weight: bold;
        margin: 10px 0;
    }
            
            
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model_data():
    try:
        data = joblib.load('churn_prediction_bestmodel.pkl')
        return data
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

data = load_model_data()

if not data:
    st.stop()

model = data['model']
scaler = data['scaler']
feature_names = data['features']

# Ambil mean training untuk referensi
train_means = dict(zip(['tenure', 'MonthlyCharges', 'TotalCharges'], scaler.mean_))

# ==========================================
# SIDEBAR & INPUT FORM
# ==========================================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/128/12489/12489733.png", width=100)
    st.title("Input Data Pelanggan")
    st.write("Lengkapi form di bawah ini:")
    
    # --- DEMOGRAFI ---
    with st.expander("👤 Demografi", expanded=True):
        gender = st.selectbox("Gender", ["Male", "Female"])
        senior = st.selectbox("Senior Citizen", ["No", "Yes"])
        partner = st.selectbox("Partner", ["No", "Yes"])
        dependents = st.selectbox("Dependents", ["No", "Yes"])

    # --- LAYANAN UTAMA ---
    with st.expander("📡 Layanan", expanded=True):
        tenure = st.slider("Tenure (Bulan)", 0, 72, 12)
        phone_service = st.selectbox("Phone Service", ["No", "Yes"])
        
        # Logika dinamis untuk Multiple Lines
        ml_opts = ["No phone service"] if phone_service == "No" else ["No", "Yes"]
        multiple_lines = st.selectbox("Multiple Lines", ml_opts)
        
        internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])

    # --- LAYANAN TAMBAHAN ---
    with st.expander("🛡️ Layanan Tambahan"):
        # Opsi dinamis berdasarkan internet service
        if internet_service == "No":
            opts_internet = ["No internet service"]
        else:
            opts_internet = ["No", "Yes"]
            
        online_sec = st.selectbox("Online Security", opts_internet)
        online_backup = st.selectbox("Online Backup", opts_internet)
        device_prot = st.selectbox("Device Protection", opts_internet)
        tech_supp = st.selectbox("Tech Support", opts_internet)
        stream_tv = st.selectbox("Streaming TV", opts_internet)
        stream_mov = st.selectbox("Streaming Movies", opts_internet)

    # --- AKUN & TAGIHAN ---
    with st.expander("💳 Akun & Pembayaran", expanded=True):
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless = st.selectbox("Paperless Billing", ["No", "Yes"])
        payment = st.selectbox("Payment Method", [
            "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
        ])
        monthly_charges = st.number_input("Monthly Charges", min_value=0.0, value=50.0)
        total_charges = st.number_input("Total Charges", min_value=0.0, value=500.0)

    btn_predict = st.button("🔍 Prediksi Sekarang", type="primary", use_container_width=True)

# ==========================================
# FUNGSI PREPROCESSING
# ==========================================
def preprocess_data():
    # 1. Inisialisasi DataFrame dengan 0 untuk semua fitur model
    input_df = pd.DataFrame(0, index=[0], columns=feature_names)
    
    # 2. Input Numerik & Scaling
    # Buat DF sementara untuk scaling agar warning sklearn hilang (fitur harus urut sesuai scaler)
    df_num = pd.DataFrame([[tenure, monthly_charges, total_charges]], 
                          columns=['tenure', 'MonthlyCharges', 'TotalCharges'])
    scaled_vals = scaler.transform(df_num)
    
    input_df['tenure'] = scaled_vals[0][0]
    input_df['MonthlyCharges'] = scaled_vals[0][1]
    input_df['TotalCharges'] = scaled_vals[0][2]
    
    # 3. Mapping Kategorikal Manual
    # Fitur Binary Sederhana
    if senior == "Yes": input_df['SeniorCitizen'] = 1
    if gender == "Male": input_df['gender_Male'] = 1
    if partner == "Yes": input_df['Partner_Yes'] = 1
    if dependents == "Yes": input_df['Dependents_Yes'] = 1
    if phone_service == "Yes": input_df['PhoneService_Yes'] = 1
    if paperless == "Yes": input_df['PaperlessBilling_Yes'] = 1
    
    # Fitur One-Hot Encoding (Otomatis mencocokkan nama kolom)
    # List tuple: (Nama Kolom di Model, Kondisi agar bernilai 1)
    
    one_hot_mappings = [
        (f'MultipleLines_{multiple_lines}', True),
        (f'InternetService_{internet_service}', True),
        (f'OnlineSecurity_{online_sec}', True),
        (f'OnlineBackup_{online_backup}', True),
        (f'DeviceProtection_{device_prot}', True),
        (f'TechSupport_{tech_supp}', True),
        (f'StreamingTV_{stream_tv}', True),
        (f'StreamingMovies_{stream_mov}', True),
        (f'Contract_{contract}', True),
        (f'PaymentMethod_{payment}', True)
    ]
    
    for col_name, condition in one_hot_mappings:
        # Cek apakah kolom hasil kombinasi string ini ada di fitur model
        # Contoh: Jika user pilih 'DSL', kita cek apakah 'InternetService_DSL' ada di model?
        # Jika tidak ada (misal DSL adalah base case/dummy trap), maka biarkan 0 semua.
        if col_name in input_df.columns and condition:
            input_df[col_name] = 1
            
    return input_df

# ==========================================
# 5. DASHBOARD UTAMA
# ==========================================
st.title("📊 Customer Churn Prediction - Dashboard")

# Tab Layout
tab_pred, tab_feat, tab_about = st.tabs(["⚡ Hasil Prediksi", "🧠 Informasi Fitur", "ℹ️ Tentang Project"])

# --- TAB 1: HASIL PREDIKSI ---
with tab_pred:
    if btn_predict:
        final_input = preprocess_data()
        
        # Prediksi
        prob = model.predict_proba(final_input)[0][1]
        
        # ==============================
        # LAYOUT SATU KOLOM (VERTICAL)
        # ==============================
        
        st.markdown("### Status Risiko")
        
        # 1. Kotak Status Utama
        if prob > 0.5:
            # Menggunakan f-string HTML untuk Churn
            st.markdown(f"""
            <div class="status-box box-churn">
                <div class="status-title">⚠️ CHURN</div>
                <div class="status-subtitle">Pelanggan Berisiko Tinggi</div>
                <p>Model mendeteksi pola perilaku yang mengarah pada pemberhentian layanan. Disarankan segera berikan penawaran retensi.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            # Menggunakan f-string HTML untuk Loyal
            st.markdown(f"""
            <div class="status-box box-loyal">
                <div class="status-title">✅ LOYAL</div>
                <div class="status-subtitle">Pelanggan Aman</div>
                <p>Model memprediksi pelanggan ini akan tetap setia. Tidak diperlukan tindakan pencegahan khusus saat ini.</p>
            </div>
            """, unsafe_allow_html=True)
            
        # 2. Angka Probabilitas & Progress Bar
        # Saya tambahkan progress bar agar visualnya lebih informatif di tampilan satu kolom
        st.metric("Probabilitas Churn", f"{prob*100:.2f}%")
        st.progress(prob)
        
        # 3. Saran/Insight Singkat (Opsional - ambil dari logika sebelumnya)
        if contract == "Month-to-month" and prob > 0.5:
            st.warning("👉 **Saran:** Tawarkan kontrak jangka panjang (1-2 tahun) untuk mengikat pelanggan.")
        elif internet_service == "Fiber optic" and prob > 0.5:
            st.info("👉 **Saran:** Cek riwayat keluhan teknis. Pengguna Fiber Optic sensitif terhadap gangguan.")
        
        # Spacer (Jarak)
        st.write("") 
        
        # 4. Expander Data Input (Ditaruh di paling bawah sesuai request)
        with st.expander("Lihat Data Input Processed (Debug)"):
            st.dataframe(final_input)
            
    else:
        st.info("👈 Silakan isi data di sidebar dan klik 'Prediksi Sekarang'")


# --- TAB 2: Informasi Fitur ---
with tab_feat:
    st.subheader("🔍 Bobot Fitur")
    st.write("Visualisasi pengaruh setiap fitur terhadap prediksi Churn.")
    
    # Ambil semua koefisien
    coefs = model.coef_[0]
    df_coef = pd.DataFrame({
        'Fitur': feature_names,
        'Bobot (Importance)': coefs
    })
    
    # Sortir dari positif (Churn) ke negatif (Loyal)
    df_coef = df_coef.sort_values(by='Bobot (Importance)', ascending=False)
    
    # Visualisasi Tinggi (Scrollable)
    fig_feat, ax_feat = plt.subplots(figsize=(10, 12))  # Tinggi figure dibuat panjang
    
    # Warna: Merah untuk > 0 (Churn), Hijau untuk < 0 (Loyal)
    colors = ['#e74c3c' if x > 0 else '#2ecc71' for x in df_coef['Bobot (Importance)']]
    
    sns.barplot(x='Bobot (Importance)', y='Fitur', data=df_coef, palette=colors, ax=ax_feat)
    
    # Dekorasi
    ax_feat.set_xlabel("Pengaruh (Positif = Cenderung Churn)")
    ax_feat.axvline(0, color='black', linewidth=1)
    ax_feat.grid(axis='x', linestyle='--', alpha=0.7)
    
    st.pyplot(fig_feat)
    
    st.info("""
    **Keterangan:**
    * **Merah :** Faktor terkuat penyebab pelanggan KELUAR.
    * **Hijau :** Faktor terkuat penyebab pelanggan BERTAHAN.
    """)


    # --- TAB 3: ABOUT PROJECT ---
with tab_about:
    st.markdown("""
    ![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
    ![Streamlit](https://img.shields.io/badge/Streamlit-Deployment-red)
    ![Scikit-Learn](https://img.shields.io/badge/Library-Scikit--Learn-orange)
    """)

    st.header("📝 Deskripsi Proyek")
    st.write("""
    Proyek ini merupakan **Capstone Project** untuk Ujian Akhir Semester (UAS) mata kuliah **Bengkel Koding Data Science** di **Universitas Dian Nuswantoro (UDINUS)**.
    """)

    col_about1, col_about2 = st.columns(2)
    with col_about1:
        st.info("""
        **Latar Belakang**
        Dalam industri telekomunikasi, *churn* pelanggan (berhenti berlangganan) adalah masalah kritis. 
        Mempertahankan pelanggan lama jauh lebih efisien daripada mencari pelanggan baru. 
        Prediksi *churn* memungkinkan perusahaan mengambil tindakan pencegahan yang tepat.
        """)
    
    with col_about2:
        st.success("""
        **Tujuan**
        Membangun model *Machine Learning* untuk memprediksi apakah pelanggan akan berhenti berlangganan (*Yes*) 
        atau tetap berlangganan (*No*) berdasarkan data historis dan demografis.
        """)

    st.header("📂 Dataset")
    st.markdown("""
    Dataset yang digunakan adalah **Telco Customer Churn** dari Kaggle.
    * **Total Data:** 7.043 baris
    * **Fitur:** 21 fitur
    * **Target:** `Churn` (Yes/No)
    """)
    

    st.header("🛠️ Metodologi")
    st.markdown("""
    1. **EDA:** Pengecekan statistik, missing values, dan distribusi data.
    2. **Preprocessing:** Data cleaning, One-Hot Encoding, Scaling, dan Train-Test Split (80:20).
    3. **Modeling:** Menguji algoritma Logistic Regression, Random Forest, dan Voting Classifier.
    4. **Evaluasi:** Confusion Matrix yang difokuskan pada Recall dan F1 Score.
    5. **Deployment:** Integrasi model terbaik ke Streamlit.
    """)

    st.header("📊 Hasil Evaluasi Model")
    st.markdown("Berikut perbandingan performa model setelah Hyperparameter Tuning:")
    
    results_data = {
        "Model": ["Logistic Regression (Tuned)", "Random Forest (Tuned)", "Voting Classifier (Tuned)"],
        "Accuracy": ["83%", "81%", "82%"],
        "Precision": ["0.63", "0.60", "0.61"],
        "Recall": ["0.59", "0.53", "0.56"],
        "F1-Score": ["0.61", "0.56", "0.58"]
    }
    df_results = pd.DataFrame(results_data)
    st.dataframe(df_results, use_container_width=True)


    st.divider()
    st.subheader("👤 Author")
    st.markdown("""
    **Muhammad Ikhsan Asagaf** 
    * NIM: A11.2022.14255  
    * Program Studi: Teknik Informatika  
    * Universitas Dian Nuswantoro
    """)