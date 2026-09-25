import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------------------------------------------------
# Konfigurasi halaman
# -----------------------------------------------------------------------
st.set_page_config(
    page_title="Segmentasi Nasabah Kartu Kredit",
    page_icon="💳",
    layout="wide",
)

sns.set_style("whitegrid")

FEATURES = ["BALANCE", "PURCHASES", "CREDIT_LIMIT"]

CLUSTER_INFO = {
    0: {
        "label": "Nasabah Hemat / Low-usage",
        "desc": "Saldo rendah, jumlah pembelian rendah, dan limit kredit rendah. "
                "Segmen ini jarang bertransaksi namun jumlah nasabahnya paling banyak.",
        "rekomendasi": [
            "Kampanye edukasi manfaat kartu kredit",
            "Promo cashback kecil / gratis ongkir untuk transaksi pertama",
            "Dorong aktivasi penggunaan kartu secara rutin",
        ],
        "warna": "#4C72B0",
    },
    1: {
        "label": "Saldo Tinggi, Jarang Belanja",
        "desc": "Saldo dan limit kredit tinggi, namun jumlah pembelian relatif moderat "
                "(berpotensi besar namun belum tergali/underutilized).",
        "rekomendasi": [
            "Program cicilan 0% untuk mendorong penggunaan kartu belanja",
            "Diskon merchant partner",
            "Kartu tambahan dengan reward belanja",
        ],
        "warna": "#DD8452",
    },
    2: {
        "label": "Nasabah Premium / Pengguna Aktif",
        "desc": "Saldo tinggi, pembelian sangat tinggi, dan limit kredit tertinggi. "
                "Segmen paling bernilai meski jumlah nasabahnya paling sedikit.",
        "rekomendasi": [
            "Program loyalitas / reward premium",
            "Layanan prioritas (concierge, akses lounge bandara)",
            "Upsell produk finansial premium lainnya",
        ],
        "warna": "#55A868",
    },
}


# -----------------------------------------------------------------------
# Load artefak model & data pendukung (hasil training di notebook CRISP-DM)
# -----------------------------------------------------------------------
@st.cache_resource
def load_model_artifacts():
    scaler = joblib.load("scaler.pkl")
    model = joblib.load("kmeans_model.pkl")
    return scaler, model


@st.cache_data
def load_clustered_data():
    try:
        return pd.read_csv("clustered_customers.csv")
    except FileNotFoundError:
        return None


@st.cache_data
def load_eval_metrics():
    try:
        return pd.read_csv("model_evaluation_metrics.csv", index_col=0)
    except FileNotFoundError:
        return None


scaler, model = load_model_artifacts()
df_clustered = load_clustered_data()
eval_metrics = load_eval_metrics()

# -----------------------------------------------------------------------
# Sidebar navigasi
# -----------------------------------------------------------------------
st.sidebar.title("💳 Navigasi")
page = st.sidebar.radio(
    "Pilih halaman",
    ["🏠 Beranda", "📊 Dashboard Segmentasi", "📈 Evaluasi Model", "🔮 Prediksi Nasabah Baru"],
)

st.sidebar.divider()
st.sidebar.caption(
    "Proyek segmentasi nasabah kartu kredit menggunakan K-Means Clustering, "
    "dibangun mengikuti metodologi CRISP-DM. Model dilatih pada dataset "
    "*Credit Card Dataset for Clustering* (Kaggle)."
)

# =========================================================================
# HALAMAN 1: BERANDA
# =========================================================================
if page == "🏠 Beranda":
    st.title("💳 Segmentasi Nasabah Kartu Kredit")
    st.markdown(
        """
        Aplikasi ini mendemonstrasikan hasil proyek **Customer Segmentation**
        menggunakan algoritma **K-Means Clustering**, dibangun mengikuti
        metodologi **CRISP-DM** (Business Understanding → Data Understanding →
        Data Preparation → Modeling → Evaluation → Deployment).

        ### Tujuan Proyek
        Mengelompokkan nasabah kartu kredit ke dalam beberapa segmen
        berdasarkan perilaku keuangan mereka, agar perusahaan dapat memberikan
        strategi pemasaran dan kebijakan kredit yang lebih tepat sasaran.

        ### Fitur yang Digunakan untuk Clustering
        | Fitur | Deskripsi |
        |---|---|
        | `BALANCE` | Saldo yang tersisa di akun untuk melakukan pembelian |
        | `PURCHASES` | Total nilai pembelian yang dilakukan dari akun |
        | `CREDIT_LIMIT` | Batas maksimum kredit yang diberikan kepada nasabah |

        ### Cara Menggunakan Aplikasi
        Gunakan menu navigasi di sebelah kiri untuk:
        - **Dashboard Segmentasi** — melihat ringkasan dan profil tiap segmen nasabah hasil clustering.
        - **Evaluasi Model** — melihat metrik teknis yang digunakan untuk memilih jumlah cluster optimal.
        - **Prediksi Nasabah Baru** — memasukkan data nasabah baru (manual atau upload CSV) untuk mengetahui segmennya.
        """
    )

    if df_clustered is not None:
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Nasabah (data training)", f"{len(df_clustered):,}")
        col2.metric("Jumlah Segmen", df_clustered["Cluster"].nunique())
        col3.metric("Fitur Model", len(FEATURES))

# =========================================================================
# HALAMAN 2: DASHBOARD SEGMENTASI (EDA hasil cluster)
# =========================================================================
elif page == "📊 Dashboard Segmentasi":
    st.title("📊 Dashboard Hasil Segmentasi")

    if df_clustered is None:
        st.warning(
            "File `clustered_customers.csv` tidak ditemukan. Jalankan notebook "
            "terlebih dahulu agar dashboard ini bisa menampilkan data historis."
        )
    else:
        # Ringkasan jumlah & proporsi tiap cluster
        st.subheader("Ringkasan Jumlah Nasabah per Segmen")
        jumlah = df_clustered["Cluster"].value_counts().sort_index()
        proporsi = (jumlah / len(df_clustered) * 100).round(2)

        cols = st.columns(len(jumlah))
        for i, (cluster_id, count) in enumerate(jumlah.items()):
            info = CLUSTER_INFO.get(cluster_id, {})
            with cols[i]:
                st.metric(f"Cluster {cluster_id}: {info.get('label', '-')}", f"{count:,} nasabah", f"{proporsi[cluster_id]}%")

        col_a, col_b = st.columns(2)
        with col_a:
            fig, ax = plt.subplots(figsize=(5, 4))
            colors = [CLUSTER_INFO.get(i, {}).get("warna", "#999999") for i in jumlah.index]
            ax.pie(jumlah, labels=[f"Cluster {i}" for i in jumlah.index], autopct="%1.1f%%", colors=colors)
            ax.set_title("Proporsi Nasabah per Cluster")
            st.pyplot(fig)

        with col_b:
            st.write("**Profil rata-rata tiap cluster (fitur utama):**")
            profile = df_clustered.groupby("Cluster")[FEATURES].mean().round(2)
            st.dataframe(profile, use_container_width=True)

        st.divider()
        st.subheader("Visualisasi Sebaran Cluster")
        fitur_x = st.selectbox("Sumbu X", FEATURES, index=0)
        fitur_y = st.selectbox("Sumbu Y", FEATURES, index=1)

        fig2, ax2 = plt.subplots(figsize=(8, 5))
        for cluster_id in sorted(df_clustered["Cluster"].unique()):
            subset = df_clustered[df_clustered["Cluster"] == cluster_id]
            ax2.scatter(
                subset[fitur_x], subset[fitur_y], alpha=0.4, s=12,
                label=f"Cluster {cluster_id}: {CLUSTER_INFO.get(cluster_id, {}).get('label', '')}",
                color=CLUSTER_INFO.get(cluster_id, {}).get("warna", None),
            )
        ax2.set_xlabel(fitur_x)
        ax2.set_ylabel(fitur_y)
        ax2.legend()
        ax2.set_title(f"{fitur_x} vs {fitur_y} per Cluster")
        st.pyplot(fig2)

        st.divider()
        st.subheader("Detail & Rekomendasi Tiap Segmen")
        for cluster_id in sorted(df_clustered["Cluster"].unique()):
            info = CLUSTER_INFO.get(cluster_id, {})
            with st.expander(f"Cluster {cluster_id} — {info.get('label', '-')}"):
                st.write(info.get("desc", "-"))
                st.write("**Rekomendasi strategi:**")
                for r in info.get("rekomendasi", []):
                    st.write(f"- {r}")

        st.divider()
        st.subheader("Contoh Data Nasabah Ter-cluster")
        st.dataframe(df_clustered[FEATURES + ["Cluster"]].head(20), use_container_width=True)

# =========================================================================
# HALAMAN 3: EVALUASI MODEL
# =========================================================================
elif page == "📈 Evaluasi Model":
    st.title("📈 Evaluasi & Pemilihan Model")
    st.write(
        "Halaman ini menampilkan metrik yang digunakan pada tahap *Modeling* "
        "di notebook untuk menentukan jumlah cluster (k) yang optimal."
    )

    if eval_metrics is None:
        st.warning(
            "File `model_evaluation_metrics.csv` tidak ditemukan. Jalankan "
            "notebook terlebih dahulu untuk menghasilkan tabel evaluasi ini."
        )
    else:
        st.subheader("Tabel Metrik Evaluasi per Jumlah Cluster (k)")
        st.dataframe(eval_metrics.round(4), use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots(figsize=(5.5, 4))
            ax.plot(eval_metrics.index, eval_metrics["Inertia"], marker="o", color="steelblue")
            ax.set_title("Elbow Method (Inertia)")
            ax.set_xlabel("k")
            ax.set_ylabel("Inertia (WCSS)")
            st.pyplot(fig)

            fig3, ax3 = plt.subplots(figsize=(5.5, 4))
            ax3.plot(eval_metrics.index, eval_metrics["Davies-Bouldin"], marker="o", color="seagreen")
            ax3.set_title("Davies-Bouldin Index (semakin rendah semakin baik)")
            ax3.set_xlabel("k")
            st.pyplot(fig3)

        with col2:
            fig2, ax2 = plt.subplots(figsize=(5.5, 4))
            ax2.plot(eval_metrics.index, eval_metrics["Silhouette"], marker="o", color="darkorange")
            ax2.set_title("Silhouette Score (semakin tinggi semakin baik)")
            ax2.set_xlabel("k")
            st.pyplot(fig2)

            fig4, ax4 = plt.subplots(figsize=(5.5, 4))
            ax4.plot(eval_metrics.index, eval_metrics["Calinski-Harabasz"], marker="o", color="indianred")
            ax4.set_title("Calinski-Harabasz Index (semakin tinggi semakin baik)")
            ax4.set_xlabel("k")
            st.pyplot(fig4)

        st.divider()
        k_final = 3
        st.success(
            f"**Model final menggunakan k = {k_final}** — keseimbangan terbaik antara "
            "kualitas statistik (silhouette score tinggi, Davies-Bouldin rendah) "
            "dan kegunaan bisnis (jumlah segmen yang mudah dieksekusi tim marketing)."
        )

        if df_clustered is not None:
            from sklearn.metrics import silhouette_score
            X_check = df_clustered[FEATURES]
            X_scaled_check = scaler.transform(X_check)
            sil = silhouette_score(X_scaled_check, df_clustered["Cluster"])
            st.metric("Silhouette Score Model Final (dihitung ulang di aplikasi)", f"{sil:.4f}")

# =========================================================================
# HALAMAN 4: PREDIKSI NASABAH BARU
# =========================================================================
elif page == "🔮 Prediksi Nasabah Baru":
    st.title("🔮 Prediksi Segmen Nasabah Baru")
    st.write(
        "Masukkan data nasabah untuk memprediksi segmennya menggunakan model "
        "K-Means yang telah dilatih pada tahap Modeling."
    )

    tab1, tab2 = st.tabs(["🔢 Input Manual", "📄 Upload CSV (Batch)"])

    # ---------------- Tab 1: input manual satu nasabah ----------------
    with tab1:
        st.subheader("Masukkan data satu nasabah")

        col1, col2, col3 = st.columns(3)
        with col1:
            balance = st.number_input("BALANCE (saldo)", min_value=0.0, value=1000.0, step=100.0)
        with col2:
            purchases = st.number_input("PURCHASES (total pembelian)", min_value=0.0, value=500.0, step=100.0)
        with col3:
            credit_limit = st.number_input("CREDIT_LIMIT (limit kredit)", min_value=0.0, value=3000.0, step=100.0)

        if st.button("Prediksi Cluster", type="primary"):
            X_new = pd.DataFrame([[balance, purchases, credit_limit]], columns=FEATURES)
            X_scaled_new = scaler.transform(X_new)
            cluster = int(model.predict(X_scaled_new)[0])
            distances = model.transform(X_scaled_new)[0]

            info = CLUSTER_INFO.get(cluster, {})
            st.success(f"Nasabah ini termasuk **Cluster {cluster}: {info.get('label', '-')}**")
            st.write(f"**Karakteristik:** {info.get('desc', '-')}")
            st.write("**Rekomendasi strategi:**")
            for r in info.get("rekomendasi", []):
                st.write(f"- {r}")

            with st.expander("Detail teknis prediksi"):
                dist_df = pd.DataFrame(
                    {"Cluster": range(len(distances)), "Jarak ke centroid": distances.round(3)}
                ).set_index("Cluster")
                st.write("Jarak titik data ke setiap centroid (semakin kecil = semakin mirip):")
                st.dataframe(dist_df)

    # ---------------- Tab 2: upload CSV banyak nasabah ----------------
    with tab2:
        st.subheader("Upload file CSV berisi banyak nasabah")
        st.caption(f"File harus memiliki kolom: {', '.join(FEATURES)}")

        uploaded = st.file_uploader("Pilih file CSV", type=["csv"])
        if uploaded is not None:
            data = pd.read_csv(uploaded)
            missing_cols = [c for c in FEATURES if c not in data.columns]
            if missing_cols:
                st.error(f"Kolom berikut tidak ditemukan pada file: {missing_cols}")
            else:
                X_new = data[FEATURES].fillna(data[FEATURES].median())
                X_scaled_new = scaler.transform(X_new)
                data["Cluster"] = model.predict(X_scaled_new)
                data["Segmen"] = data["Cluster"].map(lambda c: CLUSTER_INFO.get(c, {}).get("label", "-"))

                st.write("Hasil prediksi:")
                st.dataframe(data, use_container_width=True)

                st.write("Distribusi jumlah nasabah per cluster:")
                st.bar_chart(data["Cluster"].value_counts().sort_index())

                csv_out = data.to_csv(index=False).encode("utf-8")
                st.download_button("Unduh hasil sebagai CSV", csv_out, "hasil_segmentasi.csv", "text/csv")

st.sidebar.divider()
st.sidebar.caption(
    "Model: K-Means (k=3) dilatih pada fitur BALANCE, PURCHASES, dan CREDIT_LIMIT "
    "yang telah distandardisasi. Lihat notebook `Credit_Card_Clustering_CRISP-DM.ipynb` "
    "untuk proses lengkapnya."
)
