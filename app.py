import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Klasifikasi Bunga Iris",
    page_icon="🌸",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent

st.title("Klasifikasi Spesies Bunga Iris")
st.write(
    "Klasifikasi spesies bunga Iris berdasarkan empat ukuran "
    "kelopak dan mahkota menggunakan model Deep Neural Network."
)

@st.cache_resource
def load_model():
    model = joblib.load(BASE_DIR / "model_iris_dnn.pkl")
    label_encoder = joblib.load(BASE_DIR / "label_encoder.pkl")
    with open(BASE_DIR / "model_info.json", "r", encoding="utf-8") as f:
        metadata = json.load(f)
    return model, label_encoder, metadata

try:
    model, label_encoder, metadata = load_model()
except Exception as e:
    st.error(f"Gagal memuat model: {e}")
    st.stop()

fitur = metadata["fitur_numerik"]
rentang = metadata["rentang_fitur"]

st.sidebar.header("Informasi Model")
st.sidebar.write(metadata["nama_model"])
st.sidebar.metric("Accuracy", f"{metadata['metrik']['accuracy'] * 100:.2f}%")
st.sidebar.metric("F1 Macro", f"{metadata['metrik']['f1_macro']:.4f}")
st.sidebar.write("Kelas bunga:", ", ".join(metadata["kelas"]))

def prediksi(data):
    data = data[fitur].copy()
    prediksi_encoded = model.predict(data)
    hasil_prediksi = label_encoder.inverse_transform(
        prediksi_encoded.astype(int)
    )

    probabilitas = model.predict_proba(data)
    nama_kelas = label_encoder.inverse_transform(
        model.named_steps["dnn"].classes_.astype(int)
    )

    hasil = data.copy()
    hasil["prediksi_spesies"] = hasil_prediksi
    hasil["keyakinan"] = probabilitas.max(axis=1) * 100

    probabilitas_df = pd.DataFrame(
        probabilitas,
        columns=nama_kelas,
        index=data.index
    )
    return hasil, probabilitas_df

tab1, tab2 = st.tabs(["Prediksi Manual", "Prediksi CSV"])

with tab1:
    st.subheader("Input Ukuran Bunga")
    with st.form("form_prediksi"):
        input_data = {}
        col1, col2 = st.columns(2)

        for i, nama in enumerate(fitur):
            kolom = col1 if i % 2 == 0 else col2
            batas_min, batas_max = rentang[nama]
            with kolom:
                input_data[nama] = st.number_input(
                    nama.replace("_", " ").title() + " (cm)",
                    min_value=float(batas_min),
                    max_value=float(batas_max),
                    value=float((batas_min + batas_max) / 2),
                    step=0.1,
                    format="%.1f"
                )

        tombol = st.form_submit_button(
            "Prediksi Spesies Bunga",
            type="primary",
            use_container_width=True
        )

    if tombol:
        data_input = pd.DataFrame([input_data], columns=fitur)
        hasil, probabilitas_df = prediksi(data_input)
        st.success(f"Hasil prediksi: {hasil['prediksi_spesies'].iloc[0]}")
        st.metric("Keyakinan Model", f"{hasil['keyakinan'].iloc[0]:.2f}%")
        st.markdown("#### Probabilitas Setiap Kelas")
        tampil = probabilitas_df.iloc[0].sort_values(ascending=False) * 100
        st.bar_chart(tampil)

with tab2:
    st.subheader("Prediksi Menggunakan CSV")
    st.write("CSV harus memiliki empat kolom fitur sesuai dataset Iris.")

    st.download_button(
        "Unduh Template CSV",
        data=pd.DataFrame(columns=fitur).to_csv(index=False),
        file_name="template_input_iris.csv",
        mime="text/csv"
    )

    file_csv = st.file_uploader("Unggah file CSV", type=["csv"])
    if file_csv is not None:
        try:
            df_input = pd.read_csv(file_csv)
            kolom_hilang = [kolom for kolom in fitur if kolom not in df_input.columns]

            if kolom_hilang:
                st.error("Kolom belum tersedia: " + ", ".join(kolom_hilang))
            elif df_input.empty:
                st.warning("File CSV tidak memiliki data.")
            else:
                st.dataframe(df_input.head())
                if st.button("Prediksi Semua Data"):
                    data_prediksi = df_input[fitur].copy()
                    for nama in fitur:
                        data_prediksi[nama] = pd.to_numeric(
                            data_prediksi[nama], errors="coerce"
                        )

                    if data_prediksi.isnull().any().any():
                        st.error("Terdapat nilai kosong atau tidak valid pada fitur.")
                    else:
                        hasil, _ = prediksi(data_prediksi)
                        hasil_akhir = df_input.copy()
                        hasil_akhir["prediksi_spesies"] = hasil["prediksi_spesies"]
                        hasil_akhir["keyakinan"] = hasil["keyakinan"]

                        st.success(f"Berhasil memprediksi {len(hasil_akhir)} data.")
                        st.dataframe(hasil_akhir)
                        st.download_button(
                            "Unduh Hasil Prediksi",
                            data=hasil_akhir.to_csv(index=False).encode("utf-8"),
                            file_name="hasil_prediksi_iris.csv",
                            mime="text/csv"
                        )
        except Exception as e:
            st.error(f"Terjadi kesalahan: {e}")
