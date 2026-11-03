import json
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Klasifikasi Spesies Penguin (DNN)", page_icon="🐧")

# ---------- 1. Muat model & informasi model (sekali saja) ----------
@st.cache_resource
def load_model():
    return joblib.load("model_penguin.pkl")

@st.cache_data
def load_info():
    with open("model_info.json") as f:
        return json.load(f)

model = load_model()
info = load_info()
FITUR = info["fitur_numerik"] + info["fitur_kategori"]
rentang = info["rentang_fitur"]

# ---------- 2. Sidebar: informasi model ----------
with st.sidebar:
    st.header("Tentang Model")
    st.write(info["nama_model"])
    st.metric("Accuracy (data uji)", f"{info['metrik']['accuracy']:.1%}")
    st.metric("F1 macro (data uji)", f"{info['metrik']['f1_macro']:.3f}")
    st.caption(f"Dievaluasi pada {info['metrik']['n_test']} data uji · "
               f"scikit-learn {info['versi']['scikit-learn']}")
    st.caption("Dataset: Palmer Penguins (Kaggle) — 333 data, 3 spesies.")

st.title("🐧 Klasifikasi Spesies Penguin")
st.write("Masukkan ukuran tubuh penguin untuk memprediksi spesiesnya: "
         "**Adelie**, **Chinstrap**, atau **Gentoo**.")

tab1, tab2 = st.tabs(["Prediksi Satu Penguin", "Prediksi dari File CSV"])

# ---------- 3. Prediksi satu data ----------
with tab1:
    with st.form("form_prediksi"):
        col1, col2 = st.columns(2)
        with col1:
            panjang_paruh = st.number_input("Panjang paruh (mm)", 30.0, 65.0, 44.0, 0.1)
            kedalaman_paruh = st.number_input("Kedalaman paruh (mm)", 12.0, 25.0, 17.0, 0.1)
            sirip = st.slider("Panjang sirip (mm)", 165, 240, 200)
        with col2:
            massa = st.slider("Massa tubuh (gram)", 2500, 6500, 4200, 50)
            pulau = st.selectbox("Pulau asal", ["Biscoe", "Dream", "Torgersen"])
            kelamin = st.radio("Jenis kelamin", ["Jantan", "Betina"], horizontal=True)
        kirim = st.form_submit_button("Prediksi")

    if kirim:
        data = pd.DataFrame([{
            "bill_length_mm": panjang_paruh, "bill_depth_mm": kedalaman_paruh,
            "flipper_length_mm": sirip, "body_mass_g": massa,
            "island": pulau, "sex": kelamin,
        }])[FITUR]

        luar = [c for c in rentang if not rentang[c][0] <= data[c].iloc[0] <= rentang[c][1]]
        if luar:
            st.warning(f"Nilai {', '.join(luar)} di luar rentang data latih; "
                       "hasil prediksi bisa kurang akurat.")

        hasil = model.predict(data)[0]
        proba = pd.Series(model.predict_proba(data)[0], index=model.classes_)
        st.success(f"Prediksi spesies: **{hasil}** (keyakinan {proba[hasil]:.0%})")
        st.bar_chart(proba.rename("probabilitas"))

# ---------- 4. Prediksi banyak data dari CSV ----------
with tab2:
    st.write("Unggah file CSV dengan kolom: " + ", ".join(f"`{c}`" for c in FITUR))
    with open("contoh_data_baru.csv", "rb") as f:
        st.download_button("⬇️ Unduh contoh data (15 penguin)", f,
                           file_name="contoh_data_baru.csv", mime="text/csv")
    file = st.file_uploader("Pilih file CSV", type="csv")
    if file is not None:
        df = pd.read_csv(file)
        kurang = [c for c in FITUR if c not in df.columns]
        if kurang:
            st.error(f"Kolom berikut tidak ditemukan: {', '.join(kurang)}")
        else:
            df["prediksi_spesies"] = model.predict(df[FITUR])
            st.dataframe(df)
            st.write(df["prediksi_spesies"].value_counts())
            if "spesies_asli" in df.columns:
                benar = (df["prediksi_spesies"] == df["spesies_asli"]).mean()
                st.info(f"Kecocokan dengan kolom `spesies_asli`: {benar:.0%}")
            st.download_button("Unduh hasil prediksi", df.to_csv(index=False),
                               file_name="hasil_prediksi.csv", mime="text/csv")
