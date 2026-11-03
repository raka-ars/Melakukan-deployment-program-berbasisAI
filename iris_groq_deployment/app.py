
import streamlit as st
import pandas as pd
import joblib
import json

from groq import Groq

# Konfigurasi halaman
st.set_page_config(
    page_title="Iris Classification AI",
    page_icon="🌸",
    layout="wide"
)

# Load model
bundle = joblib.load("iris_model.joblib")

model = bundle["model"]
encoder = bundle["encoder"]
features = bundle["features"]

MODEL_NAME = "openai/gpt-oss-120b"

# Header
st.title("🌸 Iris Flower Classification")
st.write(
    "Machine Learning and AI Model Service "
    "for Iris Flower Classification."
)

st.divider()

# Pilihan model
selected_model = st.selectbox(
    "Pilih Model Prediksi",
    [
        "Random Forest",
        "Groq GPT-OSS 120B"
    ]
)

# Input data
st.subheader("Input Karakteristik Bunga Iris")

col1, col2 = st.columns(2)

with col1:

    sepal_length = st.number_input(
        "Sepal Length (cm)",
        min_value=0.1,
        max_value=10.0,
        value=5.1
    )

    sepal_width = st.number_input(
        "Sepal Width (cm)",
        min_value=0.1,
        max_value=10.0,
        value=3.5
    )

with col2:

    petal_length = st.number_input(
        "Petal Length (cm)",
        min_value=0.1,
        max_value=10.0,
        value=1.4
    )

    petal_width = st.number_input(
        "Petal Width (cm)",
        min_value=0.1,
        max_value=10.0,
        value=0.2
    )

# Prediksi
if st.button(
    "🔍 Prediksi Jenis Bunga",
    type="primary"
):

    input_data = pd.DataFrame(
        [[
            sepal_length,
            sepal_width,
            petal_length,
            petal_width
        ]],
        columns=features
    )

    try:

        if selected_model == "Random Forest":

            prediction = model.predict(
                input_data
            )[0]

            result = encoder.inverse_transform(
                [prediction]
            )[0]

            probabilities = model.predict_proba(
                input_data
            )[0]

            confidence = max(probabilities) * 100

            st.success(
                f"Prediksi: {result.title()}"
            )

            st.metric(
                "Model Confidence",
                f"{confidence:.2f}%"
            )

        else:

            # Mengambil API Key
            api_key = st.secrets["GROQ_API_KEY"]

            client = Groq(
                api_key=api_key
            )

            prompt = f"""
            Classify this Iris flower.

            Sepal Length: {sepal_length}
            Sepal Width: {sepal_width}
            Petal Length: {petal_length}
            Petal Width: {petal_width}

            Choose one class:
            setosa, versicolor, virginica.

            Return JSON only:
            {{"prediction": "setosa"}}
            """

            response = client.chat.completions.create(
                model=MODEL_NAME,

                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an accurate "
                            "Iris flower classifier. "
                            "Return valid JSON only."
                        )
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],

                response_format={
                    "type": "json_object"
                },

                reasoning_effort="low"
            )

            result_json = json.loads(
                response.choices[0].message.content
            )

            result = result_json["prediction"].lower()

            if result not in encoder.classes_:
                raise ValueError(
                    "Hasil prediksi tidak valid."
                )

            st.success(
                f"Prediksi: {result.title()}"
            )

            st.info(
                "Prediksi dihasilkan oleh "
                "Groq GPT-OSS 120B."
            )

    except Exception as e:

        st.error(
            f"Terjadi kesalahan: {str(e)}"
        )

st.divider()

# Download dataset
st.subheader("Download Dataset Sampel")

try:

    with open(
        "sample_iris_input.csv",
        "rb"
    ) as file:

        st.download_button(
            label="⬇ Download Sample CSV",
            data=file,
            file_name="sample_iris_input.csv",
            mime="text/csv"
        )

except FileNotFoundError:

    st.warning("File sampel belum tersedia.")

st.caption(
    "Iris Classification System | "
    "Random Forest & Groq GPT-OSS 120B"
)
