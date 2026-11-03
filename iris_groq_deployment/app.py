import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from pathlib import Path

from groq import Groq

# Mengambil lokasi folder app.py
BASE_DIR = Path(__file__).resolve().parent

# Mengambil lokasi model
MODEL_PATH = BASE_DIR / "iris_model.joblib"

# Memeriksa keberadaan model
if not MODEL_PATH.exists():
    st.error("File iris_model.joblib tidak ditemukan!")
    st.write("Lokasi yang dicari:", str(MODEL_PATH))
    st.stop()

# Memuat model
bundle = joblib.load(MODEL_PATH)

model = bundle["model"]
encoder = bundle["encoder"]
features = bundle["features"]
