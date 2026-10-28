"""Modul 1B: inferensi Streamlit dari pipeline DNN hasil Modul 1A."""
import hashlib
import json
from pathlib import Path

import joblib
import pandas as pd
import sklearn
import streamlit as st

from input_validation import (
    FEATURES,
    attach_predictions,
    outside_training_range,
    read_csv_bytes,
    validate_features,
)

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title="Status Akademik DNN", page_icon="🎓", layout="centered")


@st.cache_data
def load_info():
    return json.loads((ROOT / "model_info.json").read_text(encoding="utf-8"))


@st.cache_resource
def load_model():
    # Artefak tepercaya ini disalin dari hasil Modul 1A; aplikasi tidak menerima unggahan pickle.
    return joblib.load(ROOT / "model_status_akademik.pkl")


try:
    info = load_info()
    metrics = info["metrik"]
    expected_version = info["versi"]["scikit-learn"]
    if sklearn.__version__ != expected_version:
        raise ValueError(
            f"Versi scikit-learn harus {expected_version}, sedangkan runtime memakai {sklearn.__version__}. "
            "Pasang requirements.txt dan mulai ulang aplikasi."
        )
    if not (
        info.get("lolos_gerbang_modul_1a") is True
        and metrics["f1_macro"] >= 0.80
        and metrics["f1_macro"] > metrics["f1_baseline"]
    ):
        raise ValueError("Model belum memenuhi gerbang evaluasi Modul 1A. Evaluasi ulang sebelum deployment.")
    if info["fitur_numerik"] + info["fitur_kategori"] != FEATURES:
        raise ValueError("Skema fitur metadata tidak sesuai dengan aplikasi.")
    model = load_model()
except Exception as exc:
    st.error(f"Aplikasi belum dapat memuat model: {exc}")
    st.stop()

with st.sidebar:
    st.header("Tentang Model")
    st.write(info["nama_model"])
    st.metric("Accuracy data test", f"{metrics['accuracy']:.1%}")
    st.metric("F1 macro data test", f"{metrics['f1_macro']:.4f}")
    st.caption(f"{metrics['n_test']} data test · scikit-learn {expected_version}")
    st.success("Model lolos evaluasi Modul 1A")
    st.caption("Prediksi memakai model tersimpan. Tidak ada training di aplikasi ini.")

st.title("🎓 Prediksi Status Akademik Mahasiswa")
st.write("Masukkan data mahasiswa untuk memprediksi status **Aman**, **Waspada**, atau **Berisiko**.")
st.caption("Modul 1B · DNN · Model dilatih pada dataset sintetis praktikum.")


def show_range_warning(features):
    outside = outside_training_range(features, info["rentang_fitur"])
    if outside:
        names = ", ".join(f"{key} ({count} baris)" for key, count in outside.items())
        st.warning("Input di luar rentang data latih: " + names + ". Prediksi dapat kurang akurat.")


def show_label(label):
    display = {"Aman": st.success, "Waspada": st.warning, "Berisiko": st.error}
    display.get(label, st.info)(f"Prediksi status: **{label}**")


tab_single, tab_csv = st.tabs(["Prediksi Satu Mahasiswa", "Prediksi dari File CSV"])
with tab_single:
    with st.form("form_prediksi"):
        first, second = st.columns(2)
        with first:
            ipk = st.number_input("IPK", min_value=0.0, max_value=4.0, value=3.0, step=0.01)
            kehadiran = st.slider("Kehadiran (%)", 0, 100, 80)
            jam_belajar = st.slider("Jam belajar mandiri per minggu", 0, 40, 8)
        with second:
            tugas = st.slider("Tugas terkumpul (%)", 0, 100, 80)
            uts = st.slider("Nilai UTS", 0, 100, 70)
            organisasi = st.radio("Aktif organisasi?", ["Ya", "Tidak"], horizontal=True)
        submitted = st.form_submit_button("Prediksi", type="primary")
    if submitted:
        frame = pd.DataFrame([{
            "ipk": ipk, "kehadiran": kehadiran, "jam_belajar": jam_belajar,
            "tugas_terkumpul": tugas, "nilai_uts": uts, "aktif_organisasi": organisasi,
        }])
        features = validate_features(frame)
        show_range_warning(features)
        prediction = str(model.predict(features)[0])
        probabilities = pd.Series(model.predict_proba(features)[0], index=model.classes_, name="Probabilitas")
        st.session_state["dnn_single"] = {"prediction": prediction, "probabilities": probabilities}
    if "dnn_single" in st.session_state:
        result = st.session_state["dnn_single"]
        show_label(result["prediction"])
        st.bar_chart(result["probabilities"])
        st.dataframe(
            result["probabilities"].rename_axis("Kelas").reset_index(),
            hide_index=True,
            column_config={"Probabilitas": st.column_config.NumberColumn(format="%.6f")},
        )
        st.caption("Probabilitas merupakan keluaran model, bukan jaminan bahwa label pasti benar.")

with tab_csv:
    st.write("Unggah CSV UTF-8 dengan kolom: " + ", ".join(f"`{name}`" for name in FEATURES) + ".")
    st.caption("Maksimal 5 MB dan 5.000 baris. Kolom tambahan seperti nama tetap disertakan dalam unduhan.")
    example = ROOT / "contoh_data_baru.csv"
    if example.exists():
        st.download_button("Unduh contoh CSV", example.read_bytes(), "contoh_data_baru.csv", "text/csv")
    uploaded = st.file_uploader("Pilih file CSV", type=["csv"])
    if uploaded is not None:
        content = uploaded.getvalue()
        file_hash = hashlib.sha256(content).hexdigest()
        if st.session_state.get("dnn_csv_hash") != file_hash:
            st.session_state.pop("dnn_csv_result", None)
        try:
            original, features = read_csv_bytes(content)
        except ValueError as exc:
            st.error(str(exc))
        else:
            st.write(f"{len(features)} baris siap diprediksi.")
            show_range_warning(features)
            if st.button("Prediksi semua", type="primary"):
                with st.spinner("Memproses data dengan DNN..."):
                    output, prediction_column = attach_predictions(original, model.predict(features))
                st.session_state["dnn_csv_hash"] = file_hash
                st.session_state["dnn_csv_result"] = (output, prediction_column)
            if "dnn_csv_result" in st.session_state and st.session_state.get("dnn_csv_hash") == file_hash:
                output, prediction_column = st.session_state["dnn_csv_result"]
                st.dataframe(output, hide_index=True)
                st.write(output[prediction_column].value_counts().rename_axis("Status").rename("Jumlah"))
                st.download_button(
                    "Unduh hasil prediksi", output.to_csv(index=False).encode("utf-8-sig"),
                    file_name="hasil_prediksi_dnn.csv", mime="text/csv",
                )
