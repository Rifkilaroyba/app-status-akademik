"""Validasi enam fitur akademik tanpa melakukan training atau prediksi."""
import csv
import io

import numpy as np
import pandas as pd

FEATURES = ["ipk", "kehadiran", "jam_belajar", "tugas_terkumpul", "nilai_uts", "aktif_organisasi"]
NUMERIC_BOUNDS = {
    "ipk": (0.0, 4.0),
    "kehadiran": (0.0, 100.0),
    "jam_belajar": (0.0, 40.0),
    "tugas_terkumpul": (0.0, 100.0),
    "nilai_uts": (0.0, 100.0),
}
MAX_CSV_BYTES = 5 * 1024 * 1024
MAX_CSV_ROWS = 5000


def _rows(mask):
    positions = [str(i + 2) for i, bad in enumerate(mask) if bad]
    return ", ".join(positions[:10]) + (", ..." if len(positions) > 10 else "")


def validate_features(frame):
    """Return fitur ternormalisasi; kolom tambahan tidak ikut ke model."""
    if frame.empty:
        raise ValueError("CSV tidak memiliki baris data.")
    if frame.columns.duplicated().any():
        raise ValueError("Nama kolom CSV harus unik.")
    missing = [name for name in FEATURES if name not in frame.columns]
    if missing:
        raise ValueError("Kolom berikut tidak ditemukan: " + ", ".join(missing))
    normalized = frame[FEATURES].copy()
    errors = []
    for name, (lower, upper) in NUMERIC_BOUNDS.items():
        values = pd.to_numeric(normalized[name], errors="coerce")
        bad = values.isna() | ~np.isfinite(values) | (values < lower) | (values > upper)
        if pd.api.types.is_bool_dtype(normalized[name]):
            bad = pd.Series(True, index=normalized.index)
        if bad.any():
            errors.append(f"{name} harus angka {lower:g}–{upper:g}; periksa baris {_rows(bad)}.")
        normalized[name] = values
    categories = normalized["aktif_organisasi"].astype("string").str.strip().str.casefold()
    bad_categories = ~categories.isin(["ya", "tidak"])
    if bad_categories.any():
        errors.append("aktif_organisasi harus Ya atau Tidak; periksa baris " + _rows(bad_categories) + ".")
    normalized["aktif_organisasi"] = categories.map({"ya": "Ya", "tidak": "Tidak"})
    if errors:
        raise ValueError("\n".join(errors))
    return normalized


def read_csv_bytes(content):
    """UTF-8 CSV, reject duplicated headings before pandas can rename them."""
    if not content or len(content) > MAX_CSV_BYTES:
        raise ValueError("Pilih CSV berisi data dengan ukuran maksimal 5 MB.")
    try:
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError("CSV harus menggunakan encoding UTF-8.") from exc
    try:
        rows = list(csv.reader(io.StringIO(text), strict=True))
    except csv.Error as exc:
        raise ValueError("Format CSV tidak dapat dibaca: " + str(exc)) from exc
    if not rows:
        raise ValueError("CSV kosong.")
    headings = [item.strip() for item in rows[0]]
    if not all(headings) or len(set(headings)) != len(headings):
        raise ValueError("Nama kolom CSV tidak boleh kosong atau duplikat.")
    data_rows = [row for row in rows[1:] if row]
    if len(data_rows) > MAX_CSV_ROWS:
        raise ValueError("CSV maksimal 5.000 baris. Pecah file menjadi beberapa bagian.")
    if any(len(row) != len(headings) for row in data_rows):
        raise ValueError("Jumlah nilai pada setiap baris harus sama dengan jumlah kolom.")
    frame = pd.DataFrame(data_rows, columns=headings)
    features = validate_features(frame)
    return frame, features


def outside_training_range(features, ranges):
    return {
        name: int(((features[name] < bounds[0]) | (features[name] > bounds[1])).sum())
        for name, bounds in ranges.items()
        if name in features and ((features[name] < bounds[0]) | (features[name] > bounds[1])).any()
    }


def attach_predictions(frame, predictions):
    """Keep every original field, including an existing prediction column."""
    output = frame.copy()
    name = "prediksi_status"
    suffix = 2
    while name in output.columns:
        name = f"prediksi_status_{suffix}"
        suffix += 1
    output[name] = list(predictions)
    return output, name
