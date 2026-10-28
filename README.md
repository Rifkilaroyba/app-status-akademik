# Prediksi Status Akademik dengan DNN

Aplikasi Streamlit Modul 1B untuk memakai pipeline hasil Modul 1A. Aplikasi memuat model tersimpan, menerima enam fitur mahasiswa, lalu menghasilkan label Aman, Waspada, atau Berisiko beserta probabilitas setiap kelas. Aplikasi tidak melakukan training.

## Hasil evaluasi model

| Ukuran | Hasil Modul 1A |
| --- | ---: |
| Jumlah data test | 200 |
| Accuracy | 84,00% |
| F1 macro | 0,8269 |
| F1 baseline kelas terbanyak | 0,1800 |
| Status evaluasi | Lolos target F1 macro minimal 0,80 dan melebihi baseline |

Model memakai dataset sintetis praktikum. Probabilitas adalah keluaran model, bukan jaminan kebenaran label.

## Isi project

```text
app-status-akademik/
├── app.py
├── input_validation.py
├── model_status_akademik.pkl
├── model_info.json
├── requirements.txt
├── contoh_data_baru.csv
├── .python-version
├── .gitignore
└── README.md
```

`model_status_akademik.pkl` dan `model_info.json` disalin utuh dari hasil Modul 1A. Aplikasi membaca keduanya dari folder file aplikasi, sehingga dapat dijalankan dari direktori lain. Hanya artefak model yang disertakan dalam project ini yang dimuat; aplikasi tidak menyediakan unggahan model pickle.

## Menjalankan di komputer lokal

Gunakan Python 3.13 agar versi minor Python sesuai lingkungan training 3.13.16. Jalankan terminal di dalam folder `app-status-akademik`:

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Pada Windows, aktivasi virtualenv menggunakan `.venv\Scripts\activate`. Buka alamat lokal yang ditampilkan terminal, biasanya `http://localhost:8501`.

`requirements.txt` mengunci scikit-learn pada **1.9.1**, sesuai metadata training. Aplikasi akan menampilkan pesan dan berhenti jika versi scikit-learn tidak cocok atau metadata belum memenuhi gerbang evaluasi. Jangan mengganti versi atau mengubah model untuk menghilangkan pesan ini; gunakan lingkungan yang sesuai atau evaluasi ulang model di Modul 1A.

## Uji prediksi satu mahasiswa

1. Buka tab **Prediksi Satu Mahasiswa**.
2. Isi IPK **3,45**, kehadiran **92**, jam belajar **12**, tugas terkumpul **95**, nilai UTS **82**, dan aktif organisasi **Ya**.
3. Klik **Prediksi**. Model hasil Modul 1A menghasilkan **Aman**, dengan probabilitas sekitar **0,995285**.
4. Coba IPK **0,50**. Nilai ini masih diterima form, tetapi berada di luar rentang IPK data latih 2,00–4,00, sehingga aplikasi menampilkan peringatan.

Urutan probabilitas mengikuti `model.classes_`, sehingga setiap angka dipasangkan dengan nama kelas yang benar.

## Uji prediksi CSV

1. Buka tab **Prediksi dari File CSV**.
2. Unduh atau gunakan `contoh_data_baru.csv`, yang berisi 10 baris. File ini sama dengan contoh pada aplikasi Modul 2B.
3. Unggah file, lalu klik **Prediksi semua**.
4. Periksa tabel dan jumlah setiap label. Kolom tambahan seperti `nama` tetap disertakan.
5. Klik **Unduh hasil prediksi**. File berisi seluruh kolom asli dan kolom `prediksi_status`. Jika kolom itu sudah ada dalam input, hasil baru memakai `prediksi_status_2` dan seterusnya agar data asli tetap tersimpan.

Kolom wajib:

```text
ipk,kehadiran,jam_belajar,tugas_terkumpul,nilai_uts,aktif_organisasi
```

CSV harus UTF-8, maksimal 5 MB dan 5.000 baris. Lima kolom numerik harus berisi angka hingga dengan rentang IPK 0–4, kehadiran/tugas/UTS 0–100, dan jam belajar 0–40. `aktif_organisasi` harus Ya atau Tidak. Nama kolom duplikat, nilai kosong, angka tidak hingga, kategori salah, dan baris tanpa data ditolak dengan pesan yang menjelaskan masalahnya. Nilai yang sah tetapi berada di luar rentang training mendapat peringatan; aplikasi tetap dapat memprediksinya.

## Menyiapkan repository GitHub

Buat repository bernama `app-status-akademik` pada akun GitHub. Untuk alur modul, pilih repository publik. Unggah seluruh file project di atas ke root repository. Pastikan `app.py`, model, metadata, dan helper `input_validation.py` berada pada tingkat folder yang sama. Jangan ikut mengunggah folder `.venv` atau `__pycache__`.

Jika memakai Git, ganti `USERNAME_GITHUB` dengan nama akun Anda sebelum menjalankan perintah:

```bash
git init
git add .
git status
git commit -m "Deploy aplikasi DNN status akademik Modul 1B"
git branch -M main
git remote add origin https://github.com/USERNAME_GITHUB/app-status-akademik.git
git push -u origin main
```

## Deployment ke Streamlit Community Cloud

1. Login ke [Streamlit Community Cloud](https://share.streamlit.io) menggunakan akun GitHub yang memiliki repository.
2. Klik **Create app**, lalu pilih aplikasi yang sudah ada di GitHub jika diminta.
3. Pilih repository `USERNAME_GITHUB/app-status-akademik`, branch `main`, dan main file path `app.py`.
4. Buka **Advanced settings** dan pilih **Python 3.13**. Pilihan Python di dashboard perlu diatur secara eksplisit; file `.python-version` juga mencatat versi minor yang dituju untuk penggunaan lokal.
5. Simpan pengaturan, lalu klik **Deploy**. Aplikasi DNN ini tidak memerlukan Groq API key.
6. Setelah aplikasi terbuka, ulangi uji satu mahasiswa dan CSV di URL `.streamlit.app` yang benar-benar diberikan dashboard.

Jika aplikasi berada dalam subfolder repository, sesuaikan main file path, misalnya `app-status-akademik/app.py`. Jangan menganggap aplikasi sudah terdeploy hanya karena file project sudah diunggah; periksa status deployment dan buka URL hasilnya.

Simpan bukti praktikum: URL repository, URL aplikasi publik, tangkapan layar form dan probabilitas, hasil 10 baris CSV, serta file CSV unduhan. Mengganti model perlu dilakukan melalui evaluasi ulang Modul 1A terlebih dahulu, kemudian model dan metadata baru diunggah bersama.

## Referensi

- Modul 1B Deploy Model DNN dengan Streamlit, materi praktikum dosen.
- [Perintah menjalankan Streamlit](https://docs.streamlit.io/develop/api-reference/cli/run).
- [Deployment dan pengaturan Python di Community Cloud](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy).
