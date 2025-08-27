# Diabetes Screening Prediction — Streamlit App
Aplikasi skrining risiko diabetes berbasis Pima Indians Diabetes dengan fokus Recall tinggi (mengurangi false negative) dan threshold tuning yang transparan. Proyek ini lengkap end-to-end: pembersihan data → pelatihan model → evaluasi (PR/ROC, kalibrasi) → deploy sebagai aplikasi Streamlit.

## 🎯 Tujuan & Kegunaan
Skrining dini: menandai individu berisiko untuk pemeriksaan lanjutan.
1. Pendukung keputusan: memilih threshold sesuai kapasitas rujukan (trade-off recall vs precision) dengan visualisasi yang jelas.
2. Edukasi data: memahami faktor yang paling berpengaruh terhadap risiko (feature importance/koefisien).
3. Demo portofolio: contoh praktik ML yang dapat dipakai (notebook + app + artefak siap deploy).

## 💎 Apa yang Menjadi Daya Tarik Proyek Ini?
- Recall-first design untuk use-case skrining, bukan sekadar akurasi.
- Threshold tuning interaktif di validation + simulasi operasional (TP/FP/FN) sehingga keputusan ambang berbasis data.
- Kalibrasi probabilitas (Calibration Plot + Brier Score) agar angka peluang “jujur” saat dikomunikasikan.
- Explainability (feature importance / opsional SHAP) untuk menjelaskan “mengapa” model memberi skor tertentu.
- Auto-loading artefak: diabetes.csv, artifacts/pima_pipeline.joblib, artifacts/threshold.json → app langsung jalan; tersedia fallback upload.
- Repo siap deploy (Streamlit Cloud / Docker) + notebook template untuk replikasi dan eksperimen.

## 🧰 Tech Stack
Python, Streamlit, scikit-learn, Pandas, NumPy, Plotly, (opsional) SHAP

## ⚖️ Disclaimer
Aplikasi ini adalah alat bantu skrining, bukan alat diagnosis. Keputusan klinis tetap harus melalui tenaga kesehatan.

## 🚀 Cara Menjalankan (Lokal):
Pada CMD ketik di lokasi tempat kamu menyimpan:
python -m venv .venv 
pip install -r requirements.txt
streamlit run streamlit_app.py

Catatan versi yang stabil (disarankan untuk Streamlit Cloud / lokal):
- Python 3.11 (runtime.txt berisi 3.11)
- numpy==1.26.4, scikit-learn==1.4.2, shap==0.45.1
(Paket lain mengikuti requirements.txt di repo.)

## 🙏 Acknowledgements
Dataset: Pima Indians Diabetes (Kaggle).
Terima kasih untuk komunitas open-source: Streamlit, scikit-learn, Pandas, NumPy, Plotly.
