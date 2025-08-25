# streamlit_app.py
import os
import json
from pathlib import Path
import numpy as np
import pandas as pd

import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix, precision_score, recall_score, f1_score, accuracy_score,
    roc_auc_score, average_precision_score, precision_recall_curve, roc_curve,
    brier_score_loss
)
from sklearn.calibration import calibration_curve
import joblib

# Plotting
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px

# Optional SHAP
try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False

st.set_page_config(
    page_title="Pima Diabetes — ML Dashboard",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===================== Constants =====================
ZERO_IS_MISSING = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
TARGET_COL = "Outcome"
DEFAULT_THRESHOLD = 0.5
RANDOM_STATE = 42

# ===================== sklearn pickle compatibility shim =====================
def safe_joblib_load(path_or_buffer):
    """
    Load joblib dengan perbaikan kompatibilitas untuk pickle yang dibuat
    di versi scikit-learn berbeda (mis. '_RemainderColsList').
    """
    try:
        return joblib.load(path_or_buffer)
    except AttributeError as e:
        msg = str(e)
        if "_RemainderColsList" in msg:
            import sklearn.compose._column_transformer as _ct
            class _RemainderColsList(list):
                """Compat shim for old scikit-learn pickles."""
                pass
            _ct._RemainderColsList = _RemainderColsList
            return joblib.load(path_or_buffer)
        raise

# ===================== Helpers =====================
@st.cache_data(show_spinner=False)
def load_csv(file) -> pd.DataFrame:
    return pd.read_csv(file)

def zero_to_nan(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    for col in ZERO_IS_MISSING:
        if col in d.columns:
            d.loc[d[col] == 0, col] = np.nan
    return d

def split_data(df: pd.DataFrame, test_size=0.15, val_size=0.15):
    df_temp, df_test = train_test_split(
        df, test_size=test_size, stratify=df[TARGET_COL], random_state=RANDOM_STATE
    )
    relative_val_size = val_size / (1.0 - test_size)
    df_train, df_val = train_test_split(
        df_temp, test_size=relative_val_size, stratify=df_temp[TARGET_COL], random_state=RANDOM_STATE
    )
    return df_train, df_val, df_test

def build_preprocess(num_cols):
    numeric_preprocess = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    preprocess = ColumnTransformer([
        ("num", numeric_preprocess, num_cols)
    ], remainder="drop")
    return preprocess

def train_quick_baseline(df_train, df_val, model_name="logreg"):
    X_train, y_train = df_train.drop(columns=[TARGET_COL]), df_train[TARGET_COL].values
    X_val, y_val     = df_val.drop(columns=[TARGET_COL]), df_val[TARGET_COL].values

    preprocess = build_preprocess(X_train.columns.tolist())
    if model_name == "logreg":
        model = LogisticRegression(max_iter=200, class_weight="balanced", random_state=RANDOM_STATE)
    else:
        model = RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=RANDOM_STATE)

    pipe = Pipeline([("preprocess", preprocess), ("model", model)])
    pipe.fit(X_train, y_train)
    val_proba = pipe.predict_proba(X_val)[:, 1]
    return pipe, (X_val, y_val, val_proba)

def compute_metrics(y_true, y_proba, threshold=DEFAULT_THRESHOLD):
    y_pred = (y_proba >= threshold).astype(int)
    return {
        "threshold": float(threshold),
        "recall": float(recall_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "pr_auc": float(average_precision_score(y_true, y_proba)),
        "brier": float(brier_score_loss(y_true, y_proba)),
        "cm": confusion_matrix(y_true, y_pred)
    }

def find_threshold_for_recall(y_true, y_proba, target_recall=0.85, min_precision=0.50):
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
    best = {"threshold": DEFAULT_THRESHOLD, "precision": None, "recall": None}
    found = False
    for t_idx, t in enumerate(np.append(thresholds, 1.0)):
        p = precisions[min(t_idx, len(precisions)-1)]
        r = recalls[min(t_idx, len(recalls)-1)]
        if r >= target_recall and p >= min_precision:
            best = {"threshold": float(t), "precision": float(p), "recall": float(r)}
            found = True
            break
    if not found:
        f1s = (2 * precisions * recalls) / (precisions + recalls + 1e-12)
        idx = int(np.nanargmax(f1s))
        t = thresholds[idx] if idx < len(thresholds) else 1.0
        best = {"threshold": float(t), "precision": float(precisions[idx]), "recall": float(recalls[idx])}
    return best

def plot_confusion_matrix(cm, labels=("No Diabetes","Diabetes")):
    z = cm.astype(int)
    fig = go.Figure(data=go.Heatmap(
        z=z, x=labels, y=labels, colorscale="Blues", text=z, texttemplate="%{text}", showscale=False
    ))
    fig.update_layout(
        xaxis_title="Predicted", yaxis_title="Actual", height=350, margin=dict(l=10,r=10,t=30,b=10)
    )
    return fig

def plot_pr_curve(y_true, y_proba):
    p, r, _ = precision_recall_curve(y_true, y_proba)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=r, y=p, mode="lines", name="PR curve"))
    fig.update_layout(xaxis_title="Recall", yaxis_title="Precision", height=350, margin=dict(l=10,r=10,t=30,b=10))
    return fig

def plot_roc_curve(y_true, y_proba):
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name="ROC"))
    fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines", name="random", line=dict(dash="dash")))
    fig.update_layout(xaxis_title="False Positive Rate", yaxis_title="True Positive Rate",
                      height=350, margin=dict(l=10,r=10,t=30,b=10))
    return fig

def plot_calibration(y_true, y_proba, n_bins=10):
    prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=n_bins, strategy="uniform")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines", name="Perfect", line=dict(dash="dash")))
    fig.add_trace(go.Scatter(x=prob_pred, y=prob_true, mode="lines+markers", name="Model"))
    fig.update_layout(xaxis_title="Predicted probability", yaxis_title="Observed frequency",
                      height=350, margin=dict(l=10,r=10,t=30,b=10))
    return fig

def kpi_card(label, value, help_txt=None):
    st.metric(label, value)
    if help_txt:
        st.caption(help_txt)

def load_pipeline_and_threshold(uploaded_pipeline_file, uploaded_thr_file):
    """
    Prioritas Pipeline:
      1) upload sidebar
      2) ./pima_pipeline.joblib
      3) ./artifacts/pima_pipeline.joblib
      4) *.joblib pertama di folder kerja
    Prioritas Threshold:
      1) upload sidebar
      2) ./threshold.json
      3) ./artifacts/threshold.json
    """
    pipeline, threshold, model_name, source = None, DEFAULT_THRESHOLD, "unknown", "—"

    # --- Pipeline ---
    if uploaded_pipeline_file is not None:
        pipeline = safe_joblib_load(uploaded_pipeline_file)
        source = "uploaded"
    else:
        candidates = [Path("pima_pipeline.joblib"), Path("artifacts/pima_pipeline.joblib")]
        candidates += list(Path(".").glob("*.joblib"))  # fallback umum
        for p in candidates:
            if p.exists():
                pipeline = safe_joblib_load(p)
                source = str(p)
                break

    # --- Threshold ---
    if uploaded_thr_file is not None:
        try:
            tjson = json.load(uploaded_thr_file)
            threshold = float(tjson.get("threshold", DEFAULT_THRESHOLD))
            model_name = tjson.get("model", model_name)
        except Exception:
            pass
    else:
        for tpath in [Path("threshold.json"), Path("artifacts/threshold.json")]:
            if tpath.exists():
                try:
                    tjson = json.load(open(tpath))
                    threshold = float(tjson.get("threshold", DEFAULT_THRESHOLD))
                    model_name = tjson.get("model", model_name)
                    break
                except Exception:
                    pass

    return pipeline, threshold, model_name, source

# ===================== Sidebar =====================
with st.sidebar:
    st.title("🩺 Pima Diabetes Dashboard")
    st.markdown("Visualisasi hasil analisis & model skrining diabetes.")
    page = st.radio("Halaman", ["Overview", "Data Explorer", "Model & Evaluasi", "Explainability", "Prediksi (Form)"])
    st.markdown("---")
    st.markdown("### Input & Artefak")
    data_file = st.file_uploader("Upload `diabetes.csv`", type=["csv"])
    pipeline_file = st.file_uploader("Upload pipeline `.joblib` (opsional)", type=["joblib"])
    thr_file = st.file_uploader("Upload `threshold.json` (opsional)", type=["json"])

# ===================== Load Data =====================
df = None
if data_file is not None:
    df = load_csv(data_file)
elif Path("diabetes.csv").exists():
    df = load_csv("diabetes.csv")

if df is None:
    st.warning("Unggah **diabetes.csv** untuk mulai. (Atau letakkan file di direktori yang sama saat menjalankan app.)")
    st.stop()

if TARGET_COL not in df.columns:
    st.error(f"Kolom target `{TARGET_COL}` tidak ditemukan di data!")
    st.stop()

df = zero_to_nan(df)
df_train, df_val, df_test = split_data(df)

# ===================== Load Pipeline / Threshold =====================
pipeline, threshold, best_model_name, model_source = load_pipeline_and_threshold(pipeline_file, thr_file)

# Jika belum ada pipeline, sediakan training cepat
if pipeline is None:
    with st.sidebar:
        st.markdown("---")
        st.subheader("Latih Cepat (tanpa artefak)")
        model_choice = st.selectbox("Model", ["Logistic Regression", "Random Forest"])
        if st.button("Train baseline"):
            model_name = "logreg" if model_choice == "Logistic Regression" else "rf"
            pipeline, (X_val_tmp, y_val_tmp, val_proba_tmp) = train_quick_baseline(df_train, df_val, model_name=model_name)
            thr_info = find_threshold_for_recall(y_val_tmp, val_proba_tmp, target_recall=0.85, min_precision=0.50)
            threshold = thr_info["threshold"]
            best_model_name = model_name
            model_source = "trained in-app"
            st.success(f"Model terlatih. Rekomendasi threshold: {threshold:.3f} (Recall≈{thr_info['recall']:.2f}, Precision≈{thr_info['precision']:.2f})")

# Hitung proba validation jika pipeline tersedia
val_proba = None
if pipeline is not None:
    X_val = df_val.drop(columns=[TARGET_COL]); y_val = df_val[TARGET_COL].values
    try:
        val_proba = pipeline.predict_proba(X_val)[:, 1]
    except Exception:
        val_proba = None

# ===================== Pages =====================
if page == "Overview":
    st.header("📌 Executive Summary")
    pos_rate = df[TARGET_COL].mean()
    col1, col2, col3, col4 = st.columns(4)
    with col1: kpi_card("Total Sampel", f"{len(df):,}")
    with col2: kpi_card("Positif (%)", f"{100*pos_rate:.1f}%")
    with col3: kpi_card("Model", best_model_name if pipeline is not None else "—")
    with col4: kpi_card("Threshold", f"{threshold:.3f}", f"Sumber model: {model_source}")

    st.markdown("**Tujuan**: model *skrining* yang memaksimalkan **Recall** sambil menjaga **Precision** wajar.")
    st.markdown("- Nilai **0** pada kolom klinis tertentu dianggap *missing* dan diimputasi.\n- Split: stratified train/val/test (70/15/15).")

    if pipeline is not None and val_proba is not None:
        m = compute_metrics(y_val, val_proba, threshold)
        a,b,c,d = st.columns(4)
        with a: kpi_card("Recall (Val)", f"{m['recall']:.2f}")
        with b: kpi_card("Precision (Val)", f"{m['precision']:.2f}")
        with c: kpi_card("PR-AUC (Val)", f"{m['pr_auc']:.2f}")
        with d: kpi_card("ROC-AUC (Val)", f"{m['roc_auc']:.2f}")

        st.markdown("**Ringkasan**: Dengan threshold di atas, model menangkap proporsi tinggi pasien yang benar-benar diabetes (Recall), "
                    "dengan trade-off beberapa *false positive* (Precision moderat). Cocok untuk tahap skrining.")

elif page == "Data Explorer":
    st.header("🔎 Data Explorer")
    st.markdown("Distribusi, korelasi, dan *missing* setelah pembersihan 0→NaN.")

    tab1, tab2, tab3 = st.tabs(["Distribusi", "Korelasi", "Missing"])

    with tab1:
        numeric_cols = [c for c in df.columns if c != TARGET_COL]
        sel = st.multiselect("Pilih fitur untuk histogram:", numeric_cols, default=numeric_cols[:4])
        ncols = 3
        rows = (len(sel) + ncols - 1)//ncols
        plot_df = df.copy()
        plot_df["OutcomeLabel"] = plot_df[TARGET_COL].map({0: "No", 1: "Yes"})
        for r in range(rows):
            cols = st.columns(ncols)
            for i, c in enumerate(sel[r*ncols:(r+1)*ncols]):
                with cols[i]:
                    fig = px.histogram(plot_df, x=c, color="OutcomeLabel", barmode="overlay", nbins=40)
                    fig.update_layout(height=260, margin=dict(l=10,r=10,t=30,b=10), legend_title_text="Outcome")
                    st.plotly_chart(fig, use_container_width=True)

    with tab2:
        corr = df.drop(columns=[TARGET_COL]).corr(numeric_only=True)
        fig = px.imshow(corr, color_continuous_scale="RdBu", zmin=-1, zmax=1, aspect="auto")
        fig.update_layout(height=600, margin=dict(l=10,r=10,t=30,b=10))
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        miss = df.isna().sum().sort_values(ascending=False)
        st.dataframe(miss.to_frame("missing_count"))

elif page == "Model & Evaluasi":
    st.header("🧪 Model & Evaluasi")
    if pipeline is None:
        st.info("Latih model dulu (sidebar) **atau** unggah pipeline `.joblib`.")
    else:
        X_val = df_val.drop(columns=[TARGET_COL]); y_val = df_val[TARGET_COL].values
        try:
            val_proba = pipeline.predict_proba(X_val)[:, 1]
        except Exception as e:
            st.error(f"Pipeline tidak mendukung predict_proba: {e}")
            st.stop()

        st.subheader("Threshold Tuning")
        c1, c2, c3 = st.columns([2,2,1])
        with c1:
            target_recall = st.slider("Target Recall", 0.5, 0.99, 0.85, 0.01)
        with c2:
            min_precision = st.slider("Minimal Precision", 0.1, 0.95, 0.50, 0.01)
        with c3:
            if st.button("Cari Threshold"):
                info = find_threshold_for_recall(y_val, val_proba, target_recall, min_precision)
                threshold = info["threshold"]
                st.success(f"Threshold rekomendasi: {threshold:.3f} (Recall≈{info['recall']:.2f}, Precision≈{info['precision']:.2f})")

        threshold = st.slider("Set Threshold", 0.0, 1.0, float(threshold), 0.01)

        m = compute_metrics(y_val, val_proba, threshold)
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Recall", f"{m['recall']:.2f}")
        k2.metric("Precision", f"{m['precision']:.2f}")
        k3.metric("F1", f"{m['f1']:.2f}")
        k4.metric("PR-AUC", f"{m['pr_auc']:.2f}")
        k5.metric("ROC-AUC", f"{m['roc_auc']:.2f}")

        cma, cmb = st.columns([1,1])
        with cma:
            st.subheader("Confusion Matrix")
            st.plotly_chart(plot_confusion_matrix(m["cm"]), use_container_width=True)
        with cmb:
            st.subheader("Calibration")
            st.plotly_chart(plot_calibration(y_val, val_proba), use_container_width=True)

        ca, cb = st.columns(2)
        with ca:
            st.subheader("Precision–Recall Curve")
            st.plotly_chart(plot_pr_curve(y_val, val_proba), use_container_width=True)
        with cb:
            st.subheader("ROC Curve")
            st.plotly_chart(plot_roc_curve(y_val, val_proba), use_container_width=True)

        st.caption("Tips: Untuk skrining, geser threshold agar **Recall** tinggi sambil menjaga **Precision** tetap wajar.")

elif page == "Explainability":
    st.header("🧠 Explainability")
    if pipeline is None:
        st.info("Latih/unggah pipeline dulu.")
        st.stop()

    X_val = df_val.drop(columns=[TARGET_COL]); y_val = df_val[TARGET_COL].values
    try:
        if SHAP_AVAILABLE:
            st.subheader("Global Importance (SHAP)")
            num_cols = X_val.columns.tolist()
            X_val_t = pipeline.named_steps["preprocess"].transform(X_val)
            X_val_t = pd.DataFrame(X_val_t, columns=num_cols)
            model = pipeline.named_steps["model"]
            try:
                explainer = shap.Explainer(model, X_val_t)
                shap_values = explainer(X_val_t)
                fig = plt.figure()
                shap.plots.beeswarm(shap_values, max_display=10, show=False)
                st.pyplot(fig, clear_figure=True)
                plt.close(fig)
            except Exception:
                if hasattr(model, "coef_"):
                    coefs = pd.Series(model.coef_.ravel(), index=num_cols).sort_values(key=lambda s: s.abs(), ascending=False)
                    st.bar_chart(coefs)
                elif hasattr(model, "feature_importances_"):
                    imps = pd.Series(model.feature_importances_, index=num_cols).sort_values(ascending=False)
                    st.bar_chart(imps)
        else:
            st.info("`shap` tidak tersedia. Menampilkan importance sederhana.")
            model = pipeline.named_steps["model"]
            num_cols = X_val.columns.tolist()
            if hasattr(model, "coef_"):
                coefs = pd.Series(model.coef_.ravel(), index=num_cols).sort_values(key=lambda s: s.abs(), ascending=False)
                st.bar_chart(coefs)
            elif hasattr(model, "feature_importances_"):
                imps = pd.Series(model.feature_importances_, index=num_cols).sort_values(ascending=False)
                st.bar_chart(imps)
            else:
                st.warning("Model tidak menyediakan koefisien/feature_importances_.")
    except Exception as e:
        st.error(f"Gagal menghitung explainability: {e}")

    st.markdown("---")
    st.subheader("Per-sampel Explanation")
    idx = st.number_input("Index sampel (validation set)", min_value=0, max_value=len(X_val)-1, value=0, step=1)
    x = X_val.iloc[[idx]]
    try:
        proba = pipeline.predict_proba(x)[0,1]
        st.write(f"Probabilitas diabetes (sampel {idx}): **{proba:.3f}**")
    except Exception as e:
        st.error(f"Gagal prediksi: {e}")

elif page == "Prediksi (Form)":
    st.header("🧮 Prediksi Pasien Baru")
    if pipeline is None:
        st.info("Latih/unggah pipeline dulu.")
        st.stop()

    cols = st.columns(4)
    with cols[0]:
        Pregnancies = st.number_input("Pregnancies", min_value=0, value=2, step=1)
        Glucose = st.number_input("Glucose", min_value=0.0, value=120.0, step=1.0)
    with cols[1]:
        BloodPressure = st.number_input("BloodPressure", min_value=0.0, value=70.0, step=1.0)
        SkinThickness = st.number_input("SkinThickness", min_value=0.0, value=20.0, step=1.0)
    with cols[2]:
        Insulin = st.number_input("Insulin", min_value=0.0, value=80.0, step=1.0)
        BMI = st.number_input("BMI", min_value=0.0, value=28.0, step=0.1, format="%.1f")
    with cols[3]:
        DiabetesPedigreeFunction = st.number_input("DiabetesPedigreeFunction", min_value=0.0, value=0.5, step=0.01, format="%.2f")
        Age = st.number_input("Age", min_value=0, value=35, step=1)

    user_df = pd.DataFrame([{
        "Pregnancies": Pregnancies,
        "Glucose": Glucose,
        "BloodPressure": BloodPressure,
        "SkinThickness": SkinThickness,
        "Insulin": Insulin,
        "BMI": BMI,
        "DiabetesPedigreeFunction": DiabetesPedigreeFunction,
        "Age": Age
    }])
    user_df = zero_to_nan(user_df)

    st.markdown("---")
    thr_val = st.slider("Threshold klasifikasi", 0.0, 1.0, float(threshold), 0.01,
                        help="Nilai di atas threshold diklasifikasikan sebagai 'Diabetes'.")
    if st.button("Prediksi"):
        try:
            proba = float(pipeline.predict_proba(user_df)[0,1])
            label = int(proba >= thr_val)
            risk = "Diabetes" if label==1 else "Tidak Diabetes"
            st.subheader(f"Hasil: **{risk}** (prob={proba:.3f})")
            st.caption("Disclaimer: alat bantu skrining, bukan diagnosis. Konsultasikan ke tenaga medis.")
        except Exception as e:
            st.error(f"Gagal memprediksi: {e}")
