import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import time
from io import StringIO
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, IsolationForest
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, confusion_matrix
import shap
import pickle

# Modul tambahan untuk modul baru:
try:
    from prophet import Prophet
except ImportError:
    Prophet = None

try:
    import mlflow
except ImportError:
    mlflow = None

# Untuk visualisasi interaktif SHAP (jika tersedia)
try:
    from streamlit_shap import st_shap
except ImportError:
    st_shap = None

# --- Modul Unggah Data ---
def upload_data_section():
    st.header("Unggah Data")
    uploaded_file = st.file_uploader("Unggah file CSV", type="csv")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.success("Data berhasil dimuat!")
        st.dataframe(df)
        st.session_state["data"] = df
    else:
        st.info("Silakan unggah file CSV.")

# --- Modul Preprocessing Data ---
def data_preprocessing_section(df):
    st.header("Preprocessing Data")
    st.write("Pilih metode untuk menangani missing values dan normalisasi data.")
    method = st.selectbox("Metode imputasi", ["Hapus baris dengan missing", "Isi dengan Mean", "Isi dengan Median"])
    df_processed = df.copy()
    if method == "Hapus baris dengan missing":
        df_processed = df_processed.dropna()
        st.success("Baris dengan missing values telah dihapus.")
    elif method == "Isi dengan Mean":
        for col in df_processed.select_dtypes(include=np.number).columns:
            df_processed[col].fillna(df_processed[col].mean(), inplace=True)
        st.success("Missing values diisi dengan mean.")
    elif method == "Isi dengan Median":
        for col in df_processed.select_dtypes(include=np.number).columns:
            df_processed[col].fillna(df_processed[col].median(), inplace=True)
        st.success("Missing values diisi dengan median.")
    
    if st.checkbox("Normalisasi Data (Standard Scaler)"):
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        num_cols = df_processed.select_dtypes(include=np.number).columns.tolist()
        df_processed[num_cols] = scaler.fit_transform(df_processed[num_cols])
        st.success("Data numerik telah dinormalisasi.")
    st.write("Data setelah preprocessing:")
    st.dataframe(df_processed)
    st.session_state["data_preprocessed"] = df_processed

# --- Modul Eksplorasi Data (EDA) ---
def eda_section(df):
    st.header("Eksplorasi Data (EDA)")
    st.dataframe(df)
    st.write("Deskripsi Data:")
    st.write(df.describe())
    numeric_df = df.select_dtypes(include=np.number)
    if numeric_df.shape[1] >= 2:
        st.subheader("Heatmap Korelasi")
        fig, ax = plt.subplots()
        sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)
    else:
        st.info("Tidak cukup data numerik untuk heatmap.")

# --- Modul Feature Engineering ---
def feature_engineering_section(df):
    st.header("Feature Engineering")
    st.write("Buat fitur baru dari fitur yang ada.")
    new_feature_name = st.text_input("Nama Fitur Baru", value="new_feature")
    feature_options = st.multiselect("Pilih fitur untuk digabungkan", df.columns.tolist())
    operation = st.selectbox("Pilih operasi", ["Jumlahkan", "Rata-rata", "Kombinasi string"])
    if st.button("Buat Fitur"):
        df_fe = df.copy()
        if operation == "Jumlahkan":
            try:
                df_fe[new_feature_name] = df_fe[feature_options].sum(axis=1)
                st.success("Fitur baru berhasil dibuat dengan operasi penjumlahan.")
            except Exception as e:
                st.error(f"Error: {e}")
        elif operation == "Rata-rata":
            try:
                df_fe[new_feature_name] = df_fe[feature_options].mean(axis=1)
                st.success("Fitur baru berhasil dibuat dengan operasi rata-rata.")
            except Exception as e:
                st.error(f"Error: {e}")
        elif operation == "Kombinasi string":
            try:
                df_fe[new_feature_name] = df_fe[feature_options].astype(str).agg(' '.join, axis=1)
                st.success("Fitur baru berhasil dibuat dengan kombinasi string.")
            except Exception as e:
                st.error(f"Error: {e}")
        st.write("Data setelah penambahan fitur:")
        st.dataframe(df_fe)
        st.session_state["data_fe"] = df_fe

# --- Modul Pelatihan Model ---
def model_training_section(df):
    st.header("Pelatihan Model")
    numeric_columns = df.select_dtypes(include=np.number).columns.tolist()
    categorical_columns = df.select_dtypes(include=["object", "category"]).columns.tolist()
    model_type = st.radio("Pilih Tipe Model", ["Regresi", "Klasifikasi"])
    if model_type == "Regresi" and len(numeric_columns) >= 2:
        target = st.selectbox("Pilih kolom target (numerik)", numeric_columns)
        features = st.multiselect("Pilih kolom fitur", [col for col in numeric_columns if col != target])
        algorithm = st.selectbox("Pilih Algoritma", ["Linear Regression", "Random Forest Regressor"])
        if st.button("Latih Model Regresi") and features:
            X = df[features]
            y = df[target]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model = LinearRegression() if algorithm == "Linear Regression" else RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            mse = mean_squared_error(y_test, predictions)
            r2 = r2_score(y_test, predictions)
            st.write("Model telah dilatih.")
            st.write("Mean Squared Error (MSE):", mse)
            st.write("R² Score:", r2)
            result_df = X_test.copy()
            result_df["Nilai Aktual"] = y_test
            result_df["Nilai Prediksi"] = predictions
            st.dataframe(result_df)
            st.subheader("Visualisasi: Actual vs Prediksi")
            fig = px.scatter(result_df, x="Nilai Aktual", y="Nilai Prediksi", title="Perbandingan Aktual vs Prediksi")
            st.plotly_chart(fig)
            st.session_state["model"] = model
            st.session_state["features"] = features
            st.session_state["X"] = X
    elif model_type == "Klasifikasi" and (len(numeric_columns) + len(categorical_columns)) >= 2:
        all_columns = numeric_columns + categorical_columns
        target = st.selectbox("Pilih kolom target (kategorikal)", all_columns)
        features = st.multiselect("Pilih kolom fitur", [col for col in all_columns if col != target])
        algorithm = st.selectbox("Pilih Algoritma", ["Logistic Regression", "Random Forest Classifier"])
        if st.button("Latih Model Klasifikasi") and features:
            X = df[features]
            y = df[target]
            X = pd.get_dummies(X)
            if y.dtype == object or str(y.dtype).startswith("category"):
                y = pd.factorize(y)[0]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model = LogisticRegression(max_iter=200) if algorithm == "Logistic Regression" else RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            acc = accuracy_score(y_test, predictions)
            st.write("Model telah dilatih.")
            st.write("Akurasi:", acc)
            cm = confusion_matrix(y_test, predictions)
            st.write("Confusion Matrix:")
            st.write(cm)
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
            st.pyplot(fig)
            st.session_state["model"] = model
            st.session_state["features"] = features
            st.session_state["X"] = df[features]
    else:
        st.error("Dataset tidak memiliki kolom yang sesuai untuk tipe model yang dipilih.")

def hyperparameter_tuning_section(df):
    st.header("Hyperparameter Tuning - Random Forest Regressor")
    numeric_columns = df.select_dtypes(include=np.number).columns.tolist()
    if len(numeric_columns) < 2:
        st.error("Tidak cukup kolom numerik untuk tuning.")
        return
    target = st.selectbox("Pilih kolom target (numerik)", numeric_columns)
    features = st.multiselect("Pilih kolom fitur", [col for col in numeric_columns if col != target])
    if st.button("Lakukan Tuning") and features:
        X = df[features]
        y = df[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        param_grid = {
            'n_estimators': [50, 100, 150],
            'max_depth': [None, 5, 10]
        }
        rf = RandomForestRegressor(random_state=42)
        grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, scoring='r2')
        grid_search.fit(X_train, y_train)
        st.write("Best Parameters:", grid_search.best_params_)
        st.write("Best Cross-Validation Score (R²):", grid_search.best_score_)
        best_model = grid_search.best_estimator_
        predictions = best_model.predict(X_test)
        r2 = r2_score(y_test, predictions)
        st.write("R² Score pada data test:", r2)

def nlp_section(df):
    st.header("Natural Language Query")
    query = st.text_input("Masukkan pertanyaan Anda:")
    if query:
        if "jumlah" in query.lower() or "berapa" in query.lower():
            response = f"Tabel memiliki {len(df)} baris dan {df.shape[1]} kolom."
        elif "rata-rata" in query.lower():
            numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
            if numeric_cols:
                means = df[numeric_cols].mean().to_dict()
                response = "Rata-rata kolom numerik: " + ", ".join([f"{k}: {v:.2f}" for k, v in means.items()])
            else:
                response = "Tidak ada kolom numerik untuk dihitung rata-rata."
        else:
            response = "Fitur NLP untuk query kompleks masih dalam pengembangan."
        st.write("Jawaban:", response)

def realtime_dashboard_section():
    st.header("Realtime Dashboard")
    st.write("Simulasi data streaming dengan pembaruan setiap detik.")
    chart_data = pd.DataFrame(np.random.randn(10, 2), columns=["x", "y"])
    chart = st.line_chart(chart_data)
    for i in range(20):
        new_data = pd.DataFrame(np.random.randn(1, 2), columns=["x", "y"])
        chart_data = pd.concat([chart_data, new_data], ignore_index=True)
        chart.line_chart(chart_data)
        time.sleep(1)
    st.success("Simulasi selesai.")

def download_report_section(df):
    st.header("Download Laporan Analisis")
    report = StringIO()
    report.write("Laporan Analisis Data\n")
    report.write("====================\n\n")
    report.write("Ringkasan Data:\n")
    report.write(str(df.describe()))
    report.seek(0)
    st.download_button("Unduh Laporan", report, file_name="laporan_analisis.txt", mime="text/plain")

def database_connection_section():
    st.header("Koneksi Database SQL")
    st.write("Pilih metode koneksi ke database:")
    option = st.radio("Pilih opsi:", ["Gunakan st.connection (dengan secrets)", "Unggah file SQLite"])
    if option == "Gunakan st.connection (dengan secrets)":
        if "sql" in st.secrets.get("connections", {}):
            try:
                conn = st.connection("sql")
                query = st.text_area("Masukkan query SQL", "SELECT * FROM your_table LIMIT 10;")
                if st.button("Jalankan Query"):
                    df = conn.query(query, ttl=600)
                    st.dataframe(df)
            except Exception as e:
                st.error(f"Terjadi kesalahan: {e}")
        else:
            st.info("Secrets untuk koneksi SQL tidak ditemukan. Silakan konfigurasi st.secrets.")
    else:
        db_file = st.file_uploader("Unggah file SQLite (.db atau .sqlite)", type=["db", "sqlite"])
        if db_file is not None:
            with open("temp_sqlite.db", "wb") as f:
                f.write(db_file.getbuffer())
            import sqlite3
            conn = sqlite3.connect("temp_sqlite.db")
            query = st.text_area("Masukkan query SQL", "SELECT name FROM sqlite_master WHERE type='table';")
            if st.button("Jalankan Query"):
                try:
                    df = pd.read_sql_query(query, conn)
                    st.dataframe(df)
                except Exception as e:
                    st.error(f"Terjadi kesalahan: {e}")

def model_interpretability_section():
    st.header("Model Interpretability dengan SHAP")
    if "model" not in st.session_state or "X" not in st.session_state:
        st.error("Tidak ada model yang tersedia. Silakan latih model terlebih dahulu.")
        return
    model = st.session_state["model"]
    X = st.session_state["X"]
    st.write("Menghitung nilai SHAP...")
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        st.write("SHAP values berhasil dihitung.")
        st.subheader("SHAP Summary Plot")
        fig, ax = plt.subplots(figsize=(10, 6))
        shap.summary_plot(shap_values, X, plot_type="dot", show=False)
        st.pyplot(fig)
        if st_shap is not None:
            st.subheader("Interactive SHAP Force Plot")
            instance_index = 0
            force_plot = shap.force_plot(explainer.expected_value, shap_values[instance_index, :],
                                         X.iloc[instance_index, :], matplotlib=False)
            st_shap(force_plot, height=300)
    except Exception as e:
        st.error(f"Error dalam menghitung SHAP values: {e}")

def model_persistence_section():
    st.header("Model Persistence")
    st.write("Simpan model yang telah dilatih atau muat model yang telah disimpan.")
    option = st.selectbox("Pilih aksi:", ["Simpan Model", "Muat Model"])
    if option == "Simpan Model":
        if "model" in st.session_state:
            model_filename = st.text_input("Nama file untuk menyimpan model", value="model.pkl")
            if st.button("Simpan Model"):
                with open(model_filename, "wb") as f:
                    pickle.dump(st.session_state["model"], f)
                st.success(f"Model disimpan sebagai {model_filename}")
        else:
            st.error("Tidak ada model yang tersedia. Silakan latih model terlebih dahulu.")
    else:
        uploaded_model = st.file_uploader("Unggah file model (.pkl)", type=["pkl"])
        if uploaded_model is not None:
            try:
                model = pickle.load(uploaded_model)
                st.session_state["model"] = model
                st.success("Model berhasil dimuat!")
            except Exception as e:
                st.error(f"Gagal memuat model: {e}")

# --- Modul Baru: Time Series Forecasting ---
def time_series_forecasting_section(df):
    st.header("Time Series Forecasting")
    if Prophet is None:
        st.error("Library Prophet tidak terpasang. Install dengan 'pip install prophet'.")
        return
    date_col = st.selectbox("Pilih kolom tanggal", df.columns.tolist())
    target_col = st.selectbox("Pilih kolom target numerik", df.select_dtypes(include=np.number).columns.tolist())
    periods = st.number_input("Jumlah periode untuk forecast", min_value=1, value=10)
    if st.button("Lakukan Forecasting"):
        df_ts = df[[date_col, target_col]].dropna()
        try:
            df_ts[date_col] = pd.to_datetime(df_ts[date_col])
        except Exception as e:
            st.error(f"Kolom tanggal tidak bisa dikonversi: {e}")
            return
        df_ts = df_ts.rename(columns={date_col: "ds", target_col: "y"})
        m = Prophet()
        m.fit(df_ts)
        future = m.make_future_dataframe(periods=periods)
        forecast = m.predict(future)
        fig = m.plot(forecast)
        st.pyplot(fig)
        st.write("Data Forecast:")
        st.dataframe(forecast.tail(periods))

# --- Modul Baru: Anomaly Detection ---
def anomaly_detection_section(df):
    st.header("Anomaly Detection")
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    if not num_cols:
        st.error("Tidak ada kolom numerik untuk deteksi anomaly.")
        return
    col = st.selectbox("Pilih kolom untuk deteksi anomaly", num_cols)
    contamination = st.slider("Tingkat kontaminasi (prosentase anomaly)", 0.01, 0.5, 0.05)
    if st.button("Deteksi Anomaly"):
        X = df[[col]].dropna()
        iso_forest = IsolationForest(contamination=contamination, random_state=42)
        preds = iso_forest.fit_predict(X)
        df_anomaly = X.copy()
        df_anomaly["anomaly"] = preds
        fig = px.scatter(df_anomaly, x=df_anomaly.index, y=col, color="anomaly",
                         title="Deteksi Anomaly")
        st.plotly_chart(fig)
        st.write("Tabel anomaly:")
        st.dataframe(df_anomaly)

# --- Modul Baru: Interactive Data Filters ---
def interactive_filters_section(df):
    st.header("Interactive Data Filters")
    st.write("Filter data secara interaktif.")
    filtered_df = df.copy()
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            min_val = float(df[col].min())
            max_val = float(df[col].max())
            val = st.slider(f"Filter {col}", min_val, max_val, (min_val, max_val))
            filtered_df = filtered_df[(df[col] >= val[0]) & (df[col] <= val[1])]
        elif pd.api.types.is_string_dtype(df[col]):
            unique_vals = df[col].unique().tolist()
            selected = st.multiselect(f"Filter {col}", unique_vals, default=unique_vals)
            filtered_df = filtered_df[df[col].isin(selected)]
    st.write("Data yang telah difilter:")
    st.dataframe(filtered_df)

# --- Modul Baru: Advanced Reporting ---
def advanced_reporting_section(df):
    st.header("Advanced Reporting")
    st.write("Buat laporan interaktif dalam format HTML.")
    report_html = f"""
    <html>
    <head>
        <title>Laporan Analisis Data</title>
    </head>
    <body>
        <h1>Laporan Analisis Data</h1>
        <h2>Ringkasan Statistik</h2>
        {df.describe().to_html()}
    </body>
    </html>
    """
    st.markdown(report_html, unsafe_allow_html=True)
    st.download_button("Unduh Laporan HTML", report_html, file_name="laporan.html", mime="text/html")

# --- Modul Baru: Scheduling & Batch Processing ---
def scheduling_batch_section():
    st.header("Scheduling & Batch Processing")
    st.write("Fitur ini sedang dalam pengembangan untuk penjadwalan analisis data secara berkala.")
    st.info("Integrasi dengan cron jobs atau scheduler lainnya akan ditambahkan pada update selanjutnya.")

# --- Modul Baru: User Authentication (Placeholder) ---
def user_authentication_section():
    st.header("User Authentication")
    st.write("Fitur autentikasi pengguna sedang dalam pengembangan.")
    st.info("Implementasi autentikasi akan menggunakan OAuth2 atau integrasi dengan layanan pihak ketiga.")

# --- Modul Baru: MLflow Model Tracking (Placeholder) ---
def mlflow_tracking_section():
    st.header("MLflow Model Tracking")
    if mlflow is None:
        st.error("Library MLflow tidak terpasang. Install dengan 'pip install mlflow'.")
        return
    st.write("Fitur tracking model dengan MLflow akan menampilkan metrik dan versi model.")
    st.info("Integrasi dengan MLflow sedang dalam tahap pengembangan.")

# --- Fungsi Utama Aplikasi ---
def main():
    st.title("Analytica360 - Platform Analisis Data Terpadu (Versi Super Lanjutan)")
    st.sidebar.title("Menu Navigasi")
    menu = st.sidebar.radio("Pilih Menu", [
        "Unggah Data", "Preprocessing Data", "Eksplorasi Data", "Feature Engineering",
        "Pelatihan Model", "Hyperparameter Tuning", "Natural Language Query",
        "Realtime Dashboard", "Download Laporan", "Koneksi Database",
        "Model Interpretability", "Model Persistence",
        "Time Series Forecasting", "Anomaly Detection", "Interactive Filters",
        "Advanced Reporting", "Scheduling & Batch Processing",
        "User Authentication", "MLflow Tracking"
    ])
    if menu == "Unggah Data":
        upload_data_section()
    elif menu == "Preprocessing Data":
        if "data" not in st.session_state:
            st.warning("Silakan unggah data terlebih dahulu pada menu 'Unggah Data'.")
        else:
            data_preprocessing_section(st.session_state["data"])
    elif menu == "Eksplorasi Data":
        if "data" not in st.session_state:
            st.warning("Silakan unggah data terlebih dahulu pada menu 'Unggah Data'.")
        else:
            df = st.session_state.get("data_preprocessed", st.session_state["data"])
            eda_section(df)
    elif menu == "Feature Engineering":
        if "data" not in st.session_state:
            st.warning("Silakan unggah data terlebih dahulu pada menu 'Unggah Data'.")
        else:
            df = st.session_state.get("data_fe", st.session_state["data"])
            feature_engineering_section(df)
    elif menu == "Pelatihan Model":
        if "data" not in st.session_state:
            st.warning("Silakan unggah data terlebih dahulu pada menu 'Unggah Data'.")
        else:
            df = st.session_state.get("data_fe", st.session_state["data"])
            model_training_section(df)
    elif menu == "Hyperparameter Tuning":
        if "data" not in st.session_state:
            st.warning("Silakan unggah data terlebih dahulu pada menu 'Unggah Data'.")
        else:
            df = st.session_state.get("data_fe", st.session_state["data"])
            hyperparameter_tuning_section(df)
    elif menu == "Natural Language Query":
        if "data" not in st.session_state:
            st.warning("Silakan unggah data terlebih dahulu pada menu 'Unggah Data'.")
        else:
            df = st.session_state.get("data_fe", st.session_state["data"])
            nlp_section(df)
    elif menu == "Realtime Dashboard":
        realtime_dashboard_section()
    elif menu == "Download Laporan":
        if "data" not in st.session_state:
            st.warning("Silakan unggah data terlebih dahulu pada menu 'Unggah Data'.")
        else:
            df = st.session_state.get("data_fe", st.session_state["data"])
            download_report_section(df)
    elif menu == "Koneksi Database":
        database_connection_section()
    elif menu == "Model Interpretability":
        model_interpretability_section()
    elif menu == "Model Persistence":
        model_persistence_section()
    elif menu == "Time Series Forecasting":
        if "data" not in st.session_state:
            st.warning("Silakan unggah data terlebih dahulu pada menu 'Unggah Data'.")
        else:
            time_series_forecasting_section(st.session_state.get("data_preprocessed", st.session_state["data"]))
    elif menu == "Anomaly Detection":
        if "data" not in st.session_state:
            st.warning("Silakan unggah data terlebih dahulu pada menu 'Unggah Data'.")
        else:
            anomaly_detection_section(st.session_state.get("data_preprocessed", st.session_state["data"]))
    elif menu == "Interactive Filters":
        if "data" not in st.session_state:
            st.warning("Silakan unggah data terlebih dahulu pada menu 'Unggah Data'.")
        else:
            interactive_filters_section(st.session_state.get("data_fe", st.session_state["data"]))
    elif menu == "Advanced Reporting":
        if "data" not in st.session_state:
            st.warning("Silakan unggah data terlebih dahulu pada menu 'Unggah Data'.")
        else:
            advanced_reporting_section(st.session_state.get("data_fe", st.session_state["data"]))
    elif menu == "Scheduling & Batch Processing":
        scheduling_batch_section()
    elif menu == "User Authentication":
        user_authentication_section()
    elif menu == "MLflow Tracking":
        mlflow_tracking_section()

if __name__ == "__main__":
    main()
