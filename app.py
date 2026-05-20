"""
Dashboard de Análisis de Datos con Machine Learning
Proyecto universitario — Python + Streamlit
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import warnings, io
warnings.filterwarnings("ignore")

# ── Configuración ──────────────────────────────────────────────────
st.set_page_config(page_title="Dashboard ML", page_icon="", layout="wide")
st.title("Dashboard de Análisis de Datos Elihu Gay")
#st.markdown("Sube un archivo CSV y obtén estadísticas, gráficas y predicciones automáticas.")
st.divider()

st.sidebar.header("Configuración")
archivo = st.sidebar.file_uploader("Sube tu archivo CSV", type=["csv"])
st.sidebar.markdown("---")
#st.sidebar.caption("Proyecto 184642")

# ── Función robusta para leer CUALQUIER CSV ────────────────────────
def leer_csv_robusto(archivo):
    """
    Intenta leer el CSV con distintas configuraciones hasta encontrar
    la que funcione: detecta separador, encoding y filas corruptas.
    """
    contenido = archivo.read()

    # Lista de configuraciones a intentar en orden
    intentos = [
        dict(sep=",",  encoding="utf-8"),
        dict(sep=";",  encoding="utf-8"),
        dict(sep="\t", encoding="utf-8"),
        dict(sep=",",  encoding="latin-1"),
        dict(sep=";",  encoding="latin-1"),
        dict(sep=",",  encoding="utf-8",   on_bad_lines="skip"),
        dict(sep=";",  encoding="latin-1", on_bad_lines="skip"),
        dict(sep=None, encoding="utf-8",   engine="python", on_bad_lines="skip"),
    ]

    for config in intentos:
        try:
            df = pd.read_csv(io.BytesIO(contenido), **config)
            if df.shape[1] > 1:   # al menos 2 columnas = lectura válida
                return df, None
        except Exception:
            continue

    return None, "No se pudo leer el archivo. Verifica que sea un CSV válido."


# ── Función robusta para limpiar el DataFrame ──────────────────────
def limpiar_df(df_raw):
    """
    Extrae solo columnas numéricas válidas y elimina NaN e infinitos.
    Retorna el DataFrame limpio y la lista de columnas numéricas.
    """
    # Solo columnas numéricas
    cols_num = df_raw.select_dtypes(include="number").columns.tolist()

    # Quitar columnas con más del 50% de nulos
    cols_num = [c for c in cols_num if df_raw[c].isnull().mean() < 0.5]

    if not cols_num:
        return None, [], "No se encontraron columnas numéricas válidas."

    df = df_raw[cols_num].copy()

    # Reemplazar infinitos por NaN y rellenar con mediana
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    for col in df.columns:
        mediana = df[col].median()
        df[col].fillna(mediana if not np.isnan(mediana) else 0, inplace=True)

    # Quitar columnas con varianza cero
    cols_num = [c for c in df.columns if df[c].std() > 0]
    df = df[cols_num]

    # Limpieza final
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    if len(df) < 10:
        return None, [], "El archivo no tiene suficientes filas válidas (mínimo 10)."

    return df, cols_num, None


# ── Estado vacío ───────────────────────────────────────────────────
if archivo is None:
    st.info("Sube un archivo CSV desde la barra lateral para comenzar.")
    np.random.seed(42)
    n = 120
    area         = np.random.randint(40, 180, n)
    habitaciones = np.round(area/35 + np.random.normal(0,.4,n)).clip(1,5).astype(int)
    antiguedad   = np.random.randint(0, 30, n)
    distancia    = np.round(np.random.uniform(0.5, 14, n), 1)
    precio       = (area*1200 + habitaciones*8000 - antiguedad*500
                    - distancia*3000 + np.random.normal(0,8000,n)).round(-2).astype(int)
    st.subheader("Dataset de ejemplo")
    st.dataframe(pd.DataFrame({
        "area_m2": area, "habitaciones": habitaciones,
        "antiguedad_anos": antiguedad, "distancia_km": distancia,
        "precio_usd": precio,
    }).head(10), use_container_width=True)
    #st.caption("Sube tu propio CSV para analizarlo.")
    st.stop()

# ── Leer archivo ───────────────────────────────────────────────────
df_raw, error_lectura = leer_csv_robusto(archivo)

if error_lectura:
    st.error(f"Error al leer el archivo: {error_lectura}")
    st.stop()

# ── Limpiar datos ──────────────────────────────────────────────────
df, cols_num, error_limpieza = limpiar_df(df_raw)

if error_limpieza:
    st.error(f"{error_limpieza}")
    st.info("Sugerencia: verifica que tu CSV tenga columnas con números.")
    st.dataframe(df_raw.head(5))
    st.stop()

# Aviso de columnas ignoradas
cols_ignoradas = [c for c in df_raw.columns if c not in cols_num]
if cols_ignoradas:
    st.sidebar.warning(
        f"Se ignoraron {len(cols_ignoradas)} columnas no numéricas:\n"
        + ", ".join(cols_ignoradas[:6])
        + ("..." if len(cols_ignoradas) > 6 else "")
    )

# ── Pestañas ───────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["Exploración", "Visualizaciones", "Predicción"])

# ══════════════════════════════════════════════════════
# PESTAÑA 1 — Exploración
# ══════════════════════════════════════════════════════
with tab1:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Filas",              df.shape[0])
    c2.metric("Columnas numéricas", len(cols_num))
    c3.metric("Columnas totales",   df_raw.shape[1])
    c4.metric("Valores nulos",      int(df.isnull().sum().sum()))

    st.subheader("Vista previa")
    st.dataframe(df.head(20), use_container_width=True)

    st.subheader("Estadísticas descriptivas")
    st.dataframe(df.describe().round(2), use_container_width=True)

# ══════════════════════════════════════════════════════
# PESTAÑA 2 — Visualizaciones
# ══════════════════════════════════════════════════════
with tab2:
    col_sel = st.selectbox("Variable para histograma", cols_num)

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    axes[0].hist(df[col_sel].dropna(), bins=25, color="#4C72B0", edgecolor="white")
    axes[0].axvline(df[col_sel].mean(), color="red", linestyle="--",
                    label=f"Media: {df[col_sel].mean():.2f}")
    axes[0].set_title(f"Distribución de '{col_sel}'")
    axes[0].set_xlabel(col_sel)
    axes[0].set_ylabel("Frecuencia")
    axes[0].legend()
    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)

    cols_corr = cols_num[:12]
    corr = df[cols_corr].corr()
    sns.heatmap(corr, annot=len(cols_corr) <= 8, fmt=".2f",
                cmap="RdBu_r", center=0, ax=axes[1],
                linewidths=0.3, square=True)
    axes[1].set_title("Correlación entre variables")
    plt.tight_layout()
    st.pyplot(fig)

    st.subheader("Dispersión entre dos variables")
    col_x = st.selectbox("Eje X", cols_num, index=0, key="x")
    col_y = st.selectbox("Eje Y", cols_num, index=len(cols_num)-1, key="y")

    fig2, ax = plt.subplots(figsize=(8, 4))
    df_sample = df[[col_x, col_y]].dropna().sample(min(2000, len(df)), random_state=42)
    ax.scatter(df_sample[col_x], df_sample[col_y], alpha=0.5,
               color="#DD8452", edgecolors="white", linewidths=0.2)
    try:
        m, b = np.polyfit(df_sample[col_x], df_sample[col_y], 1)
        x_l = np.linspace(df[col_x].min(), df[col_x].max(), 100)
        ax.plot(x_l, m*x_l + b, color="red", linewidth=2, label="Tendencia")
        ax.legend()
    except Exception:
        pass
    ax.set_xlabel(col_x); ax.set_ylabel(col_y)
    ax.set_title(f"'{col_y}' vs '{col_x}'")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig2)

# ══════════════════════════════════════════════════════
# PESTAÑA 3 — Predicción
# ══════════════════════════════════════════════════════
with tab3:
    st.subheader("Configuración del modelo")

    target   = st.selectbox("Variable a predecir (Y)", cols_num, index=len(cols_num)-1)
    features = [c for c in cols_num if c != target]
    test_size = st.slider("Proporción de datos de prueba", 0.1, 0.4, 0.2, 0.05)

    df_model = df[features + [target]].copy()
    df_model.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_model.dropna(inplace=True)
    df_model = df_model.sample(min(50000, len(df_model)), random_state=42)

    if len(df_model) < 20:
        st.error("No hay suficientes datos limpios para entrenar el modelo.")
        st.stop()

    X = df_model[features].values.astype(float)
    y = df_model[target].values.astype(float)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42)

    scaler = StandardScaler()
    modelo = LinearRegression()
    modelo.fit(scaler.fit_transform(X_train), y_train)

    y_pred = modelo.predict(scaler.transform(X_test))
    r2  = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    m1, m2, m3 = st.columns(3)
    m1.metric("R² Score", f"{r2:.4f}", help="Entre 0 y 1. Más cercano a 1 es mejor.")
    m2.metric("MAE",      f"{mae:.2f}", help="Error promedio de predicción.")
    m3.metric("Datos de prueba", len(X_test))

    fig3, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))

    idx_s = np.random.choice(len(y_test), min(500, len(y_test)), replace=False)
    ax1.scatter(y_test[idx_s], y_pred[idx_s], alpha=0.6, color="#4C72B0",
                edgecolors="white", linewidths=0.3)
    mn, mx = float(y_test.min()), float(y_test.max())
    ax1.plot([mn, mx], [mn, mx], "r--", linewidth=2, label="Predicción perfecta")
    ax1.set_xlabel(f"Valor real de '{target}'")
    ax1.set_ylabel(f"Valor predicho de '{target}'")
    ax1.set_title(f"Real vs Predicho  (R² = {r2:.4f})")
    ax1.legend()
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    ax2.hist(y_test - y_pred, bins=25, color="#55A868", edgecolor="white")
    ax2.axvline(0, color="red", linestyle="--", linewidth=2)
    ax2.set_xlabel("Error de predicción")
    ax2.set_ylabel("Frecuencia")
    ax2.set_title("Distribución de errores")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    plt.tight_layout()
    st.pyplot(fig3)

    st.subheader("Predicción")
    st.caption("Ajusta los valores y obtén una predicción instantánea.")

    n_cols_ui = min(len(features), 4)
    cols_ui   = st.columns(n_cols_ui)
    valores   = {}
    for i, feat in enumerate(features):
        with cols_ui[i % n_cols_ui]:
            valores[feat] = st.number_input(
                feat,
                min_value=float(df_model[feat].min()),
                max_value=float(df_model[feat].max()),
                value=float(df_model[feat].mean()),
                format="%.2f",
            )

    X_nuevo = scaler.transform(pd.DataFrame([valores]).astype(float))
    pred    = modelo.predict(X_nuevo)[0]
    st.success(f"**Predicción de `{target}`:** `{pred:.2f}`")
