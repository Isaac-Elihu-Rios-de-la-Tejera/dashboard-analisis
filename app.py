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
import warnings
warnings.filterwarnings("ignore")

# ── Configuración de la página ─────────────────────────────────────
st.set_page_config(
    page_title="Dashboard ML",
    page_icon="📊",
    layout="wide",
)

# ── Encabezado ─────────────────────────────────────────────────────
st.title("📊 Dashboard de Análisis de Datos")
st.markdown("Sube un archivo CSV y obtén estadísticas, gráficas y predicciones automáticas.")
st.divider()

# ── Barra lateral: carga de archivo ───────────────────────────────
st.sidebar.header("⚙️ Configuración")
archivo = st.sidebar.file_uploader("Sube tu archivo CSV", type=["csv"])
st.sidebar.markdown("---")
st.sidebar.caption("Proyecto universitario · Python · Streamlit")

# ── Estado vacío ───────────────────────────────────────────────────
if archivo is None:
    st.info("👈  Sube un archivo CSV desde la barra lateral para comenzar.")

    # Mostrar dataset de ejemplo para que no quede vacío
    st.subheader("Dataset de ejemplo")
    np.random.seed(42)
    n = 120
    area        = np.random.randint(40, 180, n)
    habitaciones = np.round(area / 35 + np.random.normal(0, 0.4, n)).clip(1, 5).astype(int)
    antiguedad  = np.random.randint(0, 30, n)
    distancia   = np.round(np.random.uniform(0.5, 14, n), 1)
    precio      = (area*1200 + habitaciones*8000 - antiguedad*500
                   - distancia*3000 + np.random.normal(0, 8000, n)).round(-2).astype(int)
    df_demo = pd.DataFrame({
        "area_m2": area, "habitaciones": habitaciones,
        "antiguedad_anos": antiguedad, "distancia_km": distancia,
        "precio_usd": precio,
    })
    st.dataframe(df_demo.head(10), use_container_width=True)
    st.caption("Este es el formato esperado. Sube tu propio CSV para analizarlo.")
    st.stop()

# ── Carga y limpieza ───────────────────────────────────────────────
df = pd.read_csv(archivo)
df.dropna(how="all", inplace=True)
df.drop_duplicates(inplace=True)
cols_num = df.select_dtypes(include="number").columns.tolist()
for col in cols_num:
    df[col].fillna(df[col].median(), inplace=True)

if len(cols_num) < 2:
    st.error("El CSV necesita al menos 2 columnas numéricas.")
    st.stop()

# ── Pestañas ───────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🔍 Exploración", "📈 Visualizaciones", "🤖 Predicción"])

# ══════════════════════════════════════════════════════
# PESTAÑA 1 — Exploración
# ══════════════════════════════════════════════════════
with tab1:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Filas",              df.shape[0])
    c2.metric("Columnas",           df.shape[1])
    c3.metric("Columnas numéricas", len(cols_num))
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

    # Histograma
    axes[0].hist(df[col_sel], bins=25, color="#4C72B0", edgecolor="white")
    axes[0].axvline(df[col_sel].mean(), color="red", linestyle="--",
                    label=f"Media: {df[col_sel].mean():.2f}")
    axes[0].set_title(f"Distribución de '{col_sel}'")
    axes[0].set_xlabel(col_sel)
    axes[0].set_ylabel("Frecuencia")
    axes[0].legend()
    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)

    # Mapa de correlación
    corr = df[cols_num].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r",
                center=0, ax=axes[1], linewidths=0.5, square=True)
    axes[1].set_title("Correlación entre variables")

    plt.tight_layout()
    st.pyplot(fig)

    # Dispersión
    st.subheader("Dispersión entre dos variables")
    col_x = st.selectbox("Eje X", cols_num, index=0, key="x")
    col_y = st.selectbox("Eje Y", cols_num, index=len(cols_num)-1, key="y")

    fig2, ax = plt.subplots(figsize=(8, 4))
    ax.scatter(df[col_x], df[col_y], alpha=0.6, color="#DD8452",
               edgecolors="white", linewidths=0.3)
    m, b = np.polyfit(df[col_x], df[col_y], 1)
    x_l = np.linspace(df[col_x].min(), df[col_x].max(), 100)
    ax.plot(x_l, m*x_l + b, color="red", linewidth=2, label="Tendencia")
    ax.set_xlabel(col_x)
    ax.set_ylabel(col_y)
    ax.set_title(f"'{col_y}' vs '{col_x}'")
    ax.legend()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig2)

# ══════════════════════════════════════════════════════
# PESTAÑA 3 — Predicción
# ══════════════════════════════════════════════════════
with tab3:
    st.subheader("Configuración del modelo")

    target = st.selectbox("Variable a predecir (Y)", cols_num,
                          index=len(cols_num)-1)
    features = [c for c in cols_num if c != target]

    if not features:
        st.warning("Necesitas al menos 2 columnas numéricas.")
        st.stop()

    test_size = st.slider("Proporción de datos de prueba", 0.1, 0.4, 0.2, 0.05)

    # Entrenamiento automático
    X = df[features].values
    y = df[target].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42)

    scaler = StandardScaler()
    modelo = LinearRegression()
    modelo.fit(scaler.fit_transform(X_train), y_train)

    y_pred = modelo.predict(scaler.transform(X_test))
    r2  = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    # Métricas
    st.subheader("Métricas del modelo")
    m1, m2, m3 = st.columns(3)
    m1.metric("R² Score", f"{r2:.4f}", help="Entre 0 y 1. Más cercano a 1 es mejor.")
    m2.metric("MAE", f"{mae:.2f}",    help="Error promedio de predicción.")
    m3.metric("Datos de prueba", len(X_test))

    # Gráfica real vs predicho
    fig3, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))

    ax1.scatter(y_test, y_pred, alpha=0.6, color="#4C72B0",
                edgecolors="white", linewidths=0.3)
    mn, mx = float(y_test.min()), float(y_test.max())
    ax1.plot([mn, mx], [mn, mx], "r--", linewidth=2, label="Predicción perfecta")
    ax1.set_xlabel(f"Valor real de '{target}'")
    ax1.set_ylabel(f"Valor predicho de '{target}'")
    ax1.set_title(f"Real vs Predicho  (R² = {r2:.4f})")
    ax1.legend()
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    residuales = y_test - y_pred
    ax2.hist(residuales, bins=20, color="#55A868", edgecolor="white")
    ax2.axvline(0, color="red", linestyle="--", linewidth=2)
    ax2.set_xlabel("Error de predicción")
    ax2.set_ylabel("Frecuencia")
    ax2.set_title("Distribución de errores")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    plt.tight_layout()
    st.pyplot(fig3)

    # Panel de predicción manual
    st.subheader("🔮 Prueba tu predicción")
    st.caption("Ajusta los valores y obtén una predicción instantánea.")

    cols_input = st.columns(len(features))
    valores = {}
    for i, feat in enumerate(features):
        with cols_input[i]:
            valores[feat] = st.number_input(
                feat,
                min_value=float(df[feat].min()),
                max_value=float(df[feat].max()),
                value=float(df[feat].mean()),
                format="%.2f",
            )

    X_nuevo = scaler.transform(pd.DataFrame([valores]))
    pred = modelo.predict(X_nuevo)[0]
    st.success(f"**Predicción de `{target}`:** `{pred:.2f}`")
