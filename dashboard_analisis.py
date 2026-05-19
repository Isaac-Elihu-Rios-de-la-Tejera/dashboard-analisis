# ================================================================
#  DASHBOARD DE ANÁLISIS DE DATOS CON CARGA DE ARCHIVO
#  Proyecto universitario - Python
#  Ejecutar en: Google Colab
# ================================================================

# ── CELDA 1: Importar librerías ──────────────────────────────────
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

plt.rcParams["figure.figsize"] = (10, 5)
plt.rcParams["axes.spines.top"]   = False
plt.rcParams["axes.spines.right"] = False
sns.set_palette("muted")

print("✅ Librerías cargadas correctamente")


# ── CELDA 2: Subir y cargar el archivo CSV ───────────────────────
# Aparecerá un botón "Elegir archivos" — selecciona tu CSV
from google.colab import files

print("📂 Selecciona tu archivo CSV:")
archivo_subido = files.upload()

# Leer el archivo subido
nombre_archivo = list(archivo_subido.keys())[0]
df = pd.read_csv(nombre_archivo)

# Limpieza básica automática
df.dropna(how="all", inplace=True)          # elimina filas totalmente vacías
df.drop_duplicates(inplace=True)            # elimina duplicados

# Rellenar nulos numéricos con la mediana de cada columna
cols_num = df.select_dtypes(include="number").columns
for col in cols_num:
    df[col].fillna(df[col].median(), inplace=True)

print(f"\n✅ Archivo cargado: '{nombre_archivo}'")
print(f"   {df.shape[0]} filas × {df.shape[1]} columnas")
print(f"   Columnas: {list(df.columns)}")
print(f"   Columnas numéricas: {list(cols_num)}")


# ── CELDA 3: Estadísticas descriptivas ───────────────────────────
print("\n" + "═"*60)
print("  ESTADÍSTICAS DESCRIPTIVAS")
print("═"*60)
print(df.describe().round(2).to_string())
print(f"\n• Total de registros : {len(df)}")
print(f"• Valores nulos      : {df.isnull().sum().sum()}")
print(f"• Tipos de datos:\n{df.dtypes.to_string()}")


# ── CELDA 4: Detectar columnas y pedir configuración ─────────────
cols_num = df.select_dtypes(include="number").columns.tolist()

print("\n" + "═"*60)
print("  COLUMNAS NUMÉRICAS DISPONIBLES:")
print("═"*60)
for i, col in enumerate(cols_num):
    print(f"  [{i}] {col}")

# ── Elige aquí la columna que quieres predecir (columna objetivo) ──
# Cambia el índice por el número de la columna que quieras predecir
INDICE_TARGET = len(cols_num) - 1   # por defecto: la última columna numérica

target   = cols_num[INDICE_TARGET]
features = [c for c in cols_num if c != target]

print(f"\n🎯 Variable objetivo (Y): '{target}'")
print(f"📊 Variables predictoras (X): {features}")
print("\n💡 Para cambiar la variable objetivo, edita INDICE_TARGET en esta celda.")


# ── CELDA 5: Visualizaciones automáticas ─────────────────────────
n_features = len(features)

# Layout dinámico según cuántas columnas tenga el CSV
n_graficas = min(n_features, 4) + 2   # dispersiones + histograma target + correlación
cols_layout = 3
filas_layout = int(np.ceil(n_graficas / cols_layout))

fig, axes = plt.subplots(filas_layout, cols_layout,
                         figsize=(16, 5 * filas_layout))
axes = axes.flatten()
fig.suptitle(f"Dashboard de Análisis — {nombre_archivo}",
             fontsize=15, fontweight="bold", y=1.01)

idx = 0  # índice de la siguiente gráfica disponible

# 1. Distribución de la variable objetivo
axes[idx].hist(df[target], bins=25, color="#4C72B0", edgecolor="white")
axes[idx].axvline(df[target].mean(), color="red", linestyle="--",
                  label=f"Media: {df[target].mean():.2f}")
axes[idx].set_title(f"Distribución de '{target}'")
axes[idx].set_xlabel(target)
axes[idx].set_ylabel("Frecuencia")
axes[idx].legend()
idx += 1

# 2. Dispersión target vs cada feature (máximo 4)
for feat in features[:4]:
    axes[idx].scatter(df[feat], df[target],
                      alpha=0.55, color="#DD8452",
                      edgecolors="white", linewidths=0.3)
    m, b = np.polyfit(df[feat], df[target], 1)
    x_l = np.linspace(df[feat].min(), df[feat].max(), 100)
    axes[idx].plot(x_l, m*x_l + b, color="red", linewidth=2, label="Tendencia")
    axes[idx].set_title(f"'{target}' vs '{feat}'")
    axes[idx].set_xlabel(feat)
    axes[idx].set_ylabel(target)
    axes[idx].legend()
    idx += 1

# 3. Mapa de correlación (ocupa el espacio restante)
if idx < len(axes):
    corr = df[cols_num].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r",
                center=0, ax=axes[idx], linewidths=0.5, square=True)
    axes[idx].set_title("Correlación entre Variables")
    idx += 1

# Ocultar ejes vacíos
for j in range(idx, len(axes)):
    axes[j].set_visible(False)

plt.tight_layout()
plt.savefig("dashboard_visualizaciones.png", dpi=150, bbox_inches="tight")
plt.show()
print("✅ Gráficas guardadas como 'dashboard_visualizaciones.png'")


# ── CELDA 6: Modelo de regresión lineal ──────────────────────────
print("\n" + "═"*60)
print("  MODELO DE PREDICCIÓN — REGRESIÓN LINEAL")
print("═"*60)

X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

modelo = LinearRegression()
modelo.fit(X_train_s, y_train)

y_pred = modelo.predict(X_test_s)
r2  = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print(f"\n  Registros de entrenamiento : {len(X_train)}")
print(f"  Registros de prueba        : {len(X_test)}")
print(f"\n  R² Score : {r2:.4f}   (entre 0 y 1, más cercano a 1 es mejor)")
print(f"  MAE      : {mae:.4f}   (error promedio en las predicciones)")

print("\n  Influencia de cada variable en la predicción:")
for feat, coef in sorted(zip(features, modelo.coef_),
                          key=lambda x: abs(x[1]), reverse=True):
    direccion = "↑ sube" if coef > 0 else "↓ baja"
    print(f"    {feat:<25} {direccion} el valor  (coef: {coef:>10.4f})")


# ── CELDA 7: Gráfica del modelo ───────────────────────────────────
fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
fig2.suptitle("Evaluación del Modelo de Regresión Lineal",
              fontsize=14, fontweight="bold")

ax1.scatter(y_test, y_pred, alpha=0.6, color="#4C72B0",
            edgecolors="white", linewidths=0.3)
mn, mx = float(y_test.min()), float(y_test.max())
ax1.plot([mn, mx], [mn, mx], "r--", linewidth=2, label="Predicción perfecta")
ax1.set_xlabel(f"Valor Real de '{target}'")
ax1.set_ylabel(f"Valor Predicho de '{target}'")
ax1.set_title(f"Real vs Predicho  (R² = {r2:.4f})")
ax1.legend()

residuales = y_test - y_pred
ax2.hist(residuales, bins=25, color="#DD8452", edgecolor="white")
ax2.axvline(0, color="red", linestyle="--", linewidth=2, label="Error = 0")
ax2.set_xlabel("Error de Predicción (Residual)")
ax2.set_ylabel("Frecuencia")
ax2.set_title("Distribución de Errores del Modelo")
ax2.legend()

plt.tight_layout()
plt.savefig("dashboard_modelo.png", dpi=150, bbox_inches="tight")
plt.show()
print("✅ Gráfica del modelo guardada como 'dashboard_modelo.png'")


# ── CELDA 8: Predicción con valores personalizados ────────────────
print("\n" + "═"*60)
print("  PREDICCIÓN CON VALORES PERSONALIZADOS")
print("═"*60)
print("  Edita el diccionario 'mi_caso' con los valores que quieras:\n")

# ── EDITA AQUÍ los valores para tu predicción ──
mi_caso = {feat: float(df[feat].mean()) for feat in features}
# Por defecto usa el promedio de cada columna.
# Ejemplo: si tienes columnas "area" y "habitaciones" puedes escribir:
# mi_caso = {"area": 90, "habitaciones": 3}

print(f"  Valores ingresados: {mi_caso}")

X_nuevo  = scaler.transform(pd.DataFrame([mi_caso]))
pred_val = modelo.predict(X_nuevo)[0]

print(f"\n  🎯 Predicción de '{target}': {pred_val:.4f}")
print("\n✅ Análisis completo. Revisa las imágenes en el panel de archivos de Colab.")
