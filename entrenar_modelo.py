import csv
import joblib
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor


# =========================
# 1. Cargar dataset sin pandas
# =========================

X = []
y = []

with open("cubic_zirconia.csv", "r", encoding="utf-8") as archivo:
    lector = csv.DictReader(archivo)

    for fila in lector:
        try:
            # Variables originales necesarias
            carat = float(fila["carat"])
            cut = fila["cut"]
            color = fila["color"]
            clarity = fila["clarity"]

            # Si depth viene vacío, lo dejamos como np.nan para imputación
            depth = float(fila["depth"]) if fila["depth"] != "" else np.nan

            table = float(fila["table"])
            x_val = float(fila["x"])
            y_val = float(fila["y"])
            z_val = float(fila["z"])
            price = float(fila["price"])

            X.append([
                carat,
                cut,
                color,
                clarity,
                depth,
                table,
                x_val,
                y_val,
                z_val
            ])

            y.append(price)

        except Exception:
            # Ignorar filas con errores graves de lectura
            continue

X = np.array(X, dtype=object)
y = np.array(y, dtype=float)


# =========================
# 2. Índices de columnas
# =========================
# Orden de X:
# 0 carat
# 1 cut
# 2 color
# 3 clarity
# 4 depth
# 5 table
# 6 x
# 7 y
# 8 z

# Características seleccionadas del mejor modelo:
# carat, y, clarity, color
columnas_numericas = [0, 7]      # carat, y
columnas_categoricas = [3, 2]    # clarity, color


# =========================
# 3. Orden de categorías
# =========================

orden_clarity = ["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"]
orden_color = ["J", "I", "H", "G", "F", "E", "D"]


# =========================
# 4. Pipeline
# =========================

pipeline_numerico = Pipeline(steps=[
    ("imputador", SimpleImputer(strategy="median")),
    ("escalador", RobustScaler())
])

pipeline_categorico = Pipeline(steps=[
    ("imputador", SimpleImputer(strategy="most_frequent")),
    ("codificador", OrdinalEncoder(
        categories=[orden_clarity, orden_color],
        handle_unknown="use_encoded_value",
        unknown_value=-1
    ))
])

preprocesador = ColumnTransformer(
    transformers=[
        ("numericas", pipeline_numerico, columnas_numericas),
        ("categoricas", pipeline_categorico, columnas_categoricas)
    ],
    remainder="drop"
)


# =========================
# 5. Modelo ligero para Render
# =========================

modelo = RandomForestRegressor(
    n_estimators=15,
    max_depth=8,
    min_samples_leaf=10,
    max_features="sqrt",
    random_state=42,
    n_jobs=1
)


pipeline_final = Pipeline(steps=[
    ("preprocesamiento", preprocesador),
    ("modelo", modelo)
])


# =========================
# 6. Entrenar y guardar
# =========================

pipeline_final.fit(X, y)

joblib.dump(pipeline_final, "modelo_zirconia.joblib", compress=9)

print("Modelo guardado correctamente como modelo_zirconia.joblib")
print("Registros usados:", len(X))