import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor

# Cargar dataset
dataset = pd.read_csv("cubic_zirconia.csv")

# Eliminar identificador si existe
dataset = dataset.drop(columns=["Unnamed: 0"], errors="ignore")

# Separar variables
X = dataset.drop(columns=["price"])
y = dataset["price"]

# Características seleccionadas del mejor modelo
columnas_numericas = ["carat", "y"]
columnas_categoricas = ["clarity", "color"]

# Orden de variables categóricas
orden_clarity = ["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"]
orden_color = ["J", "I", "H", "G", "F", "E", "D"]

# Pipeline numérico
pipeline_numerico = Pipeline(steps=[
    ("imputador", SimpleImputer(strategy="median")),
    ("escalador", RobustScaler())
])

# Pipeline categórico
pipeline_categorico = Pipeline(steps=[
    ("imputador", SimpleImputer(strategy="most_frequent")),
    ("codificador", OrdinalEncoder(
        categories=[orden_clarity, orden_color],
        handle_unknown="use_encoded_value",
        unknown_value=-1
    ))
])

# Preprocesador
preprocesador = ColumnTransformer(
    transformers=[
        ("numericas", pipeline_numerico, columnas_numericas),
        ("categoricas", pipeline_categorico, columnas_categoricas)
    ],
    remainder="drop"
)

# Modelo final seleccionado
modelo = RandomForestRegressor(
    n_estimators=60,
    max_depth=16,
    min_samples_leaf=3,
    random_state=42,
    n_jobs=1
)
# Pipeline completo
pipeline_final = Pipeline(steps=[
    ("preprocesamiento", preprocesador),
    ("modelo", modelo)
])

# Entrenar con todos los datos disponibles
pipeline_final.fit(X, y)

# Guardar modelo
joblib.dump(pipeline_final, "modelo_zirconia.joblib", compress=9)

print("Modelo guardado correctamente como modelo_zirconia.joblib")