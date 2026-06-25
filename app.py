from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Cargar pipeline guardado
modelo = joblib.load("modelo_zirconia.joblib")


@app.route("/", methods=["GET", "POST"])
def index():
    prediccion = None
    error = None

    if request.method == "POST":
        try:
            carat = float(request.form["carat"])
            cut = request.form["cut"]
            color = request.form["color"]
            clarity = request.form["clarity"]
            depth = float(request.form["depth"])
            table = float(request.form["table"])
            x = float(request.form["x"])
            y = float(request.form["y"])
            z = float(request.form["z"])

            # Validaciones
            if carat <= 0:
                raise ValueError("El peso en quilates debe ser mayor que cero.")

            if depth <= 0 or table <= 0:
                raise ValueError("Depth y table deben ser mayores que cero.")

            if x <= 0 or y <= 0 or z <= 0:
                raise ValueError("Las dimensiones x, y y z deben ser mayores que cero.")

            # Datos originales del usuario
            datos_usuario = pd.DataFrame([{
                "carat": carat,
                "cut": cut,
                "color": color,
                "clarity": clarity,
                "depth": depth,
                "table": table,
                "x": x,
                "y": y,
                "z": z
            }])

            # Predicción
            resultado = modelo.predict(datos_usuario)[0]
            prediccion = round(resultado, 2)

        except ValueError as e:
            error = str(e)

        except Exception:
            error = "Ocurrió un error al procesar los datos. Revisa que todos los campos sean válidos."

    return render_template("index.html", prediccion=prediccion, error=error)

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8000, debug=True)