from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

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

            if carat <= 0:
                raise ValueError("El peso en quilates debe ser mayor que cero.")

            if depth <= 0 or table <= 0:
                raise ValueError("Depth y table deben ser mayores que cero.")

            if x <= 0 or y <= 0 or z <= 0:
                raise ValueError("Las dimensiones x, y y z deben ser mayores que cero.")

            # Orden exacto:
            # carat, cut, color, clarity, depth, table, x, y, z
            datos_usuario = np.array([[
                carat,
                cut,
                color,
                clarity,
                depth,
                table,
                x,
                y,
                z
            ]], dtype=object)

            resultado = modelo.predict(datos_usuario)[0]
            prediccion = round(float(resultado), 2)

        except ValueError as e:
            error = str(e)

        except Exception as e:
            error = "Ocurrió un error al procesar los datos. Revisa que todos los campos sean válidos."

    return render_template("index.html", prediccion=prediccion, error=error)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8000, debug=True)