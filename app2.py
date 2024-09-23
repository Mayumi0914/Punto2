from flask import Flask, request, jsonify
from pydantic import BaseModel
import json
import pandas as pd
import pickle
from typing import Optional
from pycaret.regression import predict_model

# Crear una instancia de Flask
app = Flask(__name__)

@app.route("/",methods=["GET"])
def home():
    return "Predicciones Precio"

# Definir el archivo JSON donde se guardarán las predicciones
file_name = 'predicciones2.json'

# Cargar el modelo preentrenado desde el archivo pickle

model_path = "dt.pkl"
with open(model_path, 'rb') as model_file:
    dt2 = pickle.load(model_file)
prueba = pd.read_csv("prueba_APP.csv",header = 0,sep=";",decimal=",")

# Definir las covariables que necesitas en el modelo
cuantitativas = ['Avg. Session Length',
 'Time on App',
 'Time on Website',
 'Length of Membership'
]
categoricas = ['Address',
 'dominio',
 'Tec'
]

# Modelo de datos para la API (simplificado, adaptado según tus variables)
class InputData(BaseModel):
    Email: str  # ID del usuario ( no es covariable)
    Address: str
    dominio: str
    Tec: str
    Avg_Session_Length: float
    Time_on_App: float
    Time_on_Website: float
    Length_of_Membership: float
    price: float
# Función para guardar predicciones en un archivo JSON
def save_prediction(prediction_data):
    try:
        with open(file_name, 'r') as file:
            predictions = json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        predictions = []

    predictions.append(prediction_data)

    with open(file_name, 'w') as file:
        json.dump(predictions, file, indent=4)

# Endpoint para realizar la predicción
@app.route('/predict', methods=['POST'])
def predict():
    try: 
        data = request.json  # Recibir los datos en formato JSON
        # Crear DataFrame a partir de los datos de entrada
        user_data = pd.DataFrame([data])

        # Asegurar que las columnas del DataFrame "user_data" coincidan con las de "prueba"
        user_data.columns = prueba.columns

        # Concatenar los datos del usuario con los datos del CSV "prueba"
        prueba2 = pd.concat([user_data, prueba], axis=0)
        prueba2.index = range(prueba2.shape[0])
    
        df_test = prueba2.copy()
        predictions = predict_model(dt2, data=df_test)
        prediction_label = predictions.iloc[0,]["prediction_label"]

        # Guardar predicción con ID en el archivo JSON
        prediction_result = {"Email": data.Email, "prediction": prediction_label}
        save_prediction(prediction_result)

        return jsonify(prediction_result)
    except Exception as e:
        # Capturar el error y mostrar detalles en la consola
        print(f"Error: {str(e)}")  
        
        # Retornar un mensaje de error en el formato de respuesta JSON
        return jsonify({"error": str(e)}), 500
# Ejecutar la aplicación si se llama desde la terminal
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
