from flask import Flask, render_template, request, redirect, url_for
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import joblib
import os

app = Flask(__name__)

# Cargar o crear modelo
def cargar_modelo():
    if not os.path.exists('modelo_cuerpos_celestes.joblib'):
        from model import entrenar_modelo
        entrenar_modelo()
    
    modelo = joblib.load('modelo_cuerpos_celestes.joblib')
    return modelo

modelo = cargar_modelo()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predecir', methods=['POST'])
def predecir():
    try:
        # Obtener datos del formulario
        datos = {
            'Fe': float(request.form['Fe']),
            'Si': float(request.form['Si']),
            'Mg': float(request.form['Mg']),
            'Ni': float(request.form['Ni']),
            'H2O': float(request.form['H2O'])
        }
        
        # Convertir a array y escalar
        composicion = np.array([[datos['Fe'], datos['Si'], datos['Mg'], datos['Ni'], datos['H2O']]]).reshape(1, -1)
        scaler = joblib.load('scaler_cuerpos_celestes.joblib')
        composicion_escalada = scaler.transform(composicion)
        
        # Predecir
        probabilidades = modelo.predict(composicion_escalada)[0]
        label_encoder = joblib.load('label_encoder_cuerpos_celestes.joblib')
        
        # Preparar resultados
        resultados = []
        for i, clase in enumerate(label_encoder.classes_):
            resultados.append({
                'clase': clase,
                'probabilidad': f"{probabilidades[i]*100:.2f}%"
            })
        
        clase_predicha = label_encoder.classes_[np.argmax(probabilidades)]
        
        return render_template('resultados.html', 
                            datos=datos,
                            resultados=resultados,
                            clase_predicha=clase_predicha)
    
    except Exception as e:
        return render_template('index.html', error=str(e))

@app.route('/elementos')
def mostrar_elementos():
    return render_template('elementos.html')

if __name__ == '__main__':
    app.run(debug=True)
