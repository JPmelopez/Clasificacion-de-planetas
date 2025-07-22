import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import joblib

def entrenar_modelo():
    try:
        df = pd.read_csv('dataset_cuerpos_celestes.csv')
    except:
        # Dataset de ejemplo si no se encuentra el archivo
        data = {
            'Fe': [12.5, 8.2, 3.1, 25.7, 5.4, 18.9, 30.2, 2.8, 15.0, 7.5],
            'Si': [15.3, 22.1, 18.7, 10.5, 28.9, 12.3, 8.7, 20.4, 14.2, 25.0],
            'Mg': [8.7, 12.5, 15.2, 5.3, 18.7, 7.9, 3.5, 22.1, 10.5, 16.8],
            'Ni': [3.2, 1.8, 0.5, 7.9, 0.9, 5.4, 9.1, 0.3, 4.2, 1.2],
            'H2O': [0.1, 2.5, 8.7, 0.0, 5.3, 0.2, 0.0, 10.2, 3.5, 7.8],
            'tipo': ['Meteorito', 'Asteroide', 'Cometa', 'Meteorito', 'Cometa', 
                    'Meteorito', 'Meteorito', 'Cometa', 'Asteroide', 'Cometa']
        }
        df = pd.DataFrame(data)
        df.to_csv('dataset_cuerpos_celestes.csv', index=False)

    # Preprocesamiento
    elementos = df.columns[:-1].tolist()
    X = df[elementos].values
    y = df['tipo'].values

    # Codificar etiquetas
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Escalar características
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Crear modelo
    num_classes = len(label_encoder.classes_)
    model = Sequential([
        Dense(64, activation='relu', input_shape=(X_scaled.shape[1],)),
        Dense(32, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

    # Entrenar modelo
    model.fit(X_scaled, y_encoded, epochs=50, batch_size=8, verbose=0)

    # Guardar modelo y preprocesadores
    joblib.dump(model, 'modelo_cuerpos_celestes.joblib')
    joblib.dump(scaler, 'scaler_cuerpos_celestes.joblib')
    joblib.dump(label_encoder, 'label_encoder_cuerpos_celestes.joblib')

    return model
