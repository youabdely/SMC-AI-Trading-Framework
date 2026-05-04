import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib # Para guardar el escalador

# 1. CARGAR EL DATASET DE LA FASE 1
df = pd.read_csv("dataset_ia_oro.csv")

# Seleccionamos las columnas que extrajimos
features = ['mes', 'dia_semana', 'hora', 'tipo', 'rango_londres', 'ratio_mecha_sweep', 'distancia_sl']
X = df[features].values
y = df['resultado'].values # 1 para TP, 0 para SL

# 2. ESCALADO DE DATOS (Fundamental para redes neuronales)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
joblib.dump(scaler, "escalador_oro.pkl") # Guardamos para usarlo en el bot luego

# 3. DIVISIÓN: 80% Entrenamiento, 20% Examen (Validación)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 4. DISEÑO DE LA RED NEURONAL
model = tf.keras.Sequential([
    # Capa de entrada con 32 neuronas
    tf.keras.layers.Dense(32, activation='relu', input_shape=(len(features),)),
    tf.keras.layers.Dropout(0.2), # Para evitar que la IA "memorice" los trades (overfitting)
    
    # Capa oculta
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dropout(0.1),
    
    # Capa de salida (Sigmoid devuelve un valor entre 0 y 1 -> Probabilidad)
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 5. ENTRENAMIENTO
print("\n--- Iniciando entrenamiento del cerebro del bot ---")
history = model.fit(X_train, y_train, epochs=100, batch_size=8, 
                    validation_data=(X_test, y_test), verbose=1)

# 6. GUARDAR EL MODELO
model.save("modelo_ia_oro.keras")
print("\nModelo guardado con éxito como 'modelo_ia_oro.keras'")

# Evaluación final
loss, accuracy = model.evaluate(X_test, y_test)
print(f"\nPrecisión en el examen final (Test Accuracy): {accuracy*100:.2f}%")