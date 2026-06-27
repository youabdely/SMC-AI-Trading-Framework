import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt
import joblib
import os
from tensorflow.keras.utils import plot_model

# Parche para asegurar que Windows encuentre Graphviz
os.environ["PATH"] += os.pathsep + 'C:\\Program Files\\Graphviz\\bin'

# 1. CARGAR EL DATASET ORIGINAL
print("--- [FASE 1] Cargando datos de dataset_ia_oro.csv ---")
df = pd.read_csv("dataset_ia_oro.csv")

# Seleccionamos tus 7 características originales
features = ['mes', 'dia_semana', 'hora', 'tipo', 'rango_londres', 'ratio_mecha_sweep', 'distancia_sl']
X = df[features].values
y = df['resultado'].values  # 1 para TP, 0 para SL

# Escalado de datos
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# =====================================================================
# MODELO 1: RANDOM FOREST (Bosque Aleatorio - Enfoque Clásico)
# =====================================================================
print("\n--- [FASE 2] Entrenando Random Forest... ---")
X_train_rf, X_test_rf, y_train_rf, y_test_rf = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
rf_model.fit(X_train_rf, y_train_rf)

# Evaluamos Random Forest
y_pred_rf = rf_model.predict(X_test_rf)
acc_rf = accuracy_score(y_test_rf, y_pred_rf)
f1_rf = f1_score(y_test_rf, y_pred_rf)

joblib.dump(rf_model, "modelo_rf_oro.pkl")
print("✅ Random Forest guardado con éxito como 'modelo_rf_oro.pkl'")

# =====================================================================
# MODELO 2: LSTM (Red Neuronal Recurrente con Ventana Deslizante)
# =====================================================================
print("\n--- [FASE 3] Preparando secuencias temporales para la LSTM... ---")

# Función de Ventana Deslizante (Sliding Window) de 3 trades
def transformar_a_secuencias(data, target, window_size=3):
    X_seq, y_seq = [], []
    for i in range(len(data) - window_size):
        X_seq.append(data[i:(i + window_size)])
        y_seq.append(target[i + window_size])
    return np.array(X_seq), np.array(y_seq)

WINDOW_SIZE = 3
X_seq, y_seq = transformar_a_secuencias(X_scaled, y, window_size=WINDOW_SIZE)

# División respetando el orden cronológico
X_train_lstm, X_test_lstm, y_train_lstm, y_test_lstm = train_test_split(
    X_seq, y_seq, test_size=0.2, random_state=42, shuffle=False
)

# Definición de la arquitectura de la LSTM
lstm_model = tf.keras.Sequential([
    tf.keras.layers.LSTM(16, input_shape=(WINDOW_SIZE, len(features)), return_sequences=False),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(8, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

lstm_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

plot_model(
    lstm_model, 
    to_file='diagrama_lstm_arquitectura.png', 
    show_shapes=True, 
    show_layer_names=False, 
    show_layer_activations=True
)
print("✅ Diagrama de la LSTM guardado con éxito como 'diagrama_lstm_arquitectura.png'")

print("\n--- [FASE 4] Iniciando iteraciones (Épocas) de la Red LSTM ---")
history_lstm = lstm_model.fit(
    X_train_lstm, y_train_lstm, 
    epochs=50, 
    batch_size=8, 
    validation_data=(X_test_lstm, y_test_lstm), 
    verbose=1  # Te muestra las iteraciones por consola una a una
)

# Evaluamos la LSTM
y_pred_lstm_prob = lstm_model.predict(X_test_lstm)
y_pred_lstm = (y_pred_lstm_prob > 0.5).astype(int)
acc_lstm = accuracy_score(y_test_lstm, y_pred_lstm)
f1_lstm = f1_score(y_test_lstm, y_pred_lstm)

lstm_model.save("modelo_lstm_oro.keras")
print("\n✅ Red LSTM guardada con éxito como 'modelo_lstm_oro.keras'")

# =====================================================================
# GENERACIÓN AUTOMÁTICA DE LA GRÁFICA PARA EL TFG
# =====================================================================
print("\n--- [FASE 5] Generando gráfico de rendimiento... ---")
plt.figure(figsize=(12, 5))

# Gráfica de Precisión (Accuracy)
plt.subplot(1, 2, 1)
plt.plot(history_lstm.history['accuracy'], label='Entrenamiento (Train)')
plt.plot(history_lstm.history['val_accuracy'], label='Validación (Test)')
plt.title('Evolución de la Precisión (Accuracy)')
plt.xlabel('Época (Iteración)')
plt.ylabel('Precisión')
plt.legend()
plt.grid(True)

# Gráfica de Pérdida (Loss)
plt.subplot(1, 2, 2)
plt.plot(history_lstm.history['loss'], label='Entrenamiento (Train)')
plt.plot(history_lstm.history['val_loss'], label='Validación (Test)')
plt.title('Evolución de la Pérdida (Loss)')
plt.xlabel('Época (Iteración)')
plt.ylabel('Pérdida')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("curvas_aprendizaje.png")
print("📊 ¡Archivo 'curvas_aprendizaje.png' generado! Súbelo a Overleaf.")

# =====================================================================
# REPORTE DE CONSOLA
# =====================================================================
print("\n" + "="*60)
print("   MÉTRICAS COMPARATIVAS FINALES")
print("="*60)
print(f"Random Forest (Clásico)  -> Precisión: {acc_rf*100:.2f}% | F1-Score: {f1_rf:.4f}")
print(f"Red LSTM (Temporal N=3)  -> Precisión: {acc_lstm*100:.2f}% | F1-Score: {f1_lstm:.4f}")
print("="*60)