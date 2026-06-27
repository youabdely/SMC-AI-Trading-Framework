import os
# Corrección del parche de Graphviz para Windows (con os.pathsep correcto)
os.environ["PATH"] += os.pathsep + 'C:\\Program Files\\Graphviz\\bin'
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt  # <--- NUEVA LIBRERÍA PARA OBTENER LA GRÁFICA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib 
from tensorflow.keras.utils import plot_model

# 1. CARGAR EL DATASET DE LA FASE 1
df = pd.read_csv("dataset_ia_oro.csv")

features = ['mes', 'dia_semana', 'hora', 'tipo', 'rango_londres', 'ratio_mecha_sweep', 'distancia_sl']
X = df[features].values
y = df['resultado'].values 

# 2. ESCALADO DE DATOS
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
joblib.dump(scaler, "escalador_oro.pkl") 

# 3. DIVISIÓN: 80% Entrenamiento, 20% Examen (Validación)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 4. DISEÑO DE LA RED NEURONAL
model = tf.keras.Sequential([
    tf.keras.layers.Dense(32, activation='relu', input_shape=(len(features),)),
    tf.keras.layers.Dropout(0.2), 
    
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dropout(0.1),
    
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Genera el diagrama automático del MLP
plot_model(
    model, 
    to_file='diagrama_mlp_arquitectura.png', 
    show_shapes=True, 
    show_layer_names=False, 
    show_layer_activations=True
)
print("📊 Diagrama del MLP generado como 'diagrama_mlp_arquitectura.png'")

# =====================================================================
# MODIFICACIÓN RECOMENDADA: Early Stopping (Opcional pero ideal)
# Si quieres que la gráfica no rompa por overfitting, descomenta las 2 líneas de abajo
# y añade 'callbacks=[early_stop]' dentro del model.fit()
# early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)
# =====================================================================
# =====================================================================
# MODIFICACIÓN RECOMENDADA: Early Stopping (¡ACTIVADO!)
# =====================================================================
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', 
    patience=5,                  # Espera 5 épocas tras el mínimo para confirmar el amago de subida
    restore_best_weights=True    # ¡CLAVE! Recupera automáticamente los pesos de la época 7
)

# 5. ENTRENAMIENTO (Cambiamos a 50 épocas para calcar tu gráfica actual)
print("\n--- Iniciando entrenamiento del cerebro del bot ---")
history = model.fit(X_train, y_train, epochs=50, batch_size=8, 
                    validation_data=(X_test, y_test),callbacks=[early_stop], verbose=1)

# 6. GUARDAR EL MODELO
model.save("modelo_ia_oro.keras")
print("\nModelo guardado con éxito como 'modelo_ia_oro.keras'")

# Evaluación final
loss, accuracy = model.evaluate(X_test, y_test)
print(f"\nPrecisión en el examen final (Test Accuracy): {accuracy*100:.2f}%")


# =====================================================================
# 📊 BLOQUE NUEVO: GENERACIÓN Y GUARDADO DE LAS GRÁFICAS PARA EL TFG
# =====================================================================
print("\n📈 Generando curvas de aprendizaje para la memoria...")

# Creamos un lienzo con dos subgráficas de lado a lado
plt.figure(figsize=(14, 6))

# Subgráfica 1: Evolución de la Precisión (Accuracy)
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Entrenamiento (Train)', color='#1f77b4', linewidth=2)
plt.plot(history.history['val_accuracy'], label='Validación (Test)', color='#ff7f0e', linewidth=2)
plt.title('Evolución de la Precisión (Accuracy)', fontsize=14, pad=10)
plt.xlabel('Época (Iteración)', fontsize=12)
plt.ylabel('Precisión', fontsize=12)
plt.grid(True, linestyle='-', alpha=0.7)
plt.legend(loc='upper left', fontsize=11)

# Subgráfica 2: Evolución de la Pérdida (Loss)
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Entrenamiento (Train)', color='#1f77b4', linewidth=2)
plt.plot(history.history['val_loss'], label='Validación (Test)', color='#ff7f0e', linewidth=2)
plt.title('Evolución de la Pérdida (Loss)', fontsize=14, pad=10)
plt.xlabel('Época (Iteración)', fontsize=12)
plt.ylabel('Pérdida', fontsize=12)
plt.grid(True, linestyle='-', alpha=0.7)
plt.legend(loc='upper left', fontsize=11)

# Ajustar márgenes para que no se pisen los textos
plt.tight_layout()

# Guardar la gráfica en alta definición lista para Overleaf
plt.savefig('curva_entrenamiento_mlp.png', dpi=300)
print("✅ Gráfica exportada con éxito como 'curva_entrenamiento_mlp.png'")

# Mostrar en pantalla por si estás ejecutando en local
plt.show()