<div align="center">

# 🤖 SMC AI Trading Framework

### Framework de Trading Algorítmico impulsado por Smart Money Concepts e Inteligencia Artificial

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)]()
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)]()
[![MetaTrader5](https://img.shields.io/badge/MetaTrader5-API-green.svg)]()
[![Cython](https://img.shields.io/badge/Cython-Optimized-yellow.svg)]()
[![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)]()

*Framework modular de trading algorítmico que combina Smart Money Concepts (SMC) deterministas con Machine Learning para mejorar la selección de operaciones mediante filtrado probabilístico.*

</div>

---

## 📖 Resumen

SMC AI Trading Framework es una plataforma de trading algorítmico orientada a la investigación, desarrollada como Trabajo de Fin de Grado de Ingeniería Informática.

En lugar de intentar predecir directamente los precios del mercado, el framework transforma los **Smart Money Concepts (SMC)** en reglas matemáticas deterministas. Estas oportunidades de trading son posteriormente evaluadas por un modelo de Machine Learning que estima la probabilidad de éxito antes de su ejecución.

El proyecto combina:

- Trading basado en reglas deterministas
- Ingeniería de características (feature engineering)
- Clasificación mediante Machine Learning
- Backtesting histórico
- Evaluación de rendimiento
- Integración en vivo con MetaTrader 5

---

# ✨ Características

- 📈 Implementación de Smart Money Concepts
- 🧠 Filtrado de operaciones con IA
- ⚡ Motor de estrategia de alto rendimiento
- 🔬 Comparación de múltiples modelos de ML
- 📊 Motor de backtesting histórico
- 📉 Generación de curva de capital (equity curve)
- 📋 Informes de rendimiento profesionales
- 🛡 Sistema de gestión de riesgo
- ⚙ Integración con MetaTrader 5
- 🚀 Módulo de estrategia optimizado con Cython

---

# 🏗 Arquitectura del Sistema

```
                Datos Históricos / En Vivo
                         │
                         ▼
                Preprocesamiento de Datos
                         │
                         ▼
            Motor de Reglas Smart Money
                         │
              Oportunidad de Trading
                         │
                         ▼
          Pipeline de Ingeniería de Características
                         │
                         ▼
              Modelo de Machine Learning
                         │
            Estimación de Probabilidad
                         │
        ¿Probabilidad > Umbral de decisión?
                 │               │
               SÍ               NO
                 │               │
                 ▼               ▼
          Ejecutar Operación   Ignorar Señal
```


---

# 🧠 Machine Learning

Durante la investigación se evaluaron varios algoritmos de aprendizaje supervisado:

| Modelo | Propósito |
|--------|----------|
| Multi-Layer Perceptron (MLP) | Modelo final en producción |
| Random Forest | Modelo base de comparación |
| LSTM | Modelado secuencial |

La implementación final utiliza un **Multi-Layer Perceptron (MLP)** tras la evaluación comparativa, logrando el mejor equilibrio entre velocidad de inferencia, robustez y generalización.

---

# 📊 Lógica de Trading

El motor determinista sigue la metodología Smart Money Concepts:

1. Validación de Killzone  
2. Detección de pools de liquidez  
3. Identificación de barridos de liquidez  
4. Break of Structure (BOS)  
5. Extracción de características  
6. Estimación de probabilidad con IA  
7. Validación de riesgo  
8. Ejecución de la operación  

---

# 📂 Estructura del Proyecto

```
SMC-AI-TRADING-FRAMEWORK/
│
├── src/
│   ├── main.py
│   ├── backtest_engine.py
│   ├── backtest_engine_AI.py
│   ├── entrenar_ia.py
│   ├── train_comparativa.py
│   ├── puente_tfg.py
│   ├── utils.py
│   ├── strategy.pyd
│   ├── strategy.c
│   ├── setup.py
│   ├── modelos...
│   └── escalador...
│
├── data/
├── results/
├── README.md
```

---

# 🚀 Uso

### Entrenar el modelo

```bash
python entrenar_ia.py
```

### Ejecutar backtest histórico

```bash
python backtest_engine.py
```

### Iniciar trading en vivo

```bash
python puente_MetaTrader5.py
```

---

# 📊 Metodología de Investigación

El proyecto sigue un flujo completo de Machine Learning:

- Adquisición de datos históricos
- Ingeniería de características
- Generación del dataset
- Entrenamiento del modelo
- Optimización de hiperparámetros
- Validación In-Sample
- Validación Out-of-Sample
- Evaluación comparativa
- Despliegue en vivo

---

# 📈 Gestión de Riesgo

El framework incorpora un módulo de gestión de riesgo completamente automatizado:

- Riesgo fijo por operación en porcentaje
- Dimensionamiento dinámico de posiciones
- Cálculo automático de Stop Loss
- Cálculo automático de Take Profit
- Validación de Risk/Reward
- Máximo una operación por sesión

---

# ⚡ Optimización de Rendimiento

Los componentes críticos están compilados con **Cython**, proporcionando:

- Menor latencia
- Mayor velocidad de ejecución
- Protección del código fuente
- Despliegue listo para producción

---

# 🛠 Tecnologías

- Python
- NumPy
- Pandas
- TensorFlow
- Scikit-Learn
- MetaTrader5 API
- Matplotlib
- mplfinance
- Cython

---

# 🎓 Contexto Académico

Este proyecto fue desarrollado como Trabajo de Fin de Grado (TFG) de Ingeniería Informática.

**Título**

> Desarrollo e Implementación de una Plataforma de Trading Algorítmico Optimizada con Inteligencia Artificial

---

# ⚠ Disclaimer

Este repositorio está destinado exclusivamente a fines educativos y de investigación.

Operar en los mercados financieros implica un alto riesgo. El rendimiento pasado no garantiza resultados futuros.

---

# 👨‍💻 Autor

**Youssef Abderrahmani**

Ingeniería Informática

Inteligencia Artificial • Trading Algorítmico • Machine Learning

---

<div align="center">

⭐ Si te resulta interesante este proyecto, considera darle una estrella.

</div>
```
```