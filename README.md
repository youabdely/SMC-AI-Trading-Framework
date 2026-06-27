# Hybrid Algorithmic System: SMC Logic & Machine Learning Validation

Este proyecto de **Trabajo de Fin de Grado (TFG)** presenta el diseño, implementación y validación de un framework autónomo de trading algorítmico de alta precisión para el par **XAU/USD (Oro)**. El sistema digitaliza y automatiza la metodología de análisis técnico avanzado **SMC (Smart Money Concepts)**, optimizando la toma de decisiones mediante un supervisor probabilístico basado en **Aprendizaje Profundo (Deep Learning)**.

---

## 🎯 Visión General
El objetivo principal es la creación de un sistema analítico *end-to-end* capaz de procesar volúmenes masivos de datos históricos y flujos en tiempo real. El framework elimina el factor emocional y los sesgos cognitivos en entornos competitivos mediante una ejecución estrictamente determinista y validada estadísticamente sobre un extenso histórico continuo.

---

## 🚀 Guía de Inicio Rápido (Despliegue y Ejecución)

Siga estos pasos para compilar los módulos nativos y ejecutar el sistema en su entorno local:

### 1. Instalación de dependencias
Se requiere **Python 3.10** o superior. Instale el entorno de librerías necesarias ejecutando:
```bash
pip install pandas numpy tensorflow scikit-learn mplfinance matplotlib pytz
2. Compilación de Módulos Nativos (Optimización en C)
Para mitigar la sobrecarga (overhead) de Python y competir en latencia en la microestructura del mercado, el motor de reglas heurísticas crítico se encuentra optimizado en C puro. Compile el módulo nativo mediante Cython ejecutando:

Bash
python src/setup.py build_ext --inplace
3. Ejecución del Core y Backtesting Estadístico
Para procesar el motor de reglas y analizar la operativa detallada con el pipeline de validación inteligente:

Bash
python src/main.py
Nota: El sistema modular permite conmutar entre el motor determinista estándar (backtest_engine.py) y el motor optimizado supervisado por el Perceptrón Multicapa (backtest_engine_AI.py).

⚙️ Arquitectura Técnica
1. Ingesta y ETL (Extract, Transform, Load)
Alta Resolución Temporal: Procesamiento de series temporales en resolución de 1 minuto (1m) sobre un histórico continuo (2018-2025) con costes reales de mercado (spread dinámico, comisiones y slippage).

Pipeline Multitemporal: Sistema de resampling dinámico acoplado para el análisis de estructuras macro (generación de velas de 5m a partir de datos crudos de 1m).

Segmentación Operativa: Motor de filtrado especializado en ventanas de alta volatilidad y liquidez institucional (Killzones de la sesión de Nueva York).

2. Motor de Reglas Heurísticas (SMC Core)
Digitalización de patrones geométricos de microestructura mediante algoritmos deterministas:

Identificación de Zonas: Detección automatizada de Order Blocks (OB) y Fair Value Gaps (FVG).

Cálculo de Niveles: Determinación en tiempo real de zonas de Equilibrium (Premium/Discount) y barridos de liquidez.

Compilación Nativa: El núcleo geométrico (strategy.c) se compila anticipadamente para garantizar el máximo rendimiento del ciclo de reloj del procesador.

3. Capa de Inteligencia Artificial (Filtro Supervisor)
Gatekeeper Probabilístico: Integración de un clasificador denso basado en un Perceptrón Multicapa (MLP) y modelos comparativos regularizados frente a la degradación de redes LSTM.

Mitigación de Overfitting: El modelo evalúa el contexto vectorial del mercado estrictamente en el momento del setup geométrico, bloqueando las operaciones de baja esperanza matemática y optimizando drásticamente la curva de equidad final out-of-sample.

4. Subsistema de Enlace en Tiempo Real (Live Bridge)
Integración MetaTrader 5 (MT5): Conector bidireccional mediante el API nativo para la captura de flujos masivos de ticks y ejecución de órdenes.

Sincronización Dinámica: Control de husos horarios (pytz) alineado con el servidor de Nueva York para la ejecución síncrona de las ventanas operativas.

🛠️ Roadmap de Implementación (Hitos Completados)
[x] Fase 1: Prototipo funcional de ingesta de datos y motor de simulación.

[x] Fase 2: Modelado, abstracción matemática y compilación del motor de reglas SMC (FVG, OB, Liquidez).

[x] Fase 3: Diseño, entrenamiento robusto y validación cruzada temporal del supervisor MLP (IA).

[x] Fase 4: Integración completa con el Live Bridge bidireccional para MetaTrader 5.

[x] Fase 5: Validación ciega continua (Out-of-Sample 2018-2025) y consolidación de métricas financieras (Ratio de Sharpe, Drawdown, Profit Factor).

💻 Stack Tecnológico
Lenguaje Core: Python 3.10+ (Optimizado con Cython y C nativo).

Ciencia de Datos: Pandas, NumPy.

Marcos de Deep Learning: TensorFlow / Keras, Scikit-learn.

Visualización Avanzada: Mplfinance, Matplotlib.

Autor: Youssef Abderrahmani

Tutoría: Departamento de Lenguajes y Sistemas Informáticos (LSI) — Escuela Técnica Superior de Ingeniería Informática (ETSII), Universidad de Sevilla.