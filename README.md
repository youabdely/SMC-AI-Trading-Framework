# Hybrid Algorithmic System: SMC Logic & Machine Learning Validation

Este proyecto de **Trabajo de Fin de Grado (TFG)** presenta el diseño y desarrollo de un framework de trading algorítmico de alta precisión para el par **XAU/USD (Oro)**. El sistema combina la metodología de análisis institucional **SMC (Smart Money Concepts)** con modelos de **Aprendizaje Supervisado** para la optimización de la toma de decisiones.

---
 
## 🎯 Visión General
El objetivo principal es la creación de un sistema *end-to-end* capaz de procesar volúmenes masivos de datos históricos y en tiempo real, eliminando el factor emocional y los sesgos cognitivos mediante una ejecución estrictamente determinista y validada estadísticamente.

---

## 🚀 Guía de Inicio Rápido (Para el Tutor)

Siga estos pasos para replicar los resultados del sistema en su entorno local:

### 1. Instalación de dependencias
Se requiere **Python 3.10** o superior. Instale las librerías necesarias ejecutando:
```bash
pip install pandas mplfinance matplotlib numpy
2. Ejecución del Backtest Individual (Visualización de Trades)
Para generar reportes gráficos y analizar la operativa detallada de un año específico:

Bash
python src/main.py
Los resultados se almacenarán automáticamente en la carpeta /trades_individual.

3. Ejecución del Backtest Masivo (Estadísticas 2015-2025)
Para procesar el histórico completo de 11 años y generar la tabla de métricas global en consola:

Bash
python src/ultimate_backtest.py
⚙️ Arquitectura Técnica
1. Ingesta y ETL (Extract, Transform, Load)
Alta Resolución: Procesamiento de series temporales en resolución de 1 minuto (1m) con un histórico extenso (2015-2025).

Pipeline de Datos: Sistema de resampling dinámico para análisis multitemporal (generación de velas de 5m a partir de 1m).

Segmentación Operativa: Motor de filtrado horario especializado en la ventana de volatilidad de la sesión de Nueva York (09:30 - 10:30).

2. Motor de Reglas Heurísticas (SMC Core)
Digitalización de patrones de microestructura de mercado mediante algoritmos de reconocimiento geométrico:

Identificación de Zonas: Detección de Order Blocks (OB) y Fair Value Gaps (FVG).

Cálculo de Niveles: Determinación automática de zonas de Equilibrium y niveles de liquidez institucional.

Lógica de Setup: Validación de condiciones de entrada basadas en el cumplimiento estricto de la estrategia programada.

Optimización: El núcleo de la estrategia ha sido compilado en .pyd (Cython) para garantizar alto rendimiento y proteger la propiedad intelectual.

3. Capa de Inteligencia Artificial (En Desarrollo)
Clasificador de Calidad: Modelo de Machine Learning (Clasificación) encargado de actuar como filtro supervisor.

Objetivo: Analizar variables del entorno en el momento del setup para predecir la probabilidad de éxito, permitiendo descartar operaciones de baja esperanza matemática.

4. Subsistema de Análisis y Backtesting
Generación automática de indicadores clave de rendimiento:

Métricas Financieras: Profit Factor, Drawdown máximo, Esperanza Matemática y Ratio de Sharpe.

Reportes Visuales:

/trades_ganadores: Documentación visual de ejecuciones exitosas con niveles de entrada y salida.

/trades_perdedores: Análisis de fallos para el refinamiento del modelo (Debug).


## 🌉 Subsistema de Conexión en Tiempo Real (Live Bridge)
Se ha implementado una capa de enlace directo con el mercado para la transición del backtest a la operativa real.

- **Integración con MetaTrader 5 (MT5):** Conector bidireccional que permite la captura de flujos de datos OHLCV en tiempo real para el par XAU/USD.
- **Reloj de Precisión de Sesiones:** Sincronización automática con la zona horaria de Nueva York (`pytz`) para la activación de Killzones operativas, independientemente de la ubicación del servidor.
- **Buffer de Datos Dinámico:** Sistema de alimentación que transforma los datos crudos del broker en DataFrames compatibles con el motor de reglas de alta velocidad.

🛠 Roadmap de Implementación
[x] Fase 1: Prototipo funcional de ingesta de datos y motor de backtesting.

[ ] Fase 2: Desarrollo y compilación del motor de reglas SMC (FVG, OB, Liquidez).

[ ] Fase 3: Entrenamiento y despliegue del modelo de clasificación (IA).

[x] Fase 4: Integración con API de Broker para operativa independiente.

[ ] Fase 5: Stress-testing estadístico sobre el histórico completo.

💻 Stack Tecnológico
Lenguaje: Python 3.10+ (Core optimizado con Cython).

Procesamiento de Datos: Pandas, NumPy.

Inteligencia Artificial: Scikit-learn / TensorFlow.

Visualización: Mplfinance, Matplotlib.

Autor: Youssef Abderrahmani

Tutoría: Dpto. Lenguajes y Sistemas Informáticos (LSI) - Universidad de Sevilla.
