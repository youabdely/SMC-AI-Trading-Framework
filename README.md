# Hybrid Algorithmic System: SMC Logic & Machine Learning Validation

Este proyecto de **Trabajo de Fin de Grado (TFG)** presenta el diseño, implementación y validación de un framework autónomo de trading algorítmico de alta precisión para el par **XAU/USD (Oro)**. El sistema digitaliza y automatiza la metodología de análisis técnico avanzado **SMC (Smart Money Concepts)**, optimizando la toma de decisiones mediante un supervisor probabilístico basado en **Aprendizaje Profundo (Deep Learning)**.

---

## 🎯 Visión General
El objetivo principal es la creación de un sistema analítico *end-to-end* capaz de procesar volúmenes masivos de datos históricos y flujos en tiempo real. El framework elimina el factor emocional y los sesgos cognitivos en entornos competitivos mediante una ejecución estrictamente determinista y validada estadísticamente sobre un extenso histórico continuo.

---
## 🚀 Guía de Inicio Rápido (Despliegue y Ejecución)

Siga los siguientes pasos para compilar los módulos nativos y ejecutar el sistema en su entorno local.

### 1. Instalación de dependencias

Se requiere **Python 3.10** o superior. Instale las librerías necesarias ejecutando:

```bash
pip install pandas numpy tensorflow scikit-learn mplfinance matplotlib pytz cython MetaTrader5
```

---

### 2. Compilación de Módulos Nativos (Optimización en C)

Para mitigar la sobrecarga (*overhead*) de Python y competir en latencia dentro de la microestructura del mercado, el núcleo de reglas heurísticas se encuentra implementado en **C** y compilado mediante **Cython**.

Compile el módulo nativo ejecutando:

```bash
python src/setup.py build_ext --inplace
```

La compilación generará automáticamente el módulo compartido (`.so` en Linux/macOS o `.pyd` en Windows), que será utilizado por el sistema durante la ejecución.

---

### 3. Ejecución del Core y Backtesting Estadístico

Una vez instaladas las dependencias y compilados los módulos nativos, ejecute el framework en modo de análisis histórico mediante:

```bash
python src/main.py
```

El sistema iniciará automáticamente el pipeline completo de:

* Carga del histórico.
* Preprocesamiento y ETL.
* Construcción de temporalidades.
* Ejecución del motor SMC.
* Filtrado mediante IA (MLP).
* Backtesting estadístico.
* Generación de métricas y visualizaciones.

> **Nota:** El framework permite alternar entre el motor determinista (`backtest_engine.py`) y el motor supervisado mediante inteligencia artificial (`backtest_engine_AI.py`) sin modificar el resto del pipeline.

---

### 4. Ejecución en Tiempo Real (Live Trading Bridge)

El framework incorpora un puente bidireccional con **MetaTrader 5 (MT5)** para el procesamiento continuo de datos de mercado en tiempo real.

Ejecute el puente mediante:

```bash
python src/puente_tfg.py
```

Durante la ejecución:

* Se establece la conexión con el terminal **MetaTrader 5**.
* Se reciben automáticamente las nuevas velas de **1 minuto (M1)**.
* Cada minuto, los datos son transferidos al motor de estrategia.
* El motor procesa las reglas SMC y, opcionalmente, el filtro supervisor basado en IA.
* En función del resultado, el sistema genera la señal operativa correspondiente para su posterior ejecución o supervisión.

Este diseño permite reutilizar el mismo motor analítico empleado durante el backtesting sobre un flujo de datos en tiempo real, garantizando la coherencia entre la validación histórica y la operativa *live*.
