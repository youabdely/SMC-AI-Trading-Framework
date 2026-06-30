<div align="center">

# 🤖 SMC AI Trading Framework

### Algorithmic Trading Framework powered by Smart Money Concepts and Artificial Intelligence

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)]()
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)]()
[![MetaTrader5](https://img.shields.io/badge/MetaTrader5-API-green.svg)]()
[![Cython](https://img.shields.io/badge/Cython-Optimized-yellow.svg)]()
[![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)]()

*A modular algorithmic trading framework combining deterministic Smart Money Concepts (SMC) with Machine Learning to improve trade selection through probabilistic filtering.*

</div>

---

## 📖 Overview

SMC AI Trading Framework is a research-oriented algorithmic trading platform developed as a Computer Engineering Final Degree Project.

Instead of attempting to predict market prices directly, the framework transforms **Smart Money Concepts (SMC)** into deterministic mathematical rules. These trading opportunities are then evaluated by a Machine Learning model that estimates the probability of success before execution.

The project combines:

- Deterministic rule-based trading
- Feature engineering
- Machine Learning classification
- Historical backtesting
- Performance evaluation
- Live MetaTrader 5 integration

---

# ✨ Features

- 📈 Smart Money Concepts implementation
- 🧠 AI-powered trade filtering
- ⚡ High-performance strategy engine
- 🔬 Multiple ML models comparison
- 📊 Historical backtesting engine
- 📉 Equity curve generation
- 📋 Professional performance reports
- 🛡 Risk management system
- ⚙ MetaTrader 5 integration
- 🚀 Cython optimized strategy module

---

# 🏗 System Architecture

```
                Historical / Live Data
                         │
                         ▼
                Data Preprocessing
                         │
                         ▼
            Smart Money Rule Engine
                         │
              Trading Opportunity
                         │
                         ▼
          Feature Engineering Pipeline
                         │
                         ▼
              Machine Learning Model
                         │
            Probability Estimation
                         │
        Probability > Decision Threshold?
                 │               │
               YES               NO
                 │               │
                 ▼               ▼
          Execute Trade      Ignore Signal
```

---

# 🧠 Machine Learning

Several supervised learning algorithms were evaluated during the research:

| Model | Purpose |
|--------|----------|
| Multi-Layer Perceptron (MLP) | Final production model |
| Random Forest | Baseline comparison |
| LSTM | Sequential modelling |

The final implementation uses a **Multi-Layer Perceptron (MLP)** after comparative evaluation, achieving the best balance between inference speed, robustness and generalization.

---

# 📊 Trading Logic

The deterministic engine follows the Smart Money Concepts methodology:

1. Killzone validation
2. Liquidity pool detection
3. Liquidity sweep identification
4. Break of Structure (BOS)
5. Feature extraction
6. AI probability estimation
7. Risk validation
8. Trade execution

---

# 📂 Project Structure

```
SMC-AI-Trading-Framework/
│
├── strategy.pyx            # Cython strategy engine
├── backtest.py             # Historical simulator
├── live_trading.py         # Live trading
├── train_model.py          # AI training
├── feature_engineering.py
├── models/
│
├── datasets/
│
├── reports/
│
├── figures/
│
├── utils/
│
└── README.md
```

---

# ⚙ Installation

Clone the repository

```bash
git clone https://github.com/youabdely/SMC-AI-Trading-Framework.git

cd SMC-AI-Trading-Framework
```

Install dependencies

```bash
pip install -r requirements.txt
```

Compile the optimized strategy module

```bash
cythonize -i -3 strategy.pyx
```

---

# 🚀 Usage

### Train the model

```bash
python train_model.py
```

### Run historical backtest

```bash
python backtest.py
```

### Start live trading

```bash
python live_trading.py
```

---

# 📊 Research Methodology

The project follows a complete Machine Learning workflow:

- Historical data acquisition
- Feature Engineering
- Dataset generation
- Model training
- Hyperparameter optimization
- In-Sample validation
- Out-of-Sample validation
- Comparative evaluation
- Live deployment

---

# 📈 Risk Management

The framework incorporates a fully automated risk management module:

- Fixed percentage risk per trade
- Dynamic position sizing
- Automatic Stop Loss calculation
- Automatic Take Profit calculation
- Risk/Reward validation
- Maximum one trade per session

---

# ⚡ Performance Optimization

Critical components are compiled using **Cython**, providing:

- Lower latency
- Faster execution
- Source code protection
- Production-ready deployment

---

# 🛠 Technologies

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

# 🎓 Academic Context

This project was developed as the Final Degree Project (TFG) for the Computer Engineering degree.

**Title**

> Development and Implementation of an Artificial Intelligence Optimized Algorithmic Trading Platform

---

# ⚠ Disclaimer

This repository is intended exclusively for educational and research purposes.

Trading financial markets involves significant risk. Past performance does not guarantee future results.

---

# 👨‍💻 Author

**Youssef Abderrahmani**

Computer Engineering

Artificial Intelligence • Algorithmic Trading • Machine Learning

---

<div align="center">

⭐ If you find this project interesting, consider giving it a star.

</div>