import MetaTrader5 as mt5
import pandas as pd
import numpy as np
# import strategy  # <--- Descomenta esto cuando el .pyd esté en la carpeta src

def test_motor_live():
    mt5.initialize()
    simbolo = "XAUUSD"
    
    # 1. Pedimos suficientes velas para que el motor calcule fractales (ej. 100 velas)
    velas = mt5.copy_rates_from_pos(simbolo, mt5.TIMEFRAME_M5, 0, 100)
    df = pd.DataFrame(velas)

    # 2. Convertimos a formato Numpy (lo que pide el .pyd)
    highs = df['high'].values.astype(np.float64)
    lows = df['low'].values.astype(np.float64)
    closes = df['close'].values.astype(np.float64)

    # 3. Extraemos la hora actual para las Killzones
    # Importante: Usamos la hora de la última vela del broker
    hora_actual = pd.to_datetime(df['time'].iloc[-1], unit='s').hour
    minuto_actual = pd.to_datetime(df['time'].iloc[-1], unit='s').minute

    print(f"Datos preparados. Hora servidor: {hora_actual}:{minuto_actual}")
    print(f"Último Close: {closes[-1]}")

    # --- AQUÍ LLAMARÁS AL PYD ---
    # resultado = strategy.check_signals(highs, lows, closes, hora_actual, ...)
    # print(f"Resultado del motor: {resultado}")

    mt5.shutdown()

if __name__ == "__main__":
    test_motor_live()