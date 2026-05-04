import MetaTrader5 as mt5
import pandas as pd
import time
from datetime import datetime
import pytz
import strategy  # Asegúrate de que strategy.py esté en la misma carpeta

# --- CONFIGURACIÓN ---
SIMBOLO = "XAUUSD"
TIMEFRAME = mt5.TIMEFRAME_M5

def conectar_mt5():
    if not mt5.initialize():
        print("❌ Error al conectar con MetaTrader 5")
        return False
    print(f"✅ Conectado a MT5 - Símbolo: {SIMBOLO}")
    return True

def obtener_datos(n_velas=200):
    rates = mt5.copy_rates_from_pos(SIMBOLO, TIMEFRAME, 0, n_velas)
    if rates is None:
        return None
    df = pd.DataFrame(rates)
    # Convertimos a float64 para que coincida con lo que espera numpy en tu estrategia
    return df

def ejecutar_puente():
    if not conectar_mt5(): return

    print("🚀 PUENTE ACTIVO. Esperando sesión de Nueva York...")

    try:
        while True:
            # 1. Filtro de sesión (NY)
            tz_ny = pytz.timezone('America/New_York')
            hora_ny = datetime.now(tz_ny)
            # Simplificamos para que puedas probarlo ahora mismo si quieres
            # Mañana activamos el filtro estricto AM/PM
            
            # 2. Captura de datos
            df = obtener_datos(200)
            if df is not None:
                # 3. Preparar arrays para strategy.py
                highs = df['high'].values.astype(float)
                lows = df['low'].values.astype(float)
                closes = df['close'].values.astype(float)
                opens = df['open'].values.astype(float)
                
                # 4. Parámetros que pide tu función check_signals
                # Estos valores los puliremos mañana para que sean los reales de Londres
                lon_high = highs[-50:].max() 
                lon_low = lows[-50:].min()
                target_h = lon_high + 5.0 # Objetivo provisional
                target_l = lon_low - 5.0  # Objetivo provisional
                idx_actual = len(df) - 1

                # 5. LLAMADA A TU ESTRATEGIA
                tipo, entry, sl, tp = strategy.check_signals(
                    highs, lows, closes, opens,
                    lon_high, lon_low,
                    idx_actual,
                    target_h, target_l
                )
                
                if tipo:
                    print(f"🔥 ¡SEÑAL! {tipo} | Entry: {entry} | SL: {sl} | TP: {tp}")
                else:
                    print(f"⌛ Buscando Setup... Precio: {closes[-1]}", end="\r")

            # Esperar 60 segundos para la siguiente vela
            time.sleep(60)

    except KeyboardInterrupt:
        print("\n👋 Puente detenido.")
    finally:
        mt5.shutdown()

if __name__ == "__main__":
    ejecutar_puente()