import MetaTrader5 as mt5
import pandas as pd
import time
from datetime import datetime
import pytz
import os
import strategy  

# --- CONFIGURACIÓN ---
SIMBOLO = "XAUUSD"
TIMEFRAME = mt5.TIMEFRAME_M5

def conectar_mt5():
    # 1. Intentamos la inicialización estándar (por si ya está abierto)
    if mt5.initialize():
        print(f"✅ Conectado a MT5 - Símbolo: {SIMBOLO} (Gráfico M5)")
        return True
        
    # 2. Si falla, probamos las rutas de instalación más comunes en Windows
    rutas_probables = [
        "C:\\Program Files\\MetaTrader 5\\terminal64.exe",
        "C:\\Program Files (x86)\\MetaTrader 5\\terminal64.exe",
        # Esta ruta busca en la carpeta global de accesos directos de Windows
        "C:\\ProgramData\\Microsoft\\Windows\\Start Menu\\Programs\\MetaTrader 5\\MetaTrader 5.lnk",
    ]
    
    for ruta in rutas_probables:
        if os.path.exists(ruta):
            if mt5.initialize(path=ruta):
                print(f"✅ Conectado a MT5 forzando ruta: {ruta}")
                return True
                
    # 3. Si llega aquí, es que no se ha podido conectar de ninguna forma
    print("❌ Error crítico: No se encuentra MetaTrader 5 abierto ni instalado en las rutas estándar.")
    print("👉 CONSEJO: Abre la aplicación MetaTrader 5 a mano en tu escritorio ANTES de lanzar el script.")
    return False

def obtener_datos(n_velas=200):
    rates = mt5.copy_rates_from_pos(SIMBOLO, TIMEFRAME, 0, n_velas)
    if rates is None:
        return None
    df = pd.DataFrame(rates)
    # Convertimos a float64 para que coincida con lo que espera numpy en tu estrategia
    return df
def abrir_operacion(tipo_senal, precio_entrada, sl, tp):
    """Envía una orden de mercado real a MetaTrader 5 con SL y TP automatizados"""
    # Configurar si es una compra o una venta institucional
    tipo_orden = mt5.ORDER_TYPE_BUY if tipo_senal == "LONG" else mt5.ORDER_TYPE_SELL
    
    # Intentamos rascar el precio de ejecución actual según el tipo de orden
    precio_actual = mt5.symbol_info_tick(SIMBOLO).ask if tipo_senal == "LONG" else mt5.symbol_info_tick(SIMBOLO).bid
    if precio_actual is None:
        precio_actual = precio_entrada # Backup por si falla el tick instantáneo

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": SIMBOLO,
        "volume": 0.1,  # Ajusta el lotaje según la gestión de riesgo de tu TFG
        "type": tipo_orden,
        "price": precio_actual,
        "sl": float(sl),
        "tp": float(tp),
        "deviation": 20,
        "magic": 202605,  # Identificador único de tu bot
        "comment": "Bot SMC-IA TFG",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILL_IOC,
    }
    
    resultado = mt5.order_send(request)
    if resultado.retcode != mt5.TRADE_RETCODE_DONE:
        print(f"❌ Error al enviar orden a MT5: {resultado.comment} (Código: {resultado.retcode})")
    else:
        print(f"💰 ¡ORDEN EJECUTADA CON ÉXITO! Ticket: {resultado.order}")

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