import MetaTrader5 as mt5
import pandas as pd

def obtener_precios():
    if not mt5.initialize():
        print("Error al conectar")
        return

    simbolo = "XAUUSD"
    # Vamos a pedir las últimas 10 velas de 5 minutos (M5)
    velas = mt5.copy_rates_from_pos(simbolo, mt5.TIMEFRAME_M5, 0, 10)
    
    if velas is None:
        print(f"No se encontró el símbolo {simbolo}. Revisa si tu broker lo llama de otra forma.")
        mt5.shutdown()
        return

    # Convertimos los datos a un formato que Python entiende bien (DataFrame)
    df = pd.DataFrame(velas)
    # Convertimos la hora de segundos a formato legible
    df['time'] = pd.to_datetime(df['time'], unit='s')

    print(f"\n--- Últimas velas de {simbolo} ---")
    print(df[['time', 'open', 'high', 'low', 'close']])

    mt5.shutdown()

if __name__ == "__main__":
    obtener_precios()