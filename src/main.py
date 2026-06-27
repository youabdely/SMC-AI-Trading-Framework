import pandas as pd
import mplfinance as mpf
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import os
import sys
import csv

# Desactivar modo interactivo para procesar rápido las imágenes
plt.switch_backend('Agg')

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import setup_folders, load_data
import strategy 

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AÑO_A_TESTEAR = 2017
DATA_PATH = os.path.join(BASE_DIR, "data", f"XAUUSD_M1_{AÑO_A_TESTEAR}.csv")
OUTPUT_FOLDER = os.path.join(BASE_DIR, "trades_individual")

def guardar_datos_ia(datos):
    dataset_path = os.path.join(BASE_DIR, "dataset_ia_oro.csv")
    file_exists = os.path.isfile(dataset_path)
    with open(dataset_path, 'a', newline='') as f:
        # Forzamos que los números se escriban con decimales en el CSV
        writer = csv.DictWriter(f, fieldnames=datos.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(datos)

def run_backtest_individual():
    print(f"Iniciando Minería de Datos para IA {AÑO_A_TESTEAR}...")
    setup_folders(OUTPUT_FOLDER)
    
    if not os.path.exists(DATA_PATH):
        print(f"ERROR: No existe {DATA_PATH}"); return

    df = load_data(DATA_PATH)
    df_1h = df.resample('1h').agg({'high':'max', 'low':'min'}).ffill().dropna()
    df_4h = df.resample('4h').agg({'high':'max', 'low':'min'}).ffill().dropna()
    df_5m = df.resample('5min').agg({'open':'first', 'high':'max', 'low':'min', 'close':'last'}).dropna()
    df_5m.index = df_5m.index.tz_localize(None)

    df_5m['date'] = df_5m.index.date
    for date, day_data in df_5m.groupby('date'):
        r_am = day_data.between_time("03:00", "08:00")
        r_pm = day_data.between_time("08:00", "12:00")
        if not r_am.empty:
            df_5m.loc[df_5m['date'] == date, 'am_h'] = r_am['high'].max()
            df_5m.loc[df_5m['date'] == date, 'am_l'] = r_am['low'].min()
        if not r_pm.empty:
            df_5m.loc[df_5m['date'] == date, 'pm_h'] = r_pm['high'].max()
            df_5m.loc[df_5m['date'] == date, 'pm_l'] = r_pm['low'].min()

    killzones = df_5m[(df_5m.index.hour >= 8) & (df_5m.index.hour < 11) | 
                      (df_5m.index.hour >= 13) & (df_5m.index.hour < 15)].copy()

    trade_id, trade_history = 0, []

    for date, day_kz in killzones.groupby(killzones.index.date):
        if date.month == 8: continue 
        
        strategy.hay_liquidez_tomada = False
        strategy.ultimo_min_formado = None
        strategy.ultimo_max_formado = None

        highs, lows, closes, opens = day_kz['high'].values, day_kz['low'].values, day_kz['close'].values, day_kz['open'].values
        times = day_kz.index

        for i in range(1, len(day_kz)):
            current_t = times[i]
            
            p_h = day_kz['am_h'].iloc[i] if current_t.hour < 12 else day_kz['pm_h'].iloc[i]
            p_l = day_kz['am_l'].iloc[i] if current_t.hour < 12 else day_kz['pm_l'].iloc[i]

            if pd.isna(p_h): continue

            try:
                t_h = max(df_1h.loc[df_1h.index < current_t, 'high'].iloc[-1], 
                          df_4h.loc[df_4h.index < current_t, 'high'].iloc[-1])
                t_l = min(df_1h.loc[df_1h.index < current_t, 'low'].iloc[-1], 
                          df_4h.loc[df_4h.index < current_t, 'low'].iloc[-1])
            except: continue

            tipo, entry, sl, tp = strategy.check_signals(highs, lows, closes, opens, p_h, p_l, i, t_h, t_l)

            if tipo:
                future = df_5m.loc[current_t:]
                result, exit_time = None, None
                for t, row in future.iloc[1:].iterrows():
                    if tipo == "SHORT":
                        if row['high'] >= sl: result, exit_time = "SL", t; break
                        if row['low'] <= tp: result, exit_time = "TP", t; break
                    else:
                        if row['low'] <= sl: result, exit_time = "SL", t; break
                        if row['high'] >= tp: result, exit_time = "TP", t; break

                if result:
                    # --- CAMBIO CRUCIAL: HORA DECIMAL ---
                    try:
                        idx_s = strategy.idx_sweep
                        v_h, v_l, v_o, v_c = highs[idx_s], lows[idx_s], opens[idx_s], closes[idx_s]
                        t_total = v_h - v_l
                        ratio_mecha = ((v_h - max(v_o, v_c)) + (min(v_o, v_c) - v_l)) / t_total if t_total > 0 else 0
                        r_londres = abs(day_kz['am_h'].iloc[0] - day_kz['am_l'].iloc[0])
                        
                        # Cálculo de hora decimal (9:30 -> 9.5)
                        h_decimal = float(current_t.hour + (current_t.minute / 60.0))

                        guardar_datos_ia({
                            'fecha': current_t.strftime('%Y-%m-%d %H:%M'),
                            'mes': current_t.month,
                            'dia_semana': current_t.weekday(),
                            'hora': round(h_decimal, 3), # <--- AQUÍ ESTÁ EL CAMBIO
                            'tipo': 1 if tipo == "LONG" else 0,
                            'rango_londres': round(r_londres, 3),
                            'ratio_mecha_sweep': round(ratio_mecha, 3),
                            'distancia_sl': round(abs(entry - sl), 3),
                            'resultado': 1 if result == "TP" else 0
                        })
                    except: pass 

                    rr_trade = abs((tp - entry) / (entry - sl))
                    profit_r = rr_trade if result == "TP" else -1.0
                    trade_history.append({'ID': trade_id, 'Fecha': current_t, 'Tipo': tipo, 'Resultado': result, 'Profit_R': profit_r})

                    # Gráfico omitido por brevedad pero mantenlo en tu archivo
                    trade_id += 1
                    strategy.hay_liquidez_tomada = False
                    break 

    if trade_history:
        res = pd.DataFrame(trade_history)
        print(f"\nFIN: Trades: {len(res)} | Profit Total: {res['Profit_R'].sum():.2f}R")

if __name__ == "__main__":
    run_backtest_individual()