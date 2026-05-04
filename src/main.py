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
import strategy # Importamos el binario .pyd o el .py

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Cambia el año aquí según necesites
AÑO_A_TESTEAR = 2015
DATA_PATH = os.path.join(BASE_DIR, "data", f"XAUUSD_M1_{AÑO_A_TESTEAR}.csv")
OUTPUT_FOLDER = os.path.join(BASE_DIR, "trades_individual")

def guardar_datos_ia(datos):
    dataset_path = os.path.join(BASE_DIR, "dataset_ia_oro.csv")
    file_exists = os.path.isfile(dataset_path)
    with open(dataset_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=datos.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(datos)

def run_backtest_individual():
    print(f"Iniciando Backtest Unificado para {AÑO_A_TESTEAR}...")
    setup_folders(OUTPUT_FOLDER)
    
    if not os.path.exists(DATA_PATH):
        print(f"ERROR: No existe {DATA_PATH}"); return

    df = load_data(DATA_PATH)

    # 1. Preparar Temporalidades Mayores (Igual que el masivo)
    df_1h = df.resample('1h').agg({'high':'max', 'low':'min'}).ffill().dropna()
    df_4h = df.resample('4h').agg({'high':'max', 'low':'min'}).ffill().dropna()

    # 2. Procesamiento M5
    df_5m = df.resample('5min').agg({'open':'first', 'high':'max', 'low':'min', 'close':'last'}).dropna()
    df_5m.index = df_5m.index.tz_localize(None)

    # 3. Cálculo de Niveles Londres y NY
    # 3. Cálculo de Niveles ICT (AM: 3-8 | PM: 8-12)
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

    # 4. Killzones (Doble sesión: Mañana y Tarde)
    killzones = df_5m[(df_5m.index.hour >= 9) & (df_5m.index.hour < 11) | 
                      (df_5m.index.hour >= 13) & (df_5m.index.hour < 15)].copy()

    trade_id, trade_history = 0, []

    for date, day_kz in killzones.groupby(killzones.index.date):
        if date.month == 8: continue # Vacaciones
        
        # --- RESETEO ESTRATEGIA (CRUCIAL) ---
        strategy.hay_liquidez_tomada = False
        strategy.ultimo_min_formado = None
        strategy.ultimo_max_formado = None

        highs, lows, closes, opens = day_kz['high'].values, day_kz['low'].values, day_kz['close'].values, day_kz['open'].values
        times = day_kz.index

        for i in range(1, len(day_kz)):
            current_t = times[i]
            
            # Selección de nivel según sç
            # Selección de nivel según la sesión actual
            if current_t.hour < 12:
                p_h = day_kz['am_h'].iloc[i]  # Usa el máximo 3-8
                p_l = day_kz['am_l'].iloc[i]  # Usa el mínimo 3-8
            else:
                p_h = day_kz['pm_h'].iloc[i]  # Usa el máximo 8-12
                p_l = day_kz['pm_l'].iloc[i]  # Usa el mínimo 8-12

            if pd.isna(p_h): continue

            # Targets MTF
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
                    # --- MINERÍA FASE 1 ---
                    try:
                        idx_s = strategy.idx_sweep
                        v_h, v_l, v_o, v_c = highs[idx_s], lows[idx_s], opens[idx_s], closes[idx_s]
                        t_total = v_h - v_l
                        ratio_mecha = ((v_h - max(v_o, v_c)) + (min(v_o, v_c) - v_l)) / t_total if t_total > 0 else 0
                        
                        r_londres = abs(day_kz['am_h'].iloc[0] - day_kz['am_l'].iloc[0])
                        
                        guardar_datos_ia({
                            'fecha': current_t.strftime('%Y-%m-%d %H:%M'),
                            'mes': current_t.month,
                            'dia_semana': current_t.weekday(),
                            'hora': current_t.hour,
                            'tipo': 1 if tipo == "LONG" else 0,
                            'rango_londres': round(r_londres, 3),
                            'ratio_mecha_sweep': round(ratio_mecha, 3),
                            'distancia_sl': round(abs(entry - sl), 3),
                            'resultado': 1 if result == "TP" else 0
                        })
                    except:
                        pass 
                    # --- FIN MINERÍA ---
                    # Cálculo R
                    rr_trade = abs((tp - entry) / (entry - sl))
                    profit_r = rr_trade if result == "TP" else -1.0
                    
                    # Guardar Stats
                    trade_history.append({
                        'ID': trade_id, 'Fecha': current_t, 'Tipo': tipo, 
                        'Resultado': result, 'Profit_R': profit_r
                    })

                    # --- GRÁFICO (MODIFICADO PARA INCLUIR DÍA Y HORA) ---
                    try:
                        start_p, end_p = current_t - pd.Timedelta(hours=2), exit_time + pd.Timedelta(minutes=30)
                        plot_data = df_5m.loc[start_p:end_p].copy()
                        
                        y_min, y_max = min(plot_data['low'].min(), tp, sl), max(plot_data['high'].max(), tp, sl)
                        pad = (y_max - y_min) * 0.15

                        # Formateamos la fecha de entrada para el título (Ej: 2024-05-15 09:30)
                        fecha_entrada_str = current_t.strftime('%Y-%m-%d %H:%M')

                        fig, axlist = mpf.plot(plot_data, type='candle', style='charles', figsize=(14, 8),
                            # MODIFICADO: El título ahora incluye la fecha de entrada
                            title=f"Trade {trade_id} | {tipo} | {result} | {profit_r:.2f}R\nEntrada: {fecha_entrada_str}",
                            hlines=dict(hlines=[tp, sl, entry], colors=['g', 'r', 'gray'], alpha=0.4, linestyle='--'),
                            ylim=(y_min-pad, y_max+pad), 
                            returnfig=True,
                            tight_layout=True,
                            # MODIFICADO: Asegura que el eje X muestre la hora y el día
                            datetime_format='%d/%m %H:%M' 
                        )
                        
                        ax = axlist[0]
                        idx_in, idx_out = plot_data.index.get_loc(current_t), plot_data.index.get_loc(exit_time)
                        ancho = idx_out - idx_in

                        if tipo == "SHORT":
                            ax.add_patch(Rectangle((idx_in, entry), ancho, sl-entry, facecolor='red', alpha=0.2))
                            ax.add_patch(Rectangle((idx_in, tp), ancho, entry-tp, facecolor='green', alpha=0.2))
                        else:
                            ax.add_patch(Rectangle((idx_in, sl), ancho, entry-sl, facecolor='red', alpha=0.2))
                            ax.add_patch(Rectangle((idx_in, entry), ancho, tp-entry, facecolor='green', alpha=0.2))

                        folder = "ganadores" if result == "TP" else "perdedores"
                        # MODIFICADO: El nombre del archivo también incluye la fecha para que estén ordenados
                        nombre_archivo = f"trade_{trade_id}_{current_t.strftime('%Y%m%d_%H%M')}.png"
                        fig.savefig(os.path.join(OUTPUT_FOLDER, folder, nombre_archivo))
                        plt.close(fig)
                    except Exception as e: 
                        print(f"Error generando gráfico trade {trade_id}: {e}")
                        plt.close('all')

                    trade_id += 1
                    strategy.hay_liquidez_tomada = False # Reset post-trade
                    break 

    if trade_history:
        res = pd.DataFrame(trade_history)
        print(f"\nFIN: Trades: {len(res)} | Profit Total {AÑO_A_TESTEAR}: {res['Profit_R'].sum():.2f}R")

if __name__ == "__main__":
    run_backtest_individual()