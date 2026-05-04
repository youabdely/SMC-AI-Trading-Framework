import pandas as pd
import mplfinance as mpf
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import os
import sys
import shutil
import tensorflow as tf
import joblib

# Desactivar modo interactivo
plt.switch_backend('Agg')

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import load_data
import strategy 

# --- CARGA DE MODELOS ---
BASE_PATH = os.path.dirname(os.path.abspath(__file__))
ia_model = tf.keras.models.load_model(os.path.join(BASE_PATH, 'modelo_ia_oro.keras'))
scaler = joblib.load(os.path.join(BASE_PATH, 'escalador_oro.pkl'))

def run_ia_backtest(years_to_test, months_to_exclude, data_folder, output_base):
    if not os.path.exists(output_base): os.makedirs(output_base)
    all_trades_global = []

    for year in years_to_test:
        print(f"\n>>> PROCESANDO AÑO (CON IA): {year}")
        file_path = os.path.join(data_folder, f"XAUUSD_M1_{year}.csv")
        year_folder = os.path.join(output_base, f"{year}_IA")

        if not os.path.exists(file_path): continue
        if os.path.exists(year_folder): shutil.rmtree(year_folder)
        os.makedirs(os.path.join(year_folder, "ganadores"), exist_ok=True)
        os.makedirs(os.path.join(year_folder, "perdedores"), exist_ok=True)

        df = load_data(file_path)
        df_1h = df.resample('1h').agg({'high':'max', 'low':'min'}).ffill().dropna()
        df_4h = df.resample('4h').agg({'high':'max', 'low':'min'}).ffill().dropna()
        df_5m = df.resample('5min').agg({'open':'first', 'high':'max', 'low':'min', 'close':'last'}).dropna()
        df_5m.index = df_5m.index.tz_localize(None)

        df_5m['date'] = df_5m.index.date
        for date, day_data in df_5m.groupby('date'):
            rango_london = day_data.between_time("03:00", "09:00")
            rango_ny = day_data.between_time("15:30", "18:00")
            if not rango_london.empty:
                df_5m.loc[df_5m['date'] == date, 'lon_h'] = rango_london['high'].max()
                df_5m.loc[df_5m['date'] == date, 'lon_l'] = rango_london['low'].min()
            if not rango_ny.empty:
                df_5m.loc[df_5m['date'] == date, 'ny_h'] = rango_ny['high'].max()
                df_5m.loc[df_5m['date'] == date, 'ny_l'] = rango_ny['low'].min()

        killzones = df_5m[(df_5m.index.hour >= 9) & (df_5m.index.hour < 11) | 
                          (df_5m.index.hour >= 19) & (df_5m.index.hour < 21)].copy()

        trade_id = 0
        for date, day_kz in killzones.groupby(killzones.index.date):
            if date.month in months_to_exclude: continue
            
            strategy.hay_liquidez_tomada = False
            strategy.ultimo_min_formado = None
            strategy.ultimo_max_formado = None

            highs, lows, closes, opens = day_kz['high'].values, day_kz['low'].values, day_kz['close'].values, day_kz['open'].values
            times = day_kz.index

            for i in range(1, len(day_kz)):
                current_t = times[i]
                p_h = day_kz['lon_h'].iloc[i] if current_t.hour < 12 else day_kz['ny_h'].iloc[i]
                p_l = day_kz['lon_l'].iloc[i] if current_t.hour < 12 else day_kz['ny_l'].iloc[i]

                try:
                    t_h = max(df_1h.loc[df_1h.index < current_t, 'high'].iloc[-1], df_4h.loc[df_4h.index < current_t, 'high'].iloc[-1])
                    t_l = min(df_1h.loc[df_1h.index < current_t, 'low'].iloc[-1], df_4h.loc[df_4h.index < current_t, 'low'].iloc[-1])
                except: continue

                tipo, entry, sl, tp = strategy.check_signals(highs, lows, closes, opens, p_h, p_l, i, t_h, t_l)

                if tipo:
                    # --- FILTRO DE IA ---
                    idx_s = strategy.idx_sweep
                    v_h, v_l, v_o, v_c = highs[idx_s], lows[idx_s], opens[idx_s], closes[idx_s]
                    ratio_mecha = ((v_h - max(v_o, v_c)) + (min(v_o, v_c) - v_l)) / (v_h - v_l) if (v_h - v_l) > 0 else 0
                    r_londres = abs(day_kz['lon_h'].iloc[0] - day_kz['lon_l'].iloc[0])
                    
                    # El orden debe coincidir con el entrenamiento: 
                    # [mes, dia_semana, hora, tipo_int, r_londres, ratio_mecha, dist_sl]
                    features = [[current_t.month, current_t.weekday(), current_t.hour, 
                                 1 if tipo == "LONG" else 0, r_londres, ratio_mecha, abs(entry - sl)]]
                    
                    f_scaled = scaler.transform(features)
                    prob = ia_model.predict(f_scaled, verbose=0)[0][0]

                    UMBRAL = 0.58  # Puedes ajustar esto (0.50 a 0.60)
                    
                    if prob < UMBRAL:
                        # IA rechaza el trade
                        strategy.hay_liquidez_tomada = False 
                        continue 

                    # Si llega aquí, el trade es aprobado por la IA
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
                        if result:
                            final_r = abs((tp - entry) / (entry - sl)) if result == "TP" else -1.0
                            all_trades_global.append({'year': year, 'result': final_r, 'prob': prob, 'year_val': year})
                            
                            # --- GENERACIÓN DE REPORTES VISUALES ---
                            try:
                                # Definimos el margen de tiempo para ver el contexto
                                start_p = current_t - pd.Timedelta(hours=3)
                                end_p = exit_time + pd.Timedelta(hours=1)
                                plot_data = df_5m.loc[start_p:end_p].copy()
                                
                                # Ajuste de escala visual
                                y_min = min(plot_data['low'].min(), tp, sl)
                                y_max = max(plot_data['high'].max(), tp, sl)
                                pad = (y_max - y_min) * 0.2

                                # Crear la figura con mpf
                                fig, axlist = mpf.plot(plot_data, type='candle', style='charles', figsize=(15, 9),
                                    title=f"SMC + IA (Confianza: {prob:.2f}) | {result} | {final_r:.2f}R\nEntrada: {current_t}",
                                    hlines=dict(hlines=[tp, sl, entry], colors=['#26a69a', '#ef5350', '#787b86'], alpha=0.5, linestyle='--'),
                                    ylim=(y_min-pad, y_max+pad), returnfig=True)
                                
                                ax = axlist[0]
                                # Localizar índices para dibujar los rectángulos de Profit/Loss
                                idx_in = plot_data.index.get_loc(current_t)
                                idx_out = plot_data.index.get_loc(exit_time)
                                ancho = idx_out - idx_in

                                # Dibujar sombreado de la operación
                                if tipo == "SHORT":
                                    ax.add_patch(Rectangle((idx_in, entry), ancho, sl-entry, facecolor='#ef5350', alpha=0.15))
                                    ax.add_patch(Rectangle((idx_in, tp), ancho, entry-tp, facecolor='#26a69a', alpha=0.15))
                                else:
                                    ax.add_patch(Rectangle((idx_in, sl), ancho, entry-sl, facecolor='#ef5350', alpha=0.15))
                                    ax.add_patch(Rectangle((idx_in, entry), ancho, tp-entry, facecolor='#26a69a', alpha=0.15))

                                # Guardar en la carpeta correspondiente
                                folder = "ganadores" if result == "TP" else "perdedores"
                                file_name = f"trade_IA_{year}_{trade_id}.png"
                                fig.savefig(os.path.join(year_folder, folder, file_name))
                                plt.close(fig)
                                
                            except Exception as e:
                                print(f"Error generando imagen del trade {trade_id}: {e}")
                                plt.close('all')

                            print(f"   [IA OK] Prob: {prob:.2f} | {result} ({final_r:.2f}R) | Imagen Guardada")
                            trade_id += 1
                            strategy.hay_liquidez_tomada = False
                            break
    
    imprimir_reporte_detallado(all_trades_global)

def imprimir_reporte_detallado(all_trades):
    if not all_trades: return
    df = pd.DataFrame(all_trades)
    print("\n" + "="*60)
    print("      ESTADÍSTICAS FINALES (OPTIMIZADAS CON IA)")
    print("="*60)
    
    total_trades = len(df)
    win_rate = (len(df[df['result'] > 0]) / total_trades) * 100
    profit_r = df['result'].sum()
    ganancias = df[df['result'] > 0]['result'].sum()
    perdidas = abs(df[df['result'] < 0]['result'].sum())
    pf = ganancias / perdidas if perdidas != 0 else ganancias
    
    print(f"BENEFICIO TOTAL:      {profit_r:>10.2f} R")
    print(f"WIN RATE:             {win_rate:>10.2f} %")
    print(f"PROFIT FACTOR:        {pf:>10.2f}")
    print(f"TOTAL OPERACIONES:    {total_trades:>10}")
    print("-" * 60)
    for año in sorted(df['year'].unique()):
        d_y = df[df['year'] == año]
        wr_y = (len(d_y[d_y['result'] > 0]) / len(d_y)) * 100
        print(f"{año:<6} | {len(d_y):<7} trades | {wr_y:>8.2f}% WR | {d_y['result'].sum():>10.2f} R")
    print("="*60)

if __name__ == "__main__":
    AÑOS = [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025]
    BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    run_ia_backtest(AÑOS, [8], os.path.join(BASE, "data"), os.path.join(BASE, "results"))