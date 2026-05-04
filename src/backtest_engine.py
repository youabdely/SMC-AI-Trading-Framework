import pandas as pd
import mplfinance as mpf
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import os
import sys
import shutil

# Desactivar modo interactivo para procesar rápido
plt.switch_backend('Agg')

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import load_data
import strategy 

def run_ultimate_backtest(years_to_test, months_to_exclude, data_folder, output_base):
    if not os.path.exists(output_base): os.makedirs(output_base)
    
    # LISTA GLOBAL PARA CAPTURAR TODOS LOS TRADES
    all_trades_global = []

    for year in years_to_test:
        print(f"\n>>> PROCESANDO AÑO: {year}")
        file_path = os.path.join(data_folder, f"XAUUSD_M1_{year}.csv")
        year_folder = os.path.join(output_base, str(year))

        if not os.path.exists(file_path): 
            print(f"Archivo no encontrado: {file_path}")
            continue
            
        if os.path.exists(year_folder): shutil.rmtree(year_folder)
        os.makedirs(os.path.join(year_folder, "ganadores"), exist_ok=True)
        os.makedirs(os.path.join(year_folder, "perdedores"), exist_ok=True)

        # --- PREPARACIÓN DE DATOS ---
        df = load_data(file_path)
        df_1h = df.resample('1h').agg({'high':'max', 'low':'min'}).ffill().dropna()
        df_4h = df.resample('4h').agg({'high':'max', 'low':'min'}).ffill().dropna()
        df_5m = df.resample('5min').agg({'open':'first', 'high':'max', 'low':'min', 'close':'last'}).dropna()
        df_5m.index = df_5m.index.tz_localize(None)

        # Cálculo de Niveles
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
            
            # REINICIO DE VARIABLES POR SESIÓN
            strategy.hay_liquidez_tomada = False
            if hasattr(strategy, 'idx_sweep'): strategy.idx_sweep = 0
            strategy.ultimo_min_formado = None
            strategy.ultimo_max_formado = None

            highs, lows, closes, opens = day_kz['high'].values, day_kz['low'].values, day_kz['close'].values, day_kz['open'].values
            times = day_kz.index

            for i in range(1, len(day_kz)):
                current_t = times[i]
                p_h = day_kz['lon_h'].iloc[i] if current_t.hour < 12 else day_kz['ny_h'].iloc[i]
                p_l = day_kz['lon_l'].iloc[i] if current_t.hour < 12 else day_kz['ny_l'].iloc[i]

                if pd.isna(p_h): continue

                try:
                    t_h = max(df_1h.loc[df_1h.index < current_t, 'high'].iloc[-1], 
                             df_4h.loc[df_4h.index < current_t, 'high'].iloc[-1])
                    t_l = min(df_1h.loc[df_1h.index < current_t, 'low'].iloc[-1], 
                             df_4h.loc[df_4h.index < current_t, 'low'].iloc[-1])
                except: continue

                # EJECUCIÓN ESTRATEGIA
                tipo, entry, sl, tp = strategy.check_signals(highs, lows, closes, opens, p_h, p_l, i, t_h, t_l)

                if tipo:
                    # GESTIÓN DEL TRADE (SIMULACIÓN DE SALIDA)
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
                        # CÁLCULO DEL RESULTADO EN R
                        rr_trade = (tp - entry) / (entry - sl) if tipo == "SHORT" else (tp - entry) / (entry - sl)
                        final_r = abs(rr_trade) if result == "TP" else -1.0
                        
                        # GUARDAR EN LA LISTA GLOBAL
                        all_trades_global.append({'year': year, 'result': final_r, 'type': tipo})

                        print(f"   [OK] Trade {trade_id} | {current_t} | {result} ({final_r:.2f}R)")
                        
                        # --- GENERACIÓN DE GRÁFICO (CON RECTÁNGULOS RESTAURADOS) ---
                        try:
                            start_p = current_t - pd.Timedelta(hours=2)
                            end_p = exit_time + pd.Timedelta(minutes=30)
                            plot_data = df_5m.loc[start_p:end_p].copy()
                            
                            y_min = min(plot_data['low'].min(), tp, sl)
                            y_max = max(plot_data['high'].max(), tp, sl)
                            pad = (y_max - y_min) * 0.15

                            fig, axlist = mpf.plot(plot_data, type='candle', style='charles', figsize=(14, 8),
                                title=f"T:{trade_id} | {tipo} | {result} | {final_r:.2f}R | {current_t}",
                                hlines=dict(hlines=[tp, sl, entry], colors=['g', 'r', 'gray'], alpha=0.4, linestyle='--'),
                                ylim=(y_min-pad, y_max+pad), returnfig=True)
                            
                            ax = axlist[0]
                            idx_in = plot_data.index.get_loc(current_t)
                            idx_out = plot_data.index.get_loc(exit_time)
                            ancho = idx_out - idx_in

                            # RESTAURADO: Sombras rojas y verdes
                            if tipo == "SHORT":
                                ax.add_patch(Rectangle((idx_in, entry), ancho, sl-entry, facecolor='red', alpha=0.2))
                                ax.add_patch(Rectangle((idx_in, tp), ancho, entry-tp, facecolor='green', alpha=0.2))
                            else:
                                ax.add_patch(Rectangle((idx_in, sl), ancho, entry-sl, facecolor='red', alpha=0.2))
                                ax.add_patch(Rectangle((idx_in, entry), ancho, tp-entry, facecolor='green', alpha=0.2))

                            folder = "ganadores" if result == "TP" else "perdedores"
                            fig.savefig(os.path.join(year_folder, folder, f"trade_{trade_id}.png"))
                            plt.close(fig)
                        except: plt.close('all')

                        trade_id += 1
                        strategy.hay_liquidez_tomada = False
                        break 
                        
    # AL FINAL DE TODOS LOS AÑOS, LANZAMOS EL REPORTE
    imprimir_reporte_detallado(all_trades_global)

def imprimir_reporte_detallado(all_trades):
    if not all_trades:
        print("\n[!] No se ejecutaron trades en el periodo.")
        return

    df = pd.DataFrame(all_trades)
    
    print("\n" + "="*60)
    print("      ESTADÍSTICAS PROFESIONALES DE ALGORITMO (SMC)")
    print("="*60)

    total_trades = len(df)
    win_rate = (len(df[df['result'] > 0]) / total_trades) * 100
    profit_total_r = df['result'].sum()
    
    # Profit Factor
    ganancias = df[df['result'] > 0]['result'].sum()
    perdidas = abs(df[df['result'] < 0]['result'].sum())
    profit_factor = ganancias / perdidas if perdidas != 0 else ganancias

    # Max Drawdown R
    df['cum_r'] = df['result'].cumsum()
    df['peak'] = df['cum_r'].cummax()
    df['drawdown'] = df['peak'] - df['cum_r']
    max_dd = df['drawdown'].max()

    print(f"BENEFICIO TOTAL:      {profit_total_r:>10.2f} R")
    print(f"WIN RATE:             {win_rate:>10.2f} %")
    print(f"PROFIT FACTOR:        {profit_factor:>10.2f}")
    print(f"MAX DRAWDOWN:         {max_dd:>10.2f} R")
    print(f"TOTAL OPERACIONES:    {total_trades:>10}")
    print("-" * 60)

    print(f"{'AÑO':<6} | {'TRADES':<7} | {'WINRATE':<10} | {'PROFIT (R)':<10}")
    print("-" * 60)
    for año in sorted(df['year'].unique()):
        d_y = df[df['year'] == año]
        wr_y = (len(d_y[d_y['result'] > 0]) / len(d_y)) * 100
        p_y = d_y['result'].sum()
        print(f"{año:<6} | {len(d_y):<7} | {wr_y:>8.2f}% | {p_y:>10.2f} R")
    print("="*60)

if __name__ == "__main__":
    AÑOS = [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024,2025]
    BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # Mes 8 excluido (Vacaciones)
    run_ultimate_backtest(AÑOS, [8], os.path.join(BASE, "data"), os.path.join(BASE, "results"))