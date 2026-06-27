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

        # 1. Cálculo de Niveles ICT (Lógica exacta del MAIN: AM 3-8 | PM 8-12)
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

        # 2. Killzones (Lógica exacta del MAIN: 8-11 y 13-15)
        killzones = df_5m[(df_5m.index.hour >= 8) & (df_5m.index.hour < 11) | 
                          (df_5m.index.hour >= 13) & (df_5m.index.hour < 15)].copy()

        trade_id = 0
        for date, day_kz in killzones.groupby(killzones.index.date):
            if date.month in months_to_exclude: continue
            
            # RESETEO POR SESIÓN
            strategy.hay_liquidez_tomada = False
            if hasattr(strategy, 'idx_sweep'): strategy.idx_sweep = 0
            strategy.ultimo_min_formado = None
            strategy.ultimo_max_formado = None

            highs, lows, closes, opens = day_kz['high'].values, day_kz['low'].values, day_kz['close'].values, day_kz['open'].values
            times = day_kz.index

            for i in range(1, len(day_kz)):
                current_t = times[i]
                
                # Selección de nivel AM/PM según el MAIN
                if current_t.hour < 12:
                    p_h = day_kz['am_h'].iloc[i]
                    p_l = day_kz['am_l'].iloc[i]
                else:
                    p_h = day_kz['pm_h'].iloc[i]
                    p_l = day_kz['pm_l'].iloc[i]

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
                        # Cálculo R (Unificado con la fórmula del MAIN)
                        rr_trade = abs((tp - entry) / (entry - sl))
                        final_r = rr_trade if result == "TP" else -1.0
                        
                        all_trades_global.append({'year': year, 'result': final_r, 'type': tipo})
                        print(f"   [OK] Trade {trade_id} | {current_t} | {result} ({final_r:.2f}R)")
                        
                        # --- GRÁFICO (CON CONTEXTO DE 7 HORAS Y NIVELES ICT) ---
                        try:
                            start_p = current_t - pd.Timedelta(hours=7)
                            end_p = exit_time + pd.Timedelta(minutes=30)
                            # Usamos df_5m para poder ver velas fuera de la killzone
                            plot_data = df_5m.loc[df_5m.index >= start_p].loc[:end_p].copy()
                            
                            y_min = min(plot_data['low'].min(), tp, sl, p_l)
                            y_max = max(plot_data['high'].max(), tp, sl, p_h)
                            pad = (y_max - y_min) * 0.15

                            fig, axlist = mpf.plot(plot_data, type='candle', style='charles', figsize=(14, 8),
                                title=f"T:{trade_id} | {tipo} | {result} | {final_r:.2f}R | {current_t}",
                                hlines=dict(hlines=[tp, sl, entry, p_h, p_l], 
                                            colors=['g', 'r', 'gray', 'blue', 'blue'], 
                                            alpha=0.4, linestyle='--'),
                                ylim=(y_min-pad, y_max+pad), returnfig=True,
                                datetime_format='%d/%m %H:%M')
                            
                            ax = axlist[0]
                            idx_in = plot_data.index.get_loc(current_t)
                            idx_out = plot_data.index.get_loc(exit_time)
                            ancho = idx_out - idx_in

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
    
    ganancias = df[df['result'] > 0]['result'].sum()
    perdidas = abs(df[df['result'] < 0]['result'].sum())
    profit_factor = ganancias / perdidas if perdidas != 0 else ganancias

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

    for año in sorted(df['year'].unique()):
        d_y = df[df['year'] == año]
        wr_y = (len(d_y[d_y['result'] > 0]) / len(d_y)) * 100
        p_y = d_y['result'].sum()
        print(f"{año:<6} | {len(d_y):<7} | {wr_y:>8.2f}% | {p_y:>10.2f} R")
    print("="*60)

if __name__ == "__main__":
    AÑOS = [2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025]
    BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    run_ultimate_backtest(AÑOS, [8], os.path.join(BASE, "data"), os.path.join(BASE, "results"))