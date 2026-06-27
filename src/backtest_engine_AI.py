import pandas as pd
import mplfinance as mpf
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import os
import sys
import shutil
import tensorflow as tf
import joblib
import numpy as np

# Desactivar modo interactivo para procesamiento rápido
plt.switch_backend('Agg')

BASE_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_PATH)
from utils import load_data
import strategy 

# --- CARGA GENERAL DEL ESCALADOR ---
scaler = joblib.load(os.path.join(BASE_PATH, 'escalador_oro.pkl'))

def cargar_modelo_especifico(tipo_modelo):
    """Carga el modelo seleccionado para el backtest"""
    if tipo_modelo == "MLP":
        return tf.keras.models.load_model(os.path.join(BASE_PATH, 'modelo_ia_oro.keras'))
    elif tipo_modelo == "RF":
        return joblib.load(os.path.join(BASE_PATH, 'modelo_rf_oro.pkl'))
    elif tipo_modelo == "LSTM":
        return tf.keras.models.load_model(os.path.join(BASE_PATH, 'modelo_lstm_oro.keras'))
    else:
        raise ValueError("Modelo no reconocido. Elige entre: 'MLP', 'RF' o 'LSTM'")

def run_ia_backtest(years_to_test, months_to_exclude, data_folder, output_base, tipo_modelo="MLP"):
    print(f"\n" + "="*70)
    print(f" INICIANDO BACKTEST MULTIANUAL USANDO MODELO: {tipo_modelo}")
    print("="*70)
    
    model = cargar_modelo_especifico(tipo_modelo)
    if not os.path.exists(output_base): os.makedirs(output_base)
    all_trades_global = []

    for year in years_to_test:
        print(f"\n>>> PROCESANDO AÑO: {year} | MODELO: {tipo_modelo}")
        file_path = os.path.join(data_folder, f"XAUUSD_M1_{year}.csv")
        year_folder = os.path.join(output_base, f"{year}_{tipo_modelo}")

        if not os.path.exists(file_path): 
            print(f"Archivo no encontrado: {file_path}")
            continue
            
        if os.path.exists(year_folder): shutil.rmtree(year_folder)
        os.makedirs(os.path.join(year_folder, "ganadores"), exist_ok=True)
        os.makedirs(os.path.join(year_folder, "perdedores"), exist_ok=True)

        # Historial temporal local para cada año
        historial_features_escaladas = []

        # --- PREPARACIÓN DE DATOS ---
        df = load_data(file_path)
        df_1h = df.resample('1h').agg({'high':'max', 'low':'min'}).ffill().dropna()
        df_4h = df.resample('4h').agg({'high':'max', 'low':'min'}).ffill().dropna()
        df_5m = df.resample('5min').agg({'open':'first', 'high':'max', 'low':'min', 'close':'last'}).dropna()
        df_5m.index = df_5m.index.tz_localize(None)

        # Cálculo de Niveles ICT
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

        # Killzones
        killzones = df_5m[(df_5m.index.hour >= 8) & (df_5m.index.hour < 11) | 
                          (df_5m.index.hour >= 13) & (df_5m.index.hour < 15)].copy()

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
                    idx_s = strategy.idx_sweep
                    v_h, v_l, v_o, v_c = highs[idx_s], lows[idx_s], opens[idx_s], closes[idx_s]
                    
                    t_total = v_h - v_l
                    ratio_mecha = ((v_h - max(v_o, v_c)) + (min(v_o, v_c) - v_l)) / t_total if t_total > 0 else 0
                    r_londres = abs(day_kz['am_h'].iloc[0] - day_kz['am_l'].iloc[0])
                    
                    # Extraer características
                    features_list = [[current_t.month, current_t.weekday(), current_t.hour, 
                                      1 if tipo == "LONG" else 0, r_londres, ratio_mecha, abs(entry - sl)]]
                    
                    f_scaled = scaler.transform(features_list)
                    
                    # --- INFERENCIA DE MODELOS ---
                    if tipo_modelo == "RF":
                        prob = model.predict_proba(f_scaled)[0][1]
                    elif tipo_modelo == "MLP":
                        prob = model.predict(f_scaled, verbose=0)[0][0]
                    elif tipo_modelo == "LSTM":
                        historial_features_escaladas.append(f_scaled[0])
                        if len(historial_features_escaladas) < 3:
                            pad_size = 3 - len(historial_features_escaladas)
                            padding = [np.zeros(7) for _ in range(pad_size)]
                            secuencia_bloque = padding + historial_features_escaladas
                        else:
                            secuencia_bloque = historial_features_escaladas[-3:]
                        
                        secuencia = np.array([secuencia_bloque])
                        prob = model.predict(secuencia, verbose=0)[0][0]

                    # --- NUEVO FILTRO OPTIMIZADO: ESPERANZA MATEMÁTICA ---
                    # 1. Calculamos el Ratio Riesgo/Beneficio (RR) real de este escenario específico
                    rr_actual = abs((tp - entry) / (entry - sl))
                    
                    # 2. Aplicamos la fórmula: Esperanza = (P_win * RR) - (P_loss * 1.0)
                    esperanza_matematica = (prob * rr_actual) - ((1.0 - prob) * 1.0)
                    
                    # 3. El bot solo aprueba el trade si la esperanza matemática es positiva (> 0)
                    if esperanza_matematica <= 0.0:
                        strategy.hay_liquidez_tomada = False 
                        continue

                    # Si se aprueba la probabilidad, gestionamos el trade
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
                        final_r = abs((tp - entry) / (entry - sl)) if result == "TP" else -1.0
                        all_trades_global.append({'year': year, 'result': final_r, 'prob': prob, 'time': current_t})
                        
                        # --- GENERACIÓN DE GRÁFICO ---
                        try:
                            start_p = current_t - pd.Timedelta(hours=7)
                            end_p = exit_time + pd.Timedelta(minutes=30)
                            plot_data = df_5m.loc[df_5m.index >= start_p].loc[:end_p].copy()
                            
                            y_min = min(plot_data['low'].min(), tp, sl, p_l)
                            y_max = max(plot_data['high'].max(), tp, sl, p_h)
                            pad = (y_max - y_min) * 0.15

                            fig, axlist = mpf.plot(plot_data, type='candle', style='charles', figsize=(15, 9),
                                title=f"IA {tipo_modelo} ({prob:.2f}) | {tipo} | {result} | {final_r:.2f}R\nEntrada: {current_t}",
                                hlines=dict(hlines=[tp, sl, entry, p_h, p_l], 
                                            colors=['#26a69a', '#ef5350', '#787b86', 'blue', 'blue'], 
                                            alpha=0.4, linestyle='--'),
                                ylim=(y_min-pad, y_max+pad), returnfig=True,
                                datetime_format='%d/%m %H:%M')
                            
                            ax = axlist[0]
                            idx_in = plot_data.index.get_loc(current_t)
                            idx_out = plot_data.index.get_loc(exit_time)
                            ancho = idx_out - idx_in

                            if tipo == "SHORT":
                                ax.add_patch(Rectangle((idx_in, entry), ancho, sl-entry, facecolor='#ef5350', alpha=0.15))
                                ax.add_patch(Rectangle((idx_in, tp), ancho, entry-tp, facecolor='#26a69a', alpha=0.15))
                            else:
                                ax.add_patch(Rectangle((idx_in, sl), ancho, entry-sl, facecolor='#ef5350', alpha=0.15))
                                ax.add_patch(Rectangle((idx_in, entry), ancho, tp-entry, facecolor='#26a69a', alpha=0.15))

                            folder = "ganadores" if result == "TP" else "perdedores"
                            file_name = f"trade_{tipo_modelo}_{trade_id}.png"
                            fig.savefig(os.path.join(year_folder, folder, file_name))
                            plt.close(fig)
                        except: plt.close('all')

                        print(f"   [{tipo_modelo} OK] Confianza: {prob:.2f} | {result} ({final_r:.2f}R)")
                        trade_id += 1
                        strategy.hay_liquidez_tomada = False
                        break
    
    imprimir_reporte_detallado(all_trades_global, tipo_modelo, output_base)

def imprimir_reporte_detallado(all_trades, tipo_modelo, output_base):
    if not all_trades:
        print(f"\n[!] No se ejecutaron trades con el modelo {tipo_modelo} en ningún año.")
        return

    df = pd.DataFrame(all_trades)
    df = df.sort_values('time') # Aseguramos un estricto orden cronológico 2015-2025

    print("\n" + "="*70)
    print(f" REPORTE MULTIANUAL FINAL - ARQUITECTURA: {tipo_modelo}")
    print("="*70)
    print(f"{'AÑO':<7} | {'TRADES':<10} | {'WINRATE':<12} | {'PROFIT (R)':<10}")
    print("-" * 70)

    for año in sorted(df['year'].unique()):
        d_y = df[df['year'] == año]
        total_y = len(d_y)
        ganadores = len(d_y[d_y['result'] > 0])
        wr_y = (ganadores / total_y) * 100 if total_y > 0 else 0
        p_y = d_y['result'].sum()
        print(f"{año:<7} | {total_y:<3} trades   | {wr_y:>8.2f}% WR  | {p_y:>10.2f} R")

    print("="*70)
    total_trades = len(df)
    win_rate_global = (len(df[df['result'] > 0]) / total_trades) * 100
    profit_total_r = df['result'].sum()
    print(f"TOTAL GLOBAL {tipo_modelo}: {total_trades} trades | WR: {win_rate_global:.2f}% | PROFIT: {profit_total_r:.2f}R")

    # --- GENERACIÓN DE LA CURVA DE EQUIDAD ACUMULADA ---
    df['profit_acumulado'] = df['result'].cumsum()

    plt.figure(figsize=(14, 7))
    plt.plot(df['time'], df['profit_acumulado'], color='#26a69a', linewidth=2, label='Retorno Acumulado (R)')
    plt.fill_between(df['time'], df['profit_acumulado'], color='#26a69a', alpha=0.1)
    
    plt.title(f"Curva de Equidad Histórica ({tipo_modelo}) - Periodo 2015--2025", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Línea Temporal (Años)", fontsize=11)
    plt.ylabel("Rendimiento Acumulado en Unidades de Riesgo (R)", fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='upper left')
    
    # Ajuste dinámico de márgenes
    plt.tight_layout()
    
    # Guardar gráfico final en la carpeta de resultados
    grafico_path = os.path.join(output_base, f"curva_equidad_{tipo_modelo}.png")
    plt.savefig(grafico_path, dpi=300)
    plt.close()
    print(f"\n[✓] Curva de equidad generada y protegida con éxito en: {grafico_path}")

if __name__ == "__main__":
    AÑOS = [2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025]
    BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # =====================================================================
    # CONFIGURACIÓN DEL MODELO A EJECUTAR 
    # Elige entre: "MLP", "RF" o "LSTM"
    # =====================================================================
    MODELO_ACTIVO = "MLP"  
    
    run_ia_backtest(AÑOS, [8], os.path.join(BASE, "data"), os.path.join(BASE, "results"), tipo_modelo=MODELO_ACTIVO)