import os
import shutil
import pandas as pd

def setup_folders(output_folder="trades"):
    """Limpia y crea la estructura de carpetas para los resultados."""
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    os.makedirs(f"{output_folder}/ganadores", exist_ok=True)
    os.makedirs(f"{output_folder}/perdedores", exist_ok=True)

def clean_price(val):
    """Limpia el formato de precio del CSV (ej: 2.625.098.000 -> 2625.09)."""
    if isinstance(val, str):
        clean = val.replace('.', '')
        return float(clean) / 1000000 
    return val

def load_data(file_path):  
    """Carga el CSV, limpia precios y establece el índice temporal."""
    # Añadimos "volume" al final de names para absorber el último valor (;0) del ASCII
    df = pd.read_csv(file_path, sep=';', header=None, 
                     names=["timestamp", "open", "high", "low", "close", "volume"])
    
    for col in ["open", "high", "low", "close"]:
        df[col] = df[col].apply(clean_price)

    df["timestamp"] = pd.to_datetime(df["timestamp"], format="%Y%m%d %H%M%S")
    df.set_index("timestamp", inplace=True)
    return df

