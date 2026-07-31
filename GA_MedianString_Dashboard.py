import streamlit as st
import pandas as pd
import plotly.express as px
import glob
import os

# Configuración de la página
st.set_page_config(page_title="GA Median String Dashboard", layout="wide")
st.title("🧬 Dashboard de Resultados: Algoritmo Genético Asimétrico")

# ==========================================
# 1. Búsqueda y Carga de Archivos (CON RUTAS DINÁMICAS)
# ==========================================

# Obtenemos la ruta absoluta de la carpeta exacta donde vive este script .py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Buscar archivos CSV disponibles en esa carpeta exacta
ruta_busqueda_csv = os.path.join(BASE_DIR, "*.csv")
archivos_csv = glob.glob(ruta_busqueda_csv)

if not archivos_csv:
    st.error("No se encontraron archivos CSV en el directorio del script.")
    st.stop()

# Extraer solo los nombres de los archivos para que el menú no se vea feo con la ruta completa
nombres_csv = [os.path.basename(f) for f in archivos_csv]

st.sidebar.header("📁 Carga de Datos")
archivo_sel_nombre = st.sidebar.selectbox("Selecciona el archivo de resultados:", nombres_csv)

# Reconstruir la ruta completa al archivo seleccionado
archivo_sel_ruta = os.path.join(BASE_DIR, archivo_sel_nombre)

@st.cache_data
def load_data(file_path):
    df = pd.read_csv(file_path)
    
    # --- CORRECCIÓN DE PARÁMETROS ---
    if "Padres_Conf" in df.columns:
        df["Padres_Conf"] = df["Padres_Conf"].apply(lambda x: 20 if float(x) == 2 else x)
        
    if "Mutacion_Conf" in df.columns:
        df["Mutacion_Conf"] = df["Mutacion_Conf"].apply(lambda x: 0.5 if float(x) == 5 else x)
        
    if "Padres_Conf" in df.columns and "Mutacion_Conf" in df.columns:
        df["Config_ID"] = "P" + df["Padres_Conf"].astype(str) + "_M" + df["Mutacion_Conf"].astype(str)
        
    return df

@st.cache_data
def load_referencias():
    # Usar la ruta absoluta también para las referencias
    ruta_referencias = os.path.join(BASE_DIR, "referencias.txt")
    
    if os.path.exists(ruta_referencias):
        try:
            return pd.read_csv(ruta_referencias, sep=None, engine='python')
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()

# Cargar usando las rutas corregidas
df_base = load_data(archivo_sel_ruta)
df_refs = load_referencias()

if df_base.empty:
    st.error(f"El archivo {archivo_sel_nombre} está vacío o no se pudo leer.")
    st.stop()
