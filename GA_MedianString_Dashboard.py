import streamlit as st
import pandas as pd
import plotly.express as px
import glob
import os

# Configuración de la página
st.set_page_config(page_title="GA Median String Dashboard", layout="wide")
st.title("🧬 Dashboard de Resultados: Algoritmos Genéticos para el Cálculo de la Cadena Media")

# ==========================================
# 1. BÚSQUEDA UNIVERSAL DE ARCHIVOS
# ==========================================

# Buscar archivos CSV en TODAS las subcarpetas del repositorio
# El parámetro recursive=True obliga a buscar en todos lados para evitar errores de rutas
archivos_csv = glob.glob("**/*.csv", recursive=True)

if not archivos_csv:
    st.error("No se encontraron archivos CSV en el repositorio.")
    st.info(f"Ruta actual del servidor: {os.getcwd()}") # Útil para depurar si falla en la nube
    st.stop()

st.sidebar.header("📁 Carga de Datos")
# Mostramos la ruta relativa completa en el menú para saber exactamente qué se está leyendo
archivo_sel_ruta = st.sidebar.selectbox("Selecciona el archivo de resultados:", archivos_csv)

@st.cache_data
def load_data(file_path):
    df = pd.read_csv(file_path)
    
    # --- CORRECCIÓN DE PARÁMETROS ---
    # Convertir Padres: Si viene como 2 (por 0.2), lo pasamos a 20; el 25 se queda en 25
    if "Padres_Conf" in df.columns:
        df["Padres_Conf"] = df["Padres_Conf"].apply(lambda x: 20 if float(x) == 2 else x)
        
    # Convertir Mutación: Corregir el 5 para que sea 0.5
    if "Mutacion_Conf" in df.columns:
        df["Mutacion_Conf"] = df["Mutacion_Conf"].apply(lambda x: 0.5 if float(x) == 5 else x)
        
    # Reconstruir Config_ID limpio (eliminando cualquier referencia al torneo K)
    # Ej: "P20_M0.5"
    if "Padres_Conf" in df.columns and "Mutacion_Conf" in df.columns:
        df["Config_ID"] = "P" + df["Padres_Conf"].astype(str) + "_M" + df["Mutacion_Conf"].astype(str)
        
    return df

@st.cache_data
def load_referencias():
    # Búsqueda universal de referencias.txt en todas las carpetas
    rutas_refs = glob.glob("**/referencias.txt", recursive=True)
    
    if rutas_refs: # Si encontró al menos uno
        ruta_correcta = rutas_refs[0]
        try:
            # engine='python' y sep=None permiten que pandas detecte el separador (comas, tabs, etc.)
            return pd.read_csv(ruta_correcta, sep=None, engine='python')
        except Exception as e:
            st.sidebar.error(f"Error leyendo {ruta_correcta}: {e}")
            return pd.DataFrame()
    return pd.DataFrame()

# Cargar los dataframes usando las rutas encontradas dinámicamente
df_base = load_data(archivo_sel_ruta)
df_refs = load_referencias()

if df_base.empty:
    st.error(f"El archivo {archivo_sel_ruta} está vacío o no se pudo leer.")
    st.stop()

# ==========================================
# 2. PANEL LATERAL (Filtros)
# ==========================================
st.sidebar.markdown("---")
st.sidebar.header("🎛️ Parámetros de Selección")

# Obtener lista ordenada de Datasets
datasets_disp = sorted(df_base["Dataset"].unique())
dataset_sel = st.sidebar.selectbox("1. Selecciona el Dataset de Análisis", datasets_disp)

# Opciones de Visualización
st.sidebar.markdown("---")
st.sidebar.subheader("📐 Opciones de Visualización")
modo_agrupacion = st.sidebar.radio(
    "¿Cómo deseas agrupar la curva de convergencia?",
    options=["Detallado (Por Configuración ID)", "Consolidado (Promedio global)"]
)

# Filtros Dinámicos (Sin el Torneo K)
padres_opts = sorted(list(df_base["Padres_Conf"].unique()))
padres_sel = st.sidebar.multiselect("2. Porcentaje de Padres (%)", options=padres_opts, default=padres_opts)

mutacion_opts = sorted(list(df_base["Mutacion_Conf"].unique()))
mutacion_sel = st.sidebar.multiselect("3. Porcentaje de Mutación (%)", options=mutacion_opts, default=mutacion_opts)

# Filtrar el dataframe completo según los parámetros seleccionados
df_filtered_global = df_base[
    (df_base["Padres_Conf"].isin(padres_sel)) &
    (df_base["Mutacion_Conf"].isin(mutacion_sel))
]

# Filtrar específicamente para el dataset seleccionado (Para gráficas A y B)
df_plot_dataset = df_filtered_global[df_filtered_global["Dataset"] == dataset_sel]

st.sidebar.markdown("---")
st.sidebar.info("💡 **Tip para el artículo:** Sitúa el cursor sobre las gráficas y haz clic en el ícono de cámara para exportar en calidad PNG.")


# ==========================================
# 3. PANEL PRINCIPAL: Métricas Consolidadas
# ==========================================
if not df_plot_dataset.empty:
    # Calcular tiempo total sumando el snapshot final de cada semilla y configuración (para el dataset seleccionado)
    tiempo_por_ejecucion = df_plot_dataset.groupby(['Config_ID', 'Semilla'])['Tiempo_Segundos'].max()
    tiempo_total_dataset = tiempo_por_ejecucion.sum()
    
    st.metric(
        label=f"⏱️ Tiempo Total de Experimentación (Dataset {dataset_sel})", 
        value=f"{tiempo_total_dataset:,.2f} Segundos",
        help="Suma del tiempo final registrado por cada configuración y semilla del dataset seleccionado."
    )
    st.markdown("---")

# ==========================================
# 4. PANEL PRINCIPAL: Gráficas de Dataset
# ==========================================
if df_plot_dataset.empty:
    st.warning("No hay datos para esta combinación de filtros.")
else:
    col1, col2 = st.columns(2)
    
    # -----------------------------------------------------
    # GRÁFICA A: Convergencia (Dataset específico)
    # -----------------------------------------------------
    with col1:
        st.subheader(f"📉 Convergencia (Dataset {dataset_sel})")
        
        if modo_agrupacion == "Consolidado (Promedio global)":
            df_conv_plot = df_plot_dataset.groupby('Generacion', as_index=False)['Mejor_Costo_Medio'].mean()
            df_conv_plot['Linea_Agrupada'] = f"Promedio (Dataset {dataset_sel})"
            color_col = 'Linea_Agrupada'
            titulo_leyenda = "Agrupación"
        else:
            df_conv_plot = df_plot_dataset.groupby(['Config_ID', 'Generacion'], as_index=False)['Mejor_Costo_Medio'].mean()
            color_col = 'Config_ID'
            titulo_leyenda = "Configuración"

        fig_conv = px.line(
            df_conv_plot, 
            x="Generacion", 
            y="Mejor_Costo_Medio", 
            color=color_col,
            title=f"Evolución del Costo para Dataset {dataset_sel}",
            labels={"Generacion": "Generaciones", "Mejor_Costo_Medio": "Mejor Costo Promedio"},
            markers=True
        )
        fig_conv.update_layout(template="plotly_white", hovermode="x unified", legend_title_text=titulo_leyenda)
        st.plotly_chart(fig_conv, use_container_width=True)

    # -----------------------------------------------------
    # GRÁFICA B: Análisis del Tiempo vs Parámetros
    # -----------------------------------------------------
    with col2:
        st.subheader("⏱️ Tiempo Total por Ejecución")
        
        # Obtenemos el tiempo final (máxima generación) por ejecución independiente
        df_time = df_plot_dataset.groupby(['Config_ID', 'Semilla', 'Mutacion_Conf', 'Padres_Conf'], as_index=False)['Tiempo_Segundos'].max()
        
        parametro_x = st.selectbox(
            "Evaluar dispersión de tiempo en función de:",
            options=["Config_ID", "Mutacion_Conf", "Padres_Conf"],
            index=0
        )

        fig_time = px.box(
            df_time, 
            x=parametro_x, 
            y="Tiempo_Segundos", 
            color=parametro_x if parametro_x != "Config_ID" else "Config_ID",
            title=f"Distribución del Tiempo Total según {parametro_x.replace('_Conf', '')}",
            labels={parametro_x: parametro_x.replace("_", " ").title(), "Tiempo_Segundos": "Tiempo Total (Segundos)"}
        )
        fig_time.update_layout(template="plotly_white", showlegend=(parametro_x != "Config_ID"))
        st.plotly_chart(fig_time, use_container_width=True)

# ==========================================
# 5. GRÁFICA GLOBAL (Sumatoria Todos los Datasets)
# ==========================================
st.markdown("---")
st.subheader("🌐 Convergencia Global Acumulada (Suma Datasets A - Z)")
st.caption("Esta gráfica muestra la sumatoria del costo promedio alcanzado en todos los datasets simultáneamente para cada iteración.")

if not df_filtered_global.empty:
    # 1. Promediamos las distintas semillas para evitar duplicados en la suma (agrupando por Config, Dataset y Gen)
    df_global_avg_seed = df_filtered_global.groupby(['Config_ID', 'Dataset', 'Generacion'], as_index=False)['Mejor_Costo_Medio'].mean()
    
    # 2. Sumamos transversalmente (a través de todos los Datasets) para la misma generación y configuración
    df_global_sum = df_global_avg_seed.groupby(['Config_ID', 'Generacion'], as_index=False)['Mejor_Costo_Medio'].sum()
    
    fig_global = px.line(
        df_global_sum,
        x="Generacion",
        y="Mejor_Costo_Medio",
        color="Config_ID",
        title="Sumatoria Total de Costos por Generación",
        labels={"Generacion": "Generaciones", "Mejor_Costo_Medio": "Sumatoria de Distancias Óptimas"},
        markers=True
    )
    fig_global.update_layout(template="plotly_white", hovermode="x unified")
    st.plotly_chart(fig_global, use_container_width=True)

# ==========================================
# 6. TABLA COMPARATIVA VS ESTADO DEL ARTE
# ==========================================
st.markdown("---")
st.subheader("🏆 Comparativa de Mejores Resultados (Estado del Arte)")

# Obtener el mínimo absoluto (mejor distancia) alcanzado para cada dataset entre las configs seleccionadas
mejor_ga_absoluto = df_filtered_global.groupby('Dataset')['Mejor_Costo_Medio'].min().reset_index()
mejor_ga_absoluto.rename(columns={'Mejor_Costo_Medio': 'Nuestro Algoritmo (Mejor Caso)'}, inplace=True)

if not df_refs.empty:
    # Validar que exista la columna 'Dataset' (o 'dataset') para poder hacer el cruce
    col_dataset = None
    if 'Dataset' in df_refs.columns:
        col_dataset = 'Dataset'
    elif 'dataset' in df_refs.columns:
        col_dataset = 'dataset'
        df_refs.rename(columns={'dataset': 'Dataset'}, inplace=True)
        col_dataset = 'Dataset'

    if col_dataset:
        # Fusionamos referencias con el mejor resultado nuestro
        tabla_comparativa = pd.merge(mejor_ga_absoluto, df_refs, on='Dataset', how='left')
        st.dataframe(tabla_comparativa, use_container_width=True)
    else:
        st.warning("El archivo 'referencias.txt' se cargó, pero no tiene una columna llamada 'Dataset' para hacer la comparación cruzada.")
        st.dataframe(mejor_ga_absoluto, use_container_width=True)
else:
    st.info("No se encontró o no se pudo cargar el archivo 'referencias.txt'. Mostrando solo los mejores resultados alcanzados por nuestro algoritmo.")
    st.dataframe(mejor_ga_absoluto, use_container_width=True)
