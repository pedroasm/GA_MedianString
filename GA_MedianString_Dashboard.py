import streamlit as st
import pandas as pd
import plotly.express as px

# Configuración de la página
st.set_page_config(page_title="GA Median String Dashboard", layout="wide")
st.title("🧬 Dashboard de Resultados: Genetic Algorithm DoE")

# 1. Carga de datos
@st.cache_data
def load_data():
    try:
        return pd.DataFrame(pd.read_csv("resultados_completos.csv"))
    except FileNotFoundError:
        return pd.DataFrame()

df = load_data()

if df.empty:
    st.error("No se encontró 'resultados_completos.csv'. Ejecuta primero el generador de CSV.")
    st.stop()

# 2. Panel lateral (Sidebar) para Filtros Múltiples y Opciones de Consolidación
st.sidebar.header("🎛️ Parámetros de Selección")

# Filtro de Dataset (Selección única)
datasets_disp = df["Dataset"].unique()
dataset_sel = st.sidebar.selectbox("1. Selecciona el Dataset", datasets_disp)

# Filtrar el df temporalmente por dataset
df_filtered = df[df["Dataset"] == dataset_sel]

# --- NUEVA OPCIÓN: Consolidación / Agrupación en los Filtros ---
st.sidebar.markdown("---")
st.sidebar.subheader("📐 Opciones de Visualización")
modo_agrupacion = st.sidebar.radio(
    "¿Cómo deseas agrupar las líneas en el gráfico?",
    options=["Detallado (Por Configuración ID)", "Consolidado (Promedio de las selecciones)"]
)

# Filtros Dinámicos
padres_opts = list(df_filtered["Padres_Conf"].unique())
padres_sel = st.sidebar.multiselect("2. Config. Padres", options=padres_opts, default=padres_opts)

torneo_opts = list(df_filtered["Torneo_K"].unique())
torneo_sel = st.sidebar.multiselect("3. Torneo (K)", options=torneo_opts, default=torneo_opts)

mutacion_opts = list(df_filtered["Mutacion_Conf"].unique())
mutacion_sel = st.sidebar.multiselect("4. Mutación", options=mutacion_opts, default=mutacion_opts)

# Aplicar los filtros base
df_plot = df_filtered[
    (df_filtered["Padres_Conf"].isin(padres_sel)) &
    (df_filtered["Torneo_K"].isin(torneo_sel)) &
    (df_filtered["Mutacion_Conf"].isin(mutacion_sel))
]

st.sidebar.markdown("---")
st.sidebar.info("💡 **Tip para el artículo:** Sitúa el cursor sobre la gráfica y haz clic en el icono de cámara (Download plot) para guardar un PNG.")

# 3. Panel Principal de Gráficas
if df_plot.empty:
    st.warning("No hay datos para esta combinación de filtros.")
else:
    col1, col2 = st.columns(2)
    
    # -----------------------------------------------------
    # GRÁFICA A: Convergencia (Generación vs Mejor Costo)
    # -----------------------------------------------------
    with col1:
        st.subheader("📉 Curva de Convergencia (Mejor Individuo)")
        
        if modo_agrupacion == "Consolidado (Promedio de las selecciones)":
            # Promediamos todas las configuraciones seleccionadas en una sola línea global
            df_conv_plot = df_plot.groupby('Generacion', as_index=False)['Mejor_Costo_Medio'].mean()
            df_conv_plot['Linea_Agrupada'] = f"Promedio Global (Dataset {dataset_sel})"
            color_col = 'Linea_Agrupada'
            titulo_leyenda = "Agrupación"
        else:
            # Agrupamos por configuración y generación (promediando las semillas de cada config)
            df_conv_plot = df_plot.groupby(['Config_ID', 'Generacion'], as_index=False)['Mejor_Costo_Medio'].mean()
            color_col = 'Config_ID'
            titulo_leyenda = "Configuración"

        fig_conv = px.line(
            df_conv_plot, 
            x="Generacion", 
            y="Mejor_Costo_Medio", 
            color=color_col,
            title=f"Evolución del Costo para Dataset {dataset_sel}", # (Semillas eliminadas del título)
            labels={"Generacion": "Generaciones", "Mejor_Costo_Medio": "Mejor Costo Promedio (Suma Edit Dist / N)"},
            markers=True
        )
        fig_conv.update_layout(template="plotly_white", hovermode="x unified", legend_title_text=titulo_leyenda)
        st.plotly_chart(fig_conv, width='stretch')

    # -----------------------------------------------------
    # GRÁFICA B: Análisis del Tiempo vs Parámetros
    # -----------------------------------------------------
    with col2:
        st.subheader("⏱️ Tiempo Total de Iteraciones")
        
        # Obtenemos el tiempo final (máxima generación registrada) por cada ejecución independiente (Semilla + Config)
        df_time = df_plot.groupby(['Config_ID', 'Semilla', 'Mutacion_Conf', 'Padres_Conf', 'Torneo_K'], as_index=False)['Tiempo_Segundos'].max()
        
        # Selector interactivo para decidir qué parámetro evaluar en el eje X de la gráfica de tiempo
        parametro_x = st.selectbox(
            "Evaluar cambio de tiempo en función de:",
            options=["Config_ID", "Mutacion_Conf", "Padres_Conf", "Torneo_K"],
            index=0
        )

        fig_time = px.box(
            df_time, 
            x=parametro_x, 
            y="Tiempo_Segundos", 
            color=parametro_x if parametro_x != "Config_ID" else "Config_ID",
            title=f"Distribución del Tiempo Total según {parametro_x}",
            labels={parametro_x: parametro_x.replace("_", " ").title(), "Tiempo_Segundos": "Tiempo Total (Segundos)"}
        )
        fig_time.update_layout(template="plotly_white", showlegend=(parametro_x != "Config_ID"))
        st.plotly_chart(fig_time, width='stretch')

    # -----------------------------------------------------
    # TABLA RESUMEN INFERIOR
    # -----------------------------------------------------
    st.markdown("---")
    st.subheader("📋 Resumen Estadístico del Óptimo Alcanzado")
    
    # Tomar la última generación de cada experimento y calcular medias/desviaciones estándar
    ultimas_gen = df_plot.sort_values("Generacion").groupby(['Config_ID', 'Semilla']).tail(1)
    
    tabla_resumen = ultimas_gen.groupby("Config_ID").agg(
        Media_Costo_Optimo=("Mejor_Costo_Medio", "mean"),
        Std_Costo_Optimo=("Mejor_Costo_Medio", "std"),
        Tiempo_Medio_Seg=("Tiempo_Segundos", "mean"),
        Ejecuciones_Exitosas=("Semilla", "count")
    ).reset_index()
    
    st.dataframe(tabla_resumen, width='stretch')