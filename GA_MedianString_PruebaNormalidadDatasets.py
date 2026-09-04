import os
from scipy import stats

def evaluar_normalidad_datasets(ruta_directorio):
    """
    Lee todos los archivos en el directorio indicado, calcula las longitudes 
    de las cadenas y aplica la prueba de normalidad de Shapiro-Wilk.
    """
    if not os.path.exists(ruta_directorio):
        print(f"Error: No se encontró el directorio '{ruta_directorio}'.")
        return

    # Obtener lista de archivos y ordenarlos alfabéticamente
    archivos = sorted([f for f in os.listdir(ruta_directorio) if os.path.isfile(os.path.join(ruta_directorio, f))])

    if not archivos:
        print(f"El directorio '{ruta_directorio}' está vacío.")
        return

    # Encabezado de la tabla de resultados
    print(f"{'Dataset':<15} | {'Cadenas':<8} | {'Estadístico (W)':<15} | {'P-Value':<12} | {'¿Es Normal?'}")
    print("-" * 75)

    for nombre_archivo in archivos:
        ruta_archivo = os.path.join(ruta_directorio, nombre_archivo)
        longitudes = []
        
        try:
            with open(ruta_archivo, 'r') as file:
                for linea in file:
                    linea = linea.strip()
                    if not linea:
                        continue # Saltar líneas vacías
                    
                    # Separar por espacios. Ejemplo: "A 33344455..." -> ["A", "33344455..."]
                    partes = linea.split()
                    
                    # Tomar la última parte asumiendo que es la cadena de Freeman
                    cadena_freeman = partes[-1] 
                    longitudes.append(len(cadena_freeman))
                    
        except Exception as e:
            print(f"{nombre_archivo:<15} | Error al procesar: {e}")
            continue

        num_cadenas = len(longitudes)
        
        # Shapiro-Wilk requiere al menos 3 datos
        if num_cadenas < 3:
            print(f"{nombre_archivo:<15} | {num_cadenas:<8} | N/A             | N/A          | Datos insuficientes")
            continue

        # --- PRUEBA DE NORMALIDAD DE SHAPIRO-WILK ---
        stat, p_value = stats.shapiro(longitudes)

        # Interpretación (Nivel de significancia Alpha = 0.05)
        # Si p_value > 0.05, NO podemos rechazar que sea normal.
        # Si p_value <= 0.05, SE RECHAZA la normalidad.
        es_normal = "SÍ" if p_value > 0.05 else "NO (Asimétrica)"

        # Imprimir fila de resultados
        print(f"{nombre_archivo:<15} | {num_cadenas:<8} | {stat:<15.4f} | {p_value:<12.4e} | {es_normal}")

# ==========================================
# INSTRUCCIONES DE EJECUCIÓN
# ==========================================
# 1. Asegúrate de tener instalada la librería scipy: pip install scipy
ruta_datasets = "./datasets" 

print("Iniciando análisis de normalidad de longitudes...\n")
evaluar_normalidad_datasets(ruta_datasets)