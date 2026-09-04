import os
import time
import glob
import random
import sys
import math
import numpy as np
import pygad
import argparse
from numba import cuda, float32, int32

# =============================================================================
# 0. FUNCIONES AUXILIARES DE RUTA
# =============================================================================
def obtener_directorio_snapshots(dataset_id, pop_size, num_generaciones, prob_mutacion, 
                                 porc_padres, k_torneo, cota_inf_mult, cota_sup_mult, semilla, base_dir="snapshots"):
    """
    Genera una ruta única para guardar/cargar snapshots basada en todos los hiperparámetros.
    """
    mut_str = str(prob_mutacion).replace('.', '')
    padres_str = str(porc_padres).replace('.', '')
    cinf_str = str(cota_inf_mult).replace('.', '')
    csup_str = str(cota_sup_mult).replace('.', '')
    
    nombre_experimento = (
        f"DS_{dataset_id}_P{pop_size}_G{num_generaciones}_Pad{padres_str}_"
        f"K{k_torneo}_Mut{mut_str}_C{cinf_str}_{csup_str}_S{semilla}"
    )
    
    ruta_completa = os.path.join(base_dir, nombre_experimento)
    return ruta_completa


# =============================================================================
# 1. CONFIGURACIÓN GLOBAL Y CAPTURA DE PARÁMETROS (ARGPARSE)
# =============================================================================
parser = argparse.ArgumentParser(description="Algoritmo Genético Híbrido CUDA - DoE")

# --- Archivos y Rutas ---
parser.add_argument("--dataset", type=str, default="B", 
                    help="Nombre del archivo del dataset (Por defecto: B)")
parser.add_argument("--output_dir", type=str, default=None, 
                    help="Directorio de salvas. Si no se indica, se genera automáticamente.")
parser.add_argument("--frecuencia_snapshot", type=int, default=1000, 
                    help="Frecuencia para guardar snapshots (Por defecto: 1000)")

# --- Parámetros Genéticos ---
parser.add_argument("--proporcion", type=float, default=(1/20)*100, 
                    help="porciento del dataset usado para la población inicial (Por defecto: 5)")
parser.add_argument("--mutacion", type=float, default=0.02, 
                    help="Tasa de mutación decimal (Por defecto: 0.02)")
parser.add_argument("--generaciones", type=int, default=100000, 
                    help="Generaciones máximas del algoritmo (Por defecto: 100000)")
parser.add_argument("--padres", type=float, default=0.2, 
                    help="porciento de la población seleccionada como padres (Por defecto: 0.2)")
parser.add_argument("--torneo", type=int, default=2, 
                    help="Tamaño del torneo para selección de padres (Por defecto: 2)")
# NUEVO PARÁMETRO: Semilla
parser.add_argument("--semilla", type=int, default=42, 
                    help="Semilla aleatoria para reproducibilidad del experimento (Por defecto: 42)")

args = parser.parse_args()

# Congelar la aleatoriedad desde el principio
SEMILLA = args.semilla
random.seed(SEMILLA)
np.random.seed(SEMILLA)

# Asignación de argumentos
ARCHIVO_DATASET_ORIGINAL = args.dataset
NOMBRE_CORTO_DATASET = os.path.basename(ARCHIVO_DATASET_ORIGINAL).split('.')[0]
PORCENTAJE_MUTACION = args.mutacion
PORCENTAJE_PADRES = args.padres
K_TOURNAMENT = args.torneo
NUM_GENERATIONS = args.generaciones
FRECUENCIA_SNAPSHOT = args.frecuencia_snapshot
PROPORCION = args.proporcion

# Cotas de penalización asimétrica (Multiplicadores)
COTA_INF_MULT = 0.70 # cambiar en dependencia de la distribución del dataset
COTA_SUP_MULT = 1.30 # cambiar en dependencia de la distribución del dataset

# =============================================================================
# 2. CARGA DE DATOS ANTICIPADA (Necesaria para definir población y rutas)
# =============================================================================
print(f"\n1. Leyendo datos del archivo {ARCHIVO_DATASET_ORIGINAL} ...")
with open(ARCHIVO_DATASET_ORIGINAL, 'r', encoding='utf-8') as f:
    datos_brutos = np.loadtxt((s[2:] for s in f), dtype=str)
    datos = datos_brutos.tolist() if datos_brutos.ndim > 0 else [str(datos_brutos)]

longitudes_dataset = [len(s) for s in datos]
LONGITUD_PROMEDIO = sum(longitudes_dataset) / len(longitudes_dataset)

# Calcular tamaño de la población objetivo
if PROPORCION >= 100:
    poblacion_objetivo = len(datos)
else:
    poblacion_objetivo = (int(len(datos) * PROPORCION)) // 100
poblacion_objetivo = max(2, poblacion_objetivo)  # Seguridad mínima

# Configuración dinámica del directorio con todos los parámetros (Huella dactilar)
if args.output_dir is not None:
    DIRECTORIO_OUTPUT = args.output_dir
else:
    DIRECTORIO_OUTPUT = obtener_directorio_snapshots(
        dataset_id=NOMBRE_CORTO_DATASET,
        pop_size=poblacion_objetivo,
        num_generaciones=NUM_GENERATIONS,
        prob_mutacion=PORCENTAJE_MUTACION,
        porc_padres=PORCENTAJE_PADRES,
        k_torneo=K_TOURNAMENT,
        cota_inf_mult=COTA_INF_MULT,
        cota_sup_mult=COTA_SUP_MULT,
        semilla=SEMILLA
    )
os.makedirs(DIRECTORIO_OUTPUT, exist_ok=True)

print("="*60)
print("⚙️ CONFIGURACIÓN DEL EXPERIMENTO (DoE)")
print(f"Dataset:       {NOMBRE_CORTO_DATASET} (Total: {len(datos)} cadenas)")
print(f"Semilla:       {SEMILLA}")
print(f"Directorio:    {DIRECTORIO_OUTPUT}")
print(f"Población:     {poblacion_objetivo} individuos (Prop: {PROPORCION}%)")
print(f"Mutación:      {PORCENTAJE_MUTACION}")
print(f"Padres:        {PORCENTAJE_PADRES}")
print(f"Torneo (K):    {K_TOURNAMENT}")
print(f"Generaciones:  {NUM_GENERATIONS}")
print(f"Cota Inferior: {COTA_INF_MULT} * Media")
print(f"Cota Superior: {COTA_SUP_MULT} * Media")
print("="*60)

# =============================================================================
# 3. CONFIGURACIÓN DEL ENTORNO CUDA Y NVVM
# =============================================================================
# Parámetros de Arquitectura
MAX_LEN = max(512, max(longitudes_dataset))

# Variables globales dinámicas para el control de snapshots
generacion_inicio = 0
tiempo_acumulado = 0.0
secuencia_snapshot_actual = 0

os.environ['CUDA_HOME'] = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4"
nvvm_bin_path = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\nvvm\bin"
os.add_dll_directory(nvvm_bin_path)

dll_path_real = os.path.join(nvvm_bin_path, "nvvm64_40_0.dll")
dll_path_fake = os.path.join(nvvm_bin_path, "nvvm.dll")
if os.path.exists(dll_path_real) and not os.path.exists(dll_path_fake):
    try:
        import shutil
        shutil.copy(dll_path_real, dll_path_fake)
    except Exception:
        pass

# =============================================================================
# 4. KERNEL CUDA Y MATRIZ DE ALINEACIÓN
# =============================================================================
alphabet = ['ε', '0', '1', '2', '3', '4', '5', '6', '7']
ALFABETO_DICT = {c: i for i, c in enumerate(alphabet)}

W = np.array([
    [0,  2,  2,  2,  2,  2,  2,  2,  2],
    [2,  0,  1,  2,  3,  4,  3,  2,  1],
    [2,  1,  0,  1,  2,  3,  4,  3,  2],
    [2,  2,  1,  0,  1,  2,  3,  4,  3],
    [2,  3,  2,  1,  0,  1,  2,  3,  4],
    [2,  4,  3,  2,  1,  0,  1,  2,  3],
    [2,  3,  4,  3,  2,  1,  0,  1,  2],
    [2,  2,  3,  4,  3,  2,  1,  0,  1],
    [2,  1,  2,  3,  4,  3,  2,  1,  0],
], dtype=np.float32)

@cuda.jit
def levenshtein_cuda_kernel(pop_matrix, pop_lens, data_matrix, data_lens, W, out_distances):
    idx = cuda.grid(1)
    total_pares = pop_matrix.shape[0] * data_matrix.shape[0]
    
    if idx >= total_pares:
        return
        
    pop_idx = idx // data_matrix.shape[0]
    data_idx = idx % data_matrix.shape[0]
    
    m = pop_lens[pop_idx]
    n = data_lens[data_idx]
    e = 0 
    
    v0 = cuda.local.array(MAX_LEN, float32)
    v1 = cuda.local.array(MAX_LEN, float32)
    
    v0[0] = 0.0
    for j in range(1, n + 1):
        b = data_matrix[data_idx, j - 1]
        v0[j] = v0[j - 1] + W[e, b]
        
    for i in range(1, m + 1):
        a = pop_matrix[pop_idx, i - 1]
        v1[0] = v0[0] + W[a, e]
        
        for j in range(1, n + 1):
            b = data_matrix[data_idx, j - 1]
            
            delete = v0[j] + W[a, e]
            insert = v1[j - 1] + W[e, b]
            substitute = v0[j - 1] + W[a, b]
            
            min_val = delete
            if insert < min_val: min_val = insert
            if substitute < min_val: min_val = substitute
                
            v1[j] = min_val
            
        for k in range(n + 1):
            v0[k] = v1[k]
            
    out_distances[idx] = v0[n]

def codificar_cadenas(lista_cadenas):
    n_cadenas = len(lista_cadenas)
    matriz = np.zeros((n_cadenas, MAX_LEN), dtype=np.int32)
    longitudes = np.zeros(n_cadenas, dtype=np.int32)
    
    for i, cadena in enumerate(lista_cadenas):
        longitudes[i] = min(len(cadena), MAX_LEN)
        for j, char in enumerate(cadena):
            if j < MAX_LEN:
                matriz[i, j] = ALFABETO_DICT[char]
                
    return matriz, longitudes

def evaluar_poblacion_gpu(poblacion_strings):
    n_pop = len(poblacion_strings)
    n_data = len(datos)
    total_hilos = n_pop * n_data 
    
    pop_matriz, pop_lens = codificar_cadenas(poblacion_strings)
    
    d_pop_matrix = cuda.to_device(pop_matriz)
    d_pop_lens = cuda.to_device(pop_lens)
    d_out = cuda.device_array(total_hilos, dtype=np.float32)
    
    hilos_por_bloque = 256
    bloques_por_grid = math.ceil(total_hilos / hilos_por_bloque)
    
    levenshtein_cuda_kernel[bloques_por_grid, hilos_por_bloque](
        d_pop_matrix, d_pop_lens, d_data_matrix, d_data_lens, d_W, d_out
    )
    
    out_host = d_out.copy_to_host()
    out_matrix = out_host.reshape((n_pop, n_data))
    costos_finales = np.sum(out_matrix, axis=1)
    
    return costos_finales

# =============================================================================
# 5. PRECARGA EN LA GRÁFICA
# =============================================================================
print(f"2. Subiendo dataset ({len(datos)} cadenas) a la VRAM...")
datos_matriz, datos_lens = codificar_cadenas(datos)
d_data_matrix = cuda.to_device(datos_matriz)
d_data_lens = cuda.to_device(datos_lens)
d_W = cuda.to_device(W)
print("   ¡Dataset precargado con éxito!")

# =============================================================================
# 6. POOL DE CADENAS Y LÓGICA GENÉTICA
# =============================================================================
strings_pool = []

def registrar_cadena(cadena):
    strings_pool.append(cadena)
    return len(strings_pool) - 1

def fitness_batch(ga_instance, population, population_idx):
    cadenas_poblacion = [strings_pool[int(ind[0])] for ind in population]
    costos = evaluar_poblacion_gpu(cadenas_poblacion)
    
    FACTOR_PENALIZACION_LONGITUD = len(datos) * 2 
    COTA_SUPERIOR = LONGITUD_PROMEDIO * COTA_SUP_MULT
    COTA_INFERIOR = LONGITUD_PROMEDIO * COTA_INF_MULT
    
    fitness_list = []
    for i, chi2 in enumerate(costos):
        
        longitud_ind = len(cadenas_poblacion[i])        
        penalizacion = 0.0
        
        if longitud_ind > COTA_SUPERIOR: 
            penalizacion = FACTOR_PENALIZACION_LONGITUD * ((longitud_ind - COTA_SUPERIOR) ** 2)
        elif longitud_ind < COTA_INFERIOR:
            penalizacion = FACTOR_PENALIZACION_LONGITUD * ((longitud_ind - COTA_INFERIOR) ** 2)
            
        costo_penalizado = chi2 + penalizacion
        fitness_list.append(-costo_penalizado)
        
    print(" 🚀 Evaluada generación en GPU masivamente.", end='\r')
    sys.stdout.flush()
    return fitness_list

def mutacion_porcentual_custom(offspring, ga_instance):
    for chromosome_idx in range(offspring.shape[0]):
        id_gen = int(offspring[chromosome_idx, 0])
        gen_str = strings_pool[id_gen]
        
        gen_list = list(gen_str)
        longitud_gen = len(gen_list)
        
        if longitud_gen == 0:
            continue
            
        num_mutaciones = int(longitud_gen * PORCENTAJE_MUTACION)
        num_mutaciones = max(1, num_mutaciones)
        
        indices_a_mutar = random.sample(range(longitud_gen), min(num_mutaciones, longitud_gen))
        indices_a_mutar.sort(reverse=True)
        
        for idx in indices_a_mutar:
            if idx >= len(gen_list):
                continue
                
            tipo_mutacion = random.random()
            if tipo_mutacion < 0.25:
                if len(gen_list) > 1:
                    gen_list.pop(idx)
            elif tipo_mutacion < 0.5:
                caracter_actual = gen_list[idx]
                gen_list.insert(idx + 1, caracter_actual)
            else:
                simbolo = gen_list[idx]
                if simbolo != 'ε':
                    valor_actual = int(simbolo)
                    signo = 1 if random.random() > 0.5 else -1
                    cambio = random.randint(1, 2)
                    nuevo_valor = valor_actual + cambio*signo
                    nuevo_valor %= 8
                    gen_list[idx] = str(nuevo_valor)
                    
        mutado_str = "".join(gen_list)
        offspring[chromosome_idx, 0] = registrar_cadena(mutado_str)

    return offspring

def cruzamiento_scattered_custom(parents, offspring_size, ga_instance):
    offspring = np.empty(offspring_size, dtype=int)

    for k in range(offspring_size[0]):
        parent1_idx = k % parents.shape[0]
        parent2_idx = (k + 1) % parents.shape[0]
        
        id_p1 = int(parents[parent1_idx, 0])
        id_p2 = int(parents[parent2_idx, 0])
        
        p1_str = strings_pool[id_p1]
        p2_str = strings_pool[id_p2]
        
        min_len = min(len(p1_str), len(p2_str))
        mask = [random.choice([0, 1]) for _ in range(min_len)]
        
        child_chars = []
        for i in range(min_len):
            if mask[i] == 1:
                child_chars.append(p1_str[i])
            else:
                child_chars.append(p2_str[i])
                
        if len(p1_str) > min_len:
            child_chars.extend(p1_str[min_len:])
        elif len(p2_str) > min_len:
            child_chars.extend(p2_str[min_len:])
            
        child_str = "".join(child_chars)
        offspring[k, 0] = registrar_cadena(child_str)
        
    return offspring

def calcular_salud_promedio(ga_instance):
    fitness_actual = ga_instance.last_generation_fitness
    if fitness_actual is not None and len(fitness_actual) > 0:
        salud_promedio = sum(fitness_actual) / len(fitness_actual)
        return float(-salud_promedio)
    return 0.0

# =============================================================================
# 7. CALLBACK ON_GENERATION Y GESTIÓN DE SNAPSHOTS
# =============================================================================
tiempo_inicio_ejecucion = time.time()

def on_generation(ga_instance):
    global strings_pool, secuencia_snapshot_actual

    gen_actual = ga_instance.generations_completed
    
    best_idx = np.argmax(ga_instance.last_generation_fitness)
    mejor_id_actual = int(ga_instance.population[best_idx, 0])
    
    cadena_real = strings_pool[mejor_id_actual]
    best_fitness_actual = ga_instance.last_generation_fitness[best_idx]
    real_chi2 = -best_fitness_actual

    poblacion_actual = ga_instance.population
    nuevas_cadenas_vivas = []
    nuevo_mapeo_ids = {}
    
    for individuo in poblacion_actual:
        id_viejo = int(individuo[0])
        cadena = strings_pool[id_viejo]
        
        if cadena not in nuevo_mapeo_ids:
            nuevo_mapeo_ids[cadena] = len(nuevas_cadenas_vivas)
            nuevas_cadenas_vivas.append(cadena)
            
        individuo[0] = nuevo_mapeo_ids[cadena]
        
    strings_pool = nuevas_cadenas_vivas
    
    mejor_costo_medio = real_chi2 / len(datos)
    salud_media = calcular_salud_promedio(ga_instance) / len(datos)

    if gen_actual % 100 == 0:
        print(f"[OK] Generación: {gen_actual} | Mejor: {mejor_costo_medio:.4f} | Promedio: {salud_media:.4f} | Cadena: {cadena_real}")
    sys.stdout.flush()

    if gen_actual % FRECUENCIA_SNAPSHOT == 0:
        secuencia_snapshot_actual += 1
        tiempo_actual_proceso = tiempo_acumulado + (time.time() - tiempo_inicio_ejecucion)
        
        nombre_snapshot = f"{NOMBRE_CORTO_DATASET}.{secuencia_snapshot_actual:06d}.out"
        ruta_completa = os.path.join(DIRECTORIO_OUTPUT, nombre_snapshot)
        
        with open(ruta_completa, 'w', encoding='utf-8') as f:
            f.write(f"generaciones_transcurridas={gen_actual}\n")
            f.write(f"tiempo_procesamiento_segundos={tiempo_actual_proceso:.4f}\n")
            f.write(f"mejor_fitness={mejor_costo_medio:.4f}\n") 
            f.write(f"promedio={salud_media:.4f}\n")
            
            for individuo in poblacion_actual:
                idx = int(individuo[0])
                f.write(f"{strings_pool[idx]}\n")
                
        print(f"\n💾 [SNAPSHOT GUARDADO] -> {ruta_completa} (Gen real: {gen_actual})")


# =============================================================================
# 8. CARGA DE POBLACIÓN (RESUME DESDE SNAPSHOT O NUEVA)
# =============================================================================
patron_busqueda = os.path.join(DIRECTORIO_OUTPUT, f"{NOMBRE_CORTO_DATASET}.*.out")
archivos_snapshots = glob.glob(patron_busqueda)
poblacion_inicial = []

if archivos_snapshots:
    archivos_snapshots.sort()
    ultimo_snapshot = archivos_snapshots[-1]
    
    nombre_archivo = os.path.basename(ultimo_snapshot)
    partes = nombre_archivo.split('.')
    if len(partes) >= 3:
        secuencia_snapshot_actual = int(partes[1])
        
    print(f"\n📂 ¡Snapshot encontrado! Reanudando desde: {ultimo_snapshot}")
    
    with open(ultimo_snapshot, 'r', encoding='utf-8') as f:
        lineas = f.readlines()
        
    generacion_inicio = int(lineas[0].strip().split('=')[1])
    tiempo_acumulado = float(lineas[1].strip().split('=')[1])
    
    for linea in lineas[4:]:
        cadena_str = linea.strip()
        if cadena_str:
            id_num = registrar_cadena(cadena_str)
            poblacion_inicial.append([id_num])
    
    poblacion = len(poblacion_inicial)
    print(f"   -> Tamaño final de la población activa configurado en: {poblacion}")
    print(f"   -> Generación de partida: {generacion_inicio}")
    print(f"   -> Secuencia de snapshot actual: {secuencia_snapshot_actual:06d}")
    print(f"   -> Tiempo acumulado previo: {tiempo_acumulado:.2f} segundos")
    print(f"   -> Individuos cargados: {len(poblacion_inicial)}")

else:
    print("3. Generando la población inicial a partir de todo el dataset...")
    if PROPORCION >= 100:
        for individuo_str in datos:
            id_numerico = registrar_cadena(individuo_str)
            poblacion_inicial.append([id_numerico])
    else:
        # Lógica de sembrado Top-K
        #TOP_K = 36  
        #TOP_K = min(TOP_K, poblacion_objetivo)
        TOP_K = poblacion_objetivo 

        print(f"   🔍 Evaluando en GPU las {len(datos)} cadenas del dataset para identificar el Top-{TOP_K}...")
        
        costos_dataset = evaluar_poblacion_gpu(datos)
        indices_ordenados = np.argsort(costos_dataset)
        
        top_k_indices = indices_ordenados[:TOP_K]
        cadenas_top_k = [datos[idx] for idx in top_k_indices]
        
        cadenas_sembradas = set()
        for cad in cadenas_top_k:
            id_num = registrar_cadena(cad)
            poblacion_inicial.append([id_num])
            cadenas_sembradas.add(cad)
            
        mejor_costo_inicial = costos_dataset[indices_ordenados[0]] / len(datos)
        print(f"   🌱 Top-{TOP_K} sembrado con éxito. (Mejor cadena inicial tiene costo medio: {mejor_costo_inicial:.4f})")
        
        cadenas_restantes = [cad for cad in datos if cad not in cadenas_sembradas]
        cupos_faltantes = poblacion_objetivo - len(poblacion_inicial)
        
        if cupos_faltantes > 0 and cadenas_restantes:
            seleccion_aleatoria = random.sample(cadenas_restantes, min(cupos_faltantes, len(cadenas_restantes)))
            for cad in seleccion_aleatoria:
                id_num = registrar_cadena(cad)
                poblacion_inicial.append([id_num])
                
        print(f"   ¡Población inicial completada con {len(poblacion_inicial)} individuos ({TOP_K} élite + {len(poblacion_inicial)-TOP_K} aleatorios)!")

    poblacion = len(poblacion_inicial)
    print(f"   ¡Población inicial configurada con {poblacion} individuos!")

# =============================================================================
# 9. EJECUCIÓN DEL ALGORITMO GENÉTICO
# =============================================================================
numero_padres = max(2, int(poblacion * PORCENTAJE_PADRES))

ga_instance = pygad.GA(
    num_generations=NUM_GENERATIONS,
    num_parents_mating=numero_padres,
    fitness_func=fitness_batch,           
    fitness_batch_size=poblacion,         
    initial_population=poblacion_inicial,
    crossover_type=cruzamiento_scattered_custom,
    on_generation=on_generation,  
    mutation_type=mutacion_porcentual_custom,
    mutation_by_replacement=False,
    parent_selection_type="tournament",   
    K_tournament=K_TOURNAMENT,
    keep_elitism=1,
    keep_parents=0,
)

if 'generacion_inicio' in locals() and generacion_inicio > 0:
    ga_instance.generations_completed = generacion_inicio
    print(f"🔄 Contador interno de PyGAD ajustado. Continuará a partir de la generación: {generacion_inicio}")
    
print("\n4. ¡Iniciando evolución impulsada por CUDA! (Pulsa Ctrl+C para detener)\n")
ga_instance.run()

# =============================================================================
# 10. RESULTADOS FINALES
# =============================================================================
best_solution, best_fitness, best_idx = ga_instance.best_solution()
mejor_generacion = ga_instance.best_solution_generation
real_chi2 = -best_fitness

cadena_final = strings_pool[int(best_solution[0])]

print("\n\n── RESULTADO FINAL ──")
print(f"Mejor solución (Cadena): {cadena_final}")
print(f"Mejor fitness (Costo Promedio): {real_chi2/len(datos):.6f}")
print(f"Encontrado en la generación:  {mejor_generacion}")
print(f"ID del string: {int(best_solution[0])}")
