import pandas as pd
import numpy as np
import sys
import time
import random
import math

# --- 1. CONFIGURACIÓN ---
BASE_SUBMISSION_FILE = "submission_vH4.5_(0.69432993).csv" # submission anterior
NEW_SUBMISSION_FILE = "submission_vSA_()" # submission que creará el SA
DATA_PATH = "C:/Users/julio/Documents/IO files/"
ALPHA = 0.5
TIME_LIMIT_SECONDS = 3600 # 1 hora

# --- PARÁMETROS DE Simulated Annealing (SA) ---
# T_INITIAL: La "temperatura" inicial.
# Debe estar en una magnitud similar a los deltas de score que esperamos.
# Un delta negativo "malo" (ej: -1e-7) dividido por T_INITIAL (ej: 1e-7) da -1.
# La probabilidad de aceptarlo sería exp(-1) = 0.36 (36%). ¡Perfecto para empezar!
T_INITIAL = 1e-7

print(f"--- Iniciando Heurística (Simulated Annealing) ---")
print(f"Semilla: {BASE_SUBMISSION_FILE}")
print(f"Límite de tiempo: {TIME_LIMIT_SECONDS} segundos")
print(f"Temperatura Inicial (T_INITIAL): {T_INITIAL}")

# --- 2. FASE DE SETUP: Cargar Todos los Datos ---
try:
    df_submission = pd.read_csv(DATA_PATH + BASE_SUBMISSION_FILE)
    df_students = pd.read_csv(DATA_PATH + "students.csv")
    df_universities = pd.read_csv(DATA_PATH + "universities.csv")
    df_merit = pd.read_csv(DATA_PATH + "merit_list.csv")
except FileNotFoundError as e:
    print(f"Error fatal: No se encontraron los archivos de datos o la semilla. {e}")
    sys.exit(1)

print("Paso 1: Todos los archivos cargados.")

# --- 3. FASE DE SETUP: Pre-cálculo de Lookups (Del Scorer) ---
N = len(df_students)
U = len(df_universities)

# 3a. Lookup de Preferencias (FE_i)
pref_cols = [f'pref_{i}' for i in range(1, 51)]
df_prefs_long = df_students.melt(
    id_vars=['student_id'], value_vars=pref_cols,
    var_name='preference_rank_str', value_name='university_id'
)
df_prefs_long['p'] = df_prefs_long['preference_rank_str'].str.extract(r'(\d+)').astype(int)
student_pref_rank_map = df_prefs_long.set_index(['student_id', 'university_id'])['p'].to_dict()

# 3b. Lookup de Mérito (FU_i)
df_merit['FU_i'] = (N + 1 - df_merit['merit_rank']) / N
student_merit_map = df_merit.set_index('student_id')['FU_i'].to_dict()

# 3c. Lookup de Capacidades (cap_u)
university_capacities_map = df_universities.set_index('university_id')['cap'].to_dict()

# --- 4. FASE DE SETUP: Calcular Estado Inicial ---
print("Paso 2: Calculando Score Inicial (Estado Base)...")

# Diccionario de asignaciones ACTUALES (el estado que "pasea"): student_id -> university_id
current_assignments = df_submission.set_index('student_id')['university_id'].to_dict()
student_ids = list(current_assignments.keys())

# Diccionario de suma de méritos por universidad: university_id -> sum(FU_i)
df_submission['FU_i'] = df_submission['student_id'].map(student_merit_map)
uni_merit_sum = df_submission.groupby('university_id')['FU_i'].sum().to_dict()

# Función helper para calcular FE_i (Felicidad Estudiante)
def get_fe_i(student_id, university_id):
    p = student_pref_rank_map.get((student_id, university_id))
    return (51 - p) / 50 if p else -2

# Calcular FPE Inicial
total_fe = sum(get_fe_i(s_id, u_id) for s_id, u_id in current_assignments.items())
FPE_inicial = total_fe / N

# Calcular FPU Inicial
total_fpu_component = 0
for u_id, cap in university_capacities_map.items():
    sum_fu_i_for_this_uni = uni_merit_sum.get(u_id, 0)
    total_fpu_component += (1 / cap) * sum_fu_i_for_this_uni
FPU_inicial = (1 / U) * total_fpu_component

# ¡SCORE INICIAL!
# 'current_score' es el score del estado "paseante"
current_score = (ALPHA * FPE_inicial) + ((1 - ALPHA) * FPU_inicial)

# 'best_score' es el MEJOR score que hemos visto HASTA AHORA.
best_score = current_score
# 'best_assignments' guarda la MEJOR solución encontrada.
best_assignments = current_assignments.copy()

print(f"Score Inicial (Heur_anterior) verificado: {best_score:.8f}")

# --- 5. FASE DE BÚSQUEDA (Simulated Annealing) ---
print(f"\n--- Iniciando Fase de Búsqueda ({TIME_LIMIT_SECONDS} seg) ---")
start_time = time.time()
last_print_time = start_time
iteration = 0
improvements_found = 0
worse_moves_accepted = 0

try:
    while (time.time() - start_time) < TIME_LIMIT_SECONDS:
        iteration += 1
        
        # 1. Proponer un "Swap" (Vecino)
        s1, s2 = random.sample(student_ids, 2)
        u1 = current_assignments[s1]
        u2 = current_assignments[s2]
        
        if u1 == u2:
            continue
            
        # 2. Calcular el "Delta (Δ)"
        old_fe_s1 = get_fe_i(s1, u1)
        old_fe_s2 = get_fe_i(s2, u2)
        new_fe_s1 = get_fe_i(s1, u2)
        new_fe_s2 = get_fe_i(s2, u1)
        delta_fpe = (1/N) * ( (new_fe_s1 + new_fe_s2) - (old_fe_s1 + old_fe_s2) )
        
        fu_s1 = student_merit_map[s1]
        fu_s2 = student_merit_map[s2]
        delta_fpu_u1 = (1 / university_capacities_map[u1]) * (fu_s2 - fu_s1)
        delta_fpu_u2 = (1 / university_capacities_map[u2]) * (fu_s1 - fu_s2)
        delta_fpu = (1/U) * (delta_fpu_u1 + delta_fpu_u2)
        
        delta_score = (ALPHA * delta_fpe) + ((1 - ALPHA) * delta_fpu)
        
        # 3. Decidir (LÓGICA DE Simulated Annealing)
        
        if delta_score > 0:
            # 3a. ACEPTACIÓN POSITIVA (Mejora)
            # Si el swap es bueno, siempre lo aceptamos.
            improvements_found += 1
            
            # Aceptar el swap en nuestra asignación "actual"
            current_assignments[s1] = u2
            current_assignments[s2] = u1
            current_score += delta_score
            
            # Actualizar las sumas de mérito de las unis afectadas
            uni_merit_sum[u1] = uni_merit_sum[u1] - fu_s1 + fu_s2
            uni_merit_sum[u2] = uni_merit_sum[u2] - fu_s2 + fu_s1
            
            # Chequear si este nuevo estado es el mejor que hemos visto.
            if current_score > best_score:
                best_score = current_score
                best_assignments = current_assignments.copy() # Guardar una copia
                print(f"¡NUEVO MÁXIMO {improvements_found}! Iter: {iteration} | Nuevo Best Score: {best_score:.8f} (+{delta_score:.8f})")

        else:
            # 3b. ACEPTACIÓN NEGATIVA (Movimiento "Malo")
            # El score empeora (delta_score <= 0).
            # Lo aceptamos con una probabilidad P = exp(delta / T)
            
            # Calcular "Temperatura" actual (Enfriamiento Lineal)
            # T va de T_INITIAL a 0, en proporción al tiempo transcurrido.
            elapsed_ratio = (time.time() - start_time) / TIME_LIMIT_SECONDS
            current_temp = T_INITIAL * (1.0 - elapsed_ratio)

            # Evitar T=0 (división por cero) si el tiempo justo se acaba
            if current_temp <= 0:
                current_temp = 1e-10 # Un valor muy pequeño

            # Calcular la probabilidad de aceptación (Criterio de Metropolis)
            try:
                acceptance_prob = math.exp(delta_score / current_temp)
            except OverflowError:
                acceptance_prob = 0 # Si el número es demasiado pequeño
            
            if random.random() < acceptance_prob:
                # ¡Movimiento malo ACEPTADO para explorar!
                worse_moves_accepted += 1
                
                # Aceptar el swap
                current_assignments[s1] = u2
                current_assignments[s2] = u1
                current_score += delta_score # El score actual baja
                
                # Actualizar las sumas de mérito
                uni_merit_sum[u1] = uni_merit_sum[u1] - fu_s1 + fu_s2
                uni_merit_sum[u2] = uni_merit_sum[u2] - fu_s2 + fu_s1
        
        # Imprimir un estado cada 10 segundos
        if (time.time() - last_print_time) > 10:
            print(f"Iter: {iteration} | Mejor Score: {best_score:.8f} | Score Actual: {current_score:.8f} | T: {current_temp:.2e} | Peores Acept: {worse_moves_accepted}")
            last_print_time = time.time()

except KeyboardInterrupt:
    print("\nBúsqueda interrumpida por el usuario.")

print("\n--- Búsqueda Finalizada ---")
print(f"Tiempo total: {time.time() - start_time:.2f} segundos.")
print(f"Total de iteraciones: {iteration}")
print(f"Total de mejoras directas: {improvements_found}")
print(f"Total de movimientos peores aceptados: {worse_moves_accepted}")
print(f"Score Final (Mejor encontrado): {best_score:.8f}")

# --- 6. FASE DE FINALIZACIÓN: Guardar ---
print(f"Guardando la MEJOR solución encontrada en '{NEW_SUBMISSION_FILE}'...")

# ¡IMPORTANTE! Guardamos 'best_assignments', NO 'current_assignments'
df_final_submission = pd.DataFrame(list(best_assignments.items()), columns=['student_id', 'university_id'])
df_final_submission = df_final_submission.sort_values(by='student_id')
df_final_submission.to_csv(NEW_SUBMISSION_FILE, index=False)

print("¡Guardado completado!")