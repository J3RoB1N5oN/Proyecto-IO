import pandas as pd
import numpy as np
import sys
import time
import random

# --- 1. CONFIGURACIÓN ---
# ¡ASEGÚRATE DE USAR NUESTRA MEJOR SOLUCIÓN COMO SEMILLA!
BASE_SUBMISSION_FILE = "submission_vSA_(0.69432993).csv" # submission anterior que se intenta mejorar
NEW_SUBMISSION_FILE = "submission_vHC_().csv" #submission que creará
DATA_PATH = "C:/Users/julio/Documents/IO files/"
ALPHA = 0.5
TIME_LIMIT_SECONDS = 3600 # modificar este número para delimitar cuando tiempo correrá el código

print(f"--- Iniciando Nueva Heurística (Búsqueda Local por Swaps) ---")
print(f"Semilla: {BASE_SUBMISSION_FILE}")
print(f"Límite de tiempo: {TIME_LIMIT_SECONDS} segundos")

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

# Diccionario de asignaciones actuales: student_id -> university_id
current_assignments = df_submission.set_index('student_id')['university_id'].to_dict()
student_ids = list(current_assignments.keys()) # Lista de todos los student_ids

# Diccionario de suma de méritos por universidad: university_id -> sum(FU_i)
# Usamos el mapa de mérito en las asignaciones actuales
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
current_best_score = (ALPHA * FPE_inicial) + ((1 - ALPHA) * FPU_inicial)
print(f"Score Inicial (Heur_anterior) verificado: {current_best_score:.8f}")

# --- 5. FASE DE BÚSQUEDA ---
print(f"\n--- Iniciando Fase de Búsqueda ({TIME_LIMIT_SECONDS} seg) ---")
start_time = time.time()
last_print_time = start_time
iteration = 0
improvements_found = 0

try:
    while (time.time() - start_time) < TIME_LIMIT_SECONDS:
        iteration += 1
        
        # 1. Proponer un "Swap"
        s1, s2 = random.sample(student_ids, 2)
        u1 = current_assignments[s1]
        u2 = current_assignments[s2]
        
        # Si están en la misma uni, el swap es inútil
        if u1 == u2:
            continue
            
        # 2. Calcular el "Delta (Δ)"
        
        # --- ΔFPE ---
        # FE_i antes del swap
        old_fe_s1 = get_fe_i(s1, u1)
        old_fe_s2 = get_fe_i(s2, u2)
        # FE_i después del swap
        new_fe_s1 = get_fe_i(s1, u2) # s1 ahora va a u2
        new_fe_s2 = get_fe_i(s2, u1) # s2 ahora va a u1
        
        # (1/N) * ( (Nuevos) - (Viejos) )
        delta_fpe = (1/N) * ( (new_fe_s1 + new_fe_s2) - (old_fe_s1 + old_fe_s2) )
        
        # --- ΔFPU ---
        # Mérito de cada estudiante (constante)
        fu_s1 = student_merit_map[s1]
        fu_s2 = student_merit_map[s2]
        
        # (1/U) * ( (Δ para u1) + (Δ para u2) )
        delta_fpu_u1 = (1 / university_capacities_map[u1]) * (fu_s2 - fu_s1) # u1 gana s2, pierde s1
        delta_fpu_u2 = (1 / university_capacities_map[u2]) * (fu_s1 - fu_s2) # u2 gana s1, pierde s2
        
        delta_fpu = (1/U) * (delta_fpu_u1 + delta_fpu_u2)
        
        # --- ΔScore Total ---
        delta_score = (ALPHA * delta_fpe) + ((1 - ALPHA) * delta_fpu)
        
        # 3. Decidir
        if delta_score > 0:
            # ¡MEJORA ENCONTRADA!
            improvements_found += 1
            
            # 3a. Aceptar el swap en nuestra asignación
            current_assignments[s1] = u2
            current_assignments[s2] = u1
            
            # 3b. Actualizar el estado para el próximo cálculo
            current_best_score += delta_score
            
            # 3c. Actualizar las sumas de mérito de las unis afectadas
            uni_merit_sum[u1] = uni_merit_sum[u1] - fu_s1 + fu_s2
            uni_merit_sum[u2] = uni_merit_sum[u2] - fu_s2 + fu_s1
            
            # Imprime por cada iteración si obtiene una mejora
            print(f"¡MEJORA {improvements_found}! Iter: {iteration} | Nuevo Score: {current_best_score:.8f} (+{delta_score:.8f})")
        
        # Imprimir un estado cada 10 segundos
        if (time.time() - last_print_time) > 10:
            print(f"Iter: {iteration} | Mejor Score: {current_best_score:.8f} | Tiempo: {time.time() - start_time:.0f}s")
            last_print_time = time.time()

except KeyboardInterrupt:
    print("\nBúsqueda interrumpida por el usuario.")

print("\n--- Búsqueda Finalizada ---")
print(f"Tiempo total: {time.time() - start_time:.2f} segundos.")
print(f"Total de iteraciones: {iteration}")
print(f"Total de mejoras encontradas: {improvements_found}")
print(f"Score Inicial (Submission_Anterior): {FPE_inicial + FPU_inicial:.8f}") # (Re-calculo para asegurar)
print(f"Score Final (Submission_Mejorada): {current_best_score:.8f}")

# --- 6. FASE DE FINALIZACIÓN: Guardar ---
print(f"Guardando la solución mejorada en '{NEW_SUBMISSION_FILE}'...")

df_final_submission = pd.DataFrame(list(current_assignments.items()), columns=['student_id', 'university_id'])
df_final_submission = df_final_submission.sort_values(by='student_id')
df_final_submission.to_csv(NEW_SUBMISSION_FILE, index=False)

print("¡Guardado completado!")

print("Puedes correr SCORER.py de este nuevo archivo para una verificación final.")
