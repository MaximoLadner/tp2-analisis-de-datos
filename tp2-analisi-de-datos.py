#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

# ======================
# CARGA Y EXPLORACIÓN INICIAL
# ======================

# Cargar el dataset
CSV = "DatasetClase3_corrupto.csv"
df = pd.read_csv(CSV)


# Mostrar dimensiones del DataFrame
print("\n=== SHAPE ===")
print(df.shape)

# Mostrar información general del dataset
print("\n=== INFO ===")
print(df.info())

# Mostrar estadísticos descriptivos (incluyendo texto y fechas)
print("\n=== DESCRIBE (GENERAL) ===")
print(df.describe(include="all"))

# Mostrar tipos de datos de todas las columnas
print("\n=== TIPOS DE DATOS (DTYPES) ===")
print(df.dtypes)



# ======================
# VALORES NULOS
# ======================

# --- Contar nulos por columna ---
print("\n=== NULOS POR COLUMNA ===")
print(df.isna().sum())

# --- Detectar y reemplazar tokens que representan nulos ---
TOKENS_NULOS = {"", " ", "   ", "-", "na", "NA", "NaN", "nan", "N/A", "Desconocido", "desconocido"} 
for c in df.columns:
    if df[c].dtype == object:
        df[c] = df[c].replace(list(TOKENS_NULOS), pd.NA)

print("\n=== NULOS TRAS REEMPLAZO DE TOKENS ===")
print(df.isna().sum())

# --- Imputaciones en columna numérica (Age) ---
# Convertimos Age a numérico
df["Age_num"] = pd.to_numeric(df["Age"], errors="coerce")

# Guardar estadísticos antes de imputar
print("\n=== ESTADÍSTICOS ANTES DE IMPUTAR (Age) ===")
print(df["Age_num"].describe()[["mean", "50%", "std"]])

# Imputación 1: con la mediana general
mediana = df["Age_num"].median()
df["Age_mediana"] = df["Age_num"].fillna(mediana)

# Imputación 2: por grupo (ej. por género)
# Convertimos Gender a texto limpio antes de agrupar
df["Gender"] = df["Gender"].astype("string").str.strip().str.lower()
df["Gender"] = df["Gender"].replace({
    "f": "F", "female": "F", "fem": "F",
    "m": "M", "male": "M", "masc": "M", "masculino": "M",
    "otro": "Otro", "other": "Otro"
})

df["Age_grupo"] = df.groupby("Gender")["Age_num"].transform(
    lambda x: x.fillna(x.median())
)

# --- Estadísticos después de imputaciones ---
print("\n=== ESTADÍSTICOS DESPUÉS DE IMPUTAR (mediana general) ===")
print(df["Age_mediana"].describe()[["mean", "50%", "std"]])

print("\n=== ESTADÍSTICOS DESPUÉS DE IMPUTAR (por grupo) ===")
print(df["Age_grupo"].describe()[["mean", "50%", "std"]])


# ======================
# DUPLICADOS
# ======================

# --- Detectar duplicados exactos ---
duplicados_exactos = df.duplicated().sum()
print(f"\n=== DUPLICADOS EXACTOS ===\nCantidad: {duplicados_exactos}")

# Si hay duplicados exactos, mostramos algunos ejemplos
if duplicados_exactos > 0:
    print("\nEjemplos de duplicados exactos:")
    print(df[df.duplicated()].head())

# --- Detectar duplicados por columnas de tipo ID ---
# Se suelen usar columnas como PatientId o AppointmentID
dup_por_patient = df.duplicated(subset=["PatientId"]).sum()
dup_por_appointment = df.duplicated(subset=["AppointmentID"]).sum()

print(f"\nDuplicados por PatientId: {dup_por_patient}")
print(f"Duplicados por AppointmentID: {dup_por_appointment}")

# --- Eliminar duplicados exactos ---
df_sin_dup = df.drop_duplicates()

print(f"\nDuplicados exactos después de eliminar: {df_sin_dup.duplicated().sum()}")

print("""
DECISIÓN:
Se eliminaron los duplicados exactos para evitar registros repetidos
que podrían distorsionar los análisis estadísticos. 
Los duplicados por ID no se eliminaron todavía, ya que podrían 
representar citas distintas del mismo paciente.
""")

# ======================
# FECHAS
# ======================


# --- Convertir columnas de fecha ---
# Convertimos las columnas 'ScheduledDay' y 'AppointmentDay' al tipo datetime
df["ScheduledDay"] = pd.to_datetime(df["ScheduledDay"], errors="coerce", utc=True)
df["AppointmentDay"] = pd.to_datetime(df["AppointmentDay"], errors="coerce", utc=True)

# --- Crear columna DiffDays ---
# Calcula la diferencia en días entre la cita y el agendamiento
df["DiffDays"] = (df["AppointmentDay"] - df["ScheduledDay"]).dt.days

# --- Detectar casos con valores negativos ---
negativos = df[df["DiffDays"] < 0]

print("\n=== DIFERENCIAS DE FECHAS ===")
print(f"Cantidad de valores negativos: {len(negativos)}")

if len(negativos) > 0:
    print("\nEjemplos de registros con DiffDays negativo:")
    print(negativos[["ScheduledDay", "AppointmentDay", "DiffDays"]].head())

# --- Revisión general ---
print("\nResumen de DiffDays:")
print(df["DiffDays"].describe())

# --- Documentar la decisión ---
print("""
DECISIÓN:
Se crearon las columnas de tipo fecha correctamente y la diferencia en días (DiffDays).
Los valores negativos indican errores en la carga de datos (fechas de cita anteriores a la programación).
Estos registros se marcarán para revisión o eliminación en pasos posteriores.
""")            


# ======================
# CATEGÓRICAS
# ======================


# --- Identificar columnas con baja cardinalidad ---
low_cardinality = [col for col in df.columns if df[col].nunique() <= 10]
print("\n=== COLUMNAS CON BAJA CARDINALIDAD ===")
print(low_cardinality)

# --- Convertir esas columnas a tipo category ---
for col in low_cardinality:
    df[col] = df[col].astype("category")

print("\nTipos de datos luego de conversión:")
print(df.dtypes[df.dtypes == "category"])

# --- Unificar valores mal escritos en columnas categóricas ---
if "Gender" in df.columns:
    df["Gender"] = df["Gender"].replace({
        "FEM": "F",
        "Fem": "F",
        "female": "F",
        "MASC": "M",
        "Masc": "M",
        "male": "M"
    })

# --- Crear variable booleana DidAttend ---
# Según la consigna, 1 = asistió, 0 = no asistió
if "No-show" in df.columns:
    # --- Limpiar y estandarizar la columna No-show ---
    df["No-show_limpio"] = df["No-show"].astype(str).str.strip().str.upper()

    # --- Definir el mapeo para corregir inconsistencias ---
    
    # Mapeo a 'YES' (No Asistió / No Show)
    no_asistio_map = {'YES', 'Y', '1', 'TRUE', 'SI'} 
    
    # Mapeo a 'NO' (Sí Asistió / Show)
    si_asistio_map = {'NO', 'N', '0', 'FALSE'}
    
    # Función de corrección para asignar solo 'YES', 'NO', o pd.NA
    def corregir_no_show(valor):
        if pd.isna(valor) or valor in {'NAN', ''}: 
            return pd.NA
        elif valor in no_asistio_map:
            return 'YES' # Se unifica a 'YES' (No Asistió)
        elif valor in si_asistio_map:
            return 'NO'  # Se unifica a 'NO' (Sí Asistió)
        else: # Si es cualquier otra categoría inválida no mapeada
            return pd.NA

    df["No-show_limpio"] = df["No-show_limpio"].apply(corregir_no_show)

    # --- Crear variable booleana DidAttend: 1 = asistió ('NO'), 0 = no asistió ('YES') ---
    mapping_didattend = {'NO': 1, 'YES': 0} 
    df["DidAttend"] = df["No-show_limpio"].map(mapping_didattend).fillna(pd.NA)
    
    # --- Usar el tipo Int64 para permitir valores enteros con nulos ---
    df["DidAttend"] = df["DidAttend"].astype('Int64')

print("\n=== COLUMNA DidAttend (LUEGO DE LIMPIEZA) ===")
print(df["DidAttend"].value_counts(dropna=False))

print(f"\nValores únicos en No-show_limpio (CORREGIDOS): {df['No-show_limpio'].unique()}")


print("\n=== COLUMNA DidAttend ===")
print(df["DidAttend"].value_counts())



# --- Mostrar valores únicos actualizados ---
if "Gender" in df.columns:
    print("\nValores únicos en Gender luego de limpieza:")
    print(df["Gender"].unique())

# --- Guardar dataset actualizado (opcional) ---
# df.to_csv("DatasetClase3_limpio.csv", index=False)

print("""
DECISIÓN:
✔ Se detectaron columnas categóricas con baja cardinalidad.
✔ Se convirtieron al tipo category.
✔ Se normalizaron valores inconsistentes (por ej. 'Fem' → 'F').
✔ Se creó la variable DidAttend, que representa si el paciente asistió (1) o no (0).
""")

# ======================
# VERIFICACIÓN DE DOMINIOS
# ======================


# --- Detección de edades fuera del rango [0, 120] ---
EDAD_MIN = 0
EDAD_MAX = 120
edades_invalidas = df[(df["Age_num"] < EDAD_MIN) | (df["Age_num"] > EDAD_MAX)]


# --- Listas de edades inválidas ---
print(f"Registros con edad < {EDAD_MIN}: {len(df[df['Age_num'] < EDAD_MIN])}")
print(f"Registros con edad > {EDAD_MAX}: {len(df[df['Age_num'] > EDAD_MAX])}")


# --- Valores únicos que son inválidos ---
print("\nValores únicos fuera de rango (muestras):")
print(edades_invalidas["Age_num"].value_counts())


# --- Corregir la columna Age_num: Convertir los valores fuera de rango a nulo ---
df["Age_num"] = np.where(
    (df["Age_num"] < EDAD_MIN) | (df["Age_num"] > EDAD_MAX), 
    pd.NA, 
    df["Age_num"]
)

print(f"\nNuevos nulos en Age_num después de corregir rangos: {df['Age_num'].isna().sum()}")


# --- Columna Gender ---
print("\n=== VALIDACIÓN DE GÉNERO ===")
valores_gender = df["Gender"].unique()
print(f"Valores únicos en Gender: {valores_gender}")


# --- Detectar y contar valores inesperados que no sean F, M, Otro o nulos ---
categorias_validas_gender = {'F', 'M', 'Otro', pd.NA}
invalidos_gender = df["Gender"][~df["Gender"].isin(categorias_validas_gender) & df["Gender"].notna()].unique()

print(f"Categorías inválidas encontradas: {invalidos_gender}")


# --- Columna No-show ---
print("\n=== VALIDACIÓN DE NO-SHOW ===")

# --- Limpiar y estandarizar la columna No-show ---
df["No-show_limpio"] = df["No-show"].astype(str).str.strip().str.upper()

# --- Mostrar valores únicos esperados ---
valores_noshow = df["No-show_limpio"].unique()
print(f"Valores únicos en No-show (limpio): {valores_noshow}")

print(f"\nValores únicos en No-show_limpio (CORREGIDOS): {df['No-show_limpio'].unique()}")

print("""
DECISIÓN:
✔ Se detectaron edades fuera del rango establecido.
✔ Se convirtieron al tipo null para tratarlas posteriormente.
✔ Los valores de Género ya fueron corregidos y unificados en 'F', 'M', y 'Otro'.
✔ Ya se han tratado los valores válidos en la limpieza de DidAttend.
""")



# ======================
# OUTLIERS
# ======================

def detectar_outliers_iqr(df, columna):
    """Detecta y cuenta outliers en una columna usando la regla 1.5 * IQR."""
    # Quitar nulos para el cálculo
    serie_sin_na = df[columna].dropna()
    
    Q1 = serie_sin_na.quantile(0.25)
    Q3 = serie_sin_na.quantile(0.75)
    IQR = Q3 - Q1
    
    limite_inferior = Q1 - 1.5 * IQR
    limite_superior = Q3 + 1.5 * IQR
    
    outliers = serie_sin_na[(serie_sin_na < limite_inferior) | (serie_sin_na > limite_superior)]
    
    print(f"\n--- Análisis de Outliers en {columna} ---")
    print(f"Q1: {Q1:.2f}, Q3: {Q3:.2f}, IQR: {IQR:.2f}")
    print(f"Límite Inferior (1.5*IQR): {limite_inferior:.2f}")
    print(f"Límite Superior (1.5*IQR): {limite_superior:.2f}")
    print(f"Cantidad de outliers detectados: {len(outliers)}")
    print(f"Porcentaje del dataset: {len(outliers)/len(df) * 100:.2f}%")
    
    return outliers, limite_inferior, limite_superior


# --- Detección en Age_num ---
outliers_age, lower_age, upper_age = detectar_outliers_iqr(df, "Age_num")


# --- Detección en DiffDays ---
### Los valores negativos ya son anómalos. El IQR los detecta como outliers.
outliers_diffdays, lower_diff, upper_diff = detectar_outliers_iqr(df, "DiffDays")


# --- Librerías para armar los Boxplots ---
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('ggplot')


fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# --- Boxplot para Age_num ---
sns.boxplot(x=df["Age_num"], ax=axes[0])
axes[0].set_title("Boxplot de Age_num (Edad)")
axes[0].axvline(lower_age, color='r', linestyle='--', linewidth=0.8)
axes[0].axvline(upper_age, color='r', linestyle='--', linewidth=0.8)

# --- Boxplot para DiffDays ---
sns.boxplot(x=df["DiffDays"], ax=axes[1])
axes[1].set_title("Boxplot de DiffDays (Días de espera)")
axes[1].axvline(lower_diff, color='r', linestyle='--', linewidth=0.8)
axes[1].axvline(upper_diff, color='r', linestyle='--', linewidth=0.8)

plt.tight_layout()
plt.show()

print("""
DECISIÓN:
✔ La Winsorización sería una buena alternativa para acotar las edades extremas a un valor máximo razonable para reducir su impacto sin eliminar datos.
✔ Los valores negativos son  deben ser eliminados o convertidos a nulo ya que no son un comportamiento real.
✔ Los valores positivos extremos se conservan inicialmente ya que representan un comportamiento real del sistema de salud que puede ser un factor importante.
""")


# ======================
# PRIMERAS AGREGACIONES
# ======================

print("\n=== AGREGACIONES DESCRIPTIVAS ===")

# --- Edad promedio y mediana por género ---
### Usamos df.groupby().agg() para calcular ambas métricas a la vez
edad_por_genero = df.groupby("Gender")["Age_num"].agg(
    [('Edad Promedio', 'mean'), ('Edad Mediana', 'median')]
)
print("\n--- Edad Promedio y Mediana por Género ---")
print(edad_por_genero.round(2)) # El método .round(2) se usa para limitar los decimales a dos.


# --- Tiempo de espera promedio según asistencia ---
tiempo_espera_por_asistencia = df.groupby("DidAttend")["DiffDays"].mean()
print("\n--- Tiempo de Espera Promedio (en días) según Asistencia ---")
print("1: Asistió | 0: No Asistió")
print(tiempo_espera_por_asistencia.round(2))


# --- Edad promedio y máxima. Tiempo de espera promedio ---
### Agrupados por si asistieron o no 
agg_multiple = df.groupby("DidAttend").agg(
    # Aplicar aggregación a Age_num
    Edad_Promedio=('Age_num', 'mean'),
    Edad_Maxima=('Age_num', 'max'),
    # Aplicar aggregación a DiffDays
    Espera_Mediana=('DiffDays', 'median'),
    Conteo=('AppointmentID', 'count')
)
print("\n--- Agregaciones Múltiples por Asistencia (DidAttend) ---")
print("1: Asistió | 0: No Asistió")
print(agg_multiple.round(2))