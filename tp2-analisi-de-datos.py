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
print(df.describe(include="all", datetime_is_numeric=True))

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
df["ScheduledDay"] = pd.to_datetime(df["ScheduledDay"], errors="coerce")
df["AppointmentDay"] = pd.to_datetime(df["AppointmentDay"], errors="coerce")

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
    df["DidAttend"] = df["No-show"].apply(lambda x: 0 if str(x).strip().upper() == "YES" else 1)

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