#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

# ======================
# 1️⃣ CARGA Y EXPLORACIÓN INICIAL
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
# 2️⃣ VALORES NULOS
# ======================

# Cargar dataset
CSV = "DatasetClase3_corrupto.csv"
df = pd.read_csv(CSV)

# --- Contar nulos por columna ---
print("\n=== NULOS POR COLUMNA ===")
print(df.isna().sum())

# --- Detectar y reemplazar tokens que representan nulos ---
TOKENS_NULOS = {"", " ", "-", "NA", "NaN", "N/A", "Desconocido", "desconocido"}
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

# 🔹 Imputación 1: con la mediana general
mediana = df["Age_num"].median()
df["Age_mediana"] = df["Age_num"].fillna(mediana)

# 🔹 Imputación 2: por grupo (ej. por género)
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
# 3️⃣ DUPLICADOS
# ======================

# Cargar dataset
CSV = "DatasetClase3_corrupto.csv"
df = pd.read_csv(CSV)

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