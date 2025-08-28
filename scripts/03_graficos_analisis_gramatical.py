# -*- coding: utf-8 -*-
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# === 1. Configuración ===
input_dir = "/home/dslab/teilor/newtron/02_analisis_gramatical/"
output_dir_modelo = "/home/dslab/teilor/newtron/03_graficos_gramaticales/por_modelo/"
output_dir_agregado = "/home/dslab/teilor/newtron/03_graficos_gramaticales/agregados/"
os.makedirs(output_dir_modelo, exist_ok=True)
os.makedirs(output_dir_agregado, exist_ok=True)

# === 2. Cargar todos los archivos por modelo ===
archivos = [f for f in os.listdir(input_dir) if f.startswith("analisis_gramatical_") and f.endswith(".csv")]
df_todos = []

for archivo in archivos:
    modelo = archivo.replace("analisis_gramatical_", "").replace(".csv", "")
    df = pd.read_csv(os.path.join(input_dir, archivo))
    df["modelo"] = modelo
    df_todos.append(df)

    # === Parte 1: gráficos individuales por modelo ===
    columnas_metrica = [col for col in df.columns if col.startswith("num_") or col.startswith("densidad_") or col.startswith("longitud_")]
    for metrica in columnas_metrica:
        plt.figure(figsize=(8, 5))
        sns.boxplot(x="label_gold_binaria", y=metrica, data=df)
        plt.title(f"{metrica} - {modelo}")
        plt.xlabel("label_gold_binaria (0=factual, 1=hallucinated)")
        plt.ylabel(metrica)
        plt.tight_layout()

        filename = f"{modelo}_{metrica}.png".replace("/", "_")
        plt.savefig(os.path.join(output_dir_modelo, filename))
        plt.close()

# === Parte 2: concatenar y preparar gráficos agregados ===
df_concat = pd.concat(df_todos, ignore_index=True)
columnas_metrica = [col for col in df_concat.columns if col.startswith("num_") or col.startswith("densidad_") or col.startswith("longitud_")]

# === Parte 3: Lectura de tests estadísticos con p_value real ===
estadisticas = {}  # clave: metrica, valor: set de modelos con p < 0.01
resumen_raw = []   # para construir resumen completo con p_value

for archivo in os.listdir(input_dir):
    if archivo.startswith("estadisticas_comparativas_") and archivo.endswith(".csv"):
        modelo = archivo.replace("estadisticas_comparativas_", "").replace(".csv", "")
        df_est = pd.read_csv(os.path.join(input_dir, archivo))

        for _, row in df_est.iterrows():
            metrica = row["metrica"]
            p = row["p_value"]
            if p < 0.01:
                estadisticas.setdefault(metrica, set()).add(modelo)
            resumen_raw.append({
                "metrica": metrica,
                "modelo": modelo,
                "p_value": p,
                "p_value_significativo": "< 0.01" if p < 0.01 else "= 0.01"
            })

# === Parte 4: gráficos agregados por métrica ===
for metrica in columnas_metrica:
    plt.figure(figsize=(12, 6))
    sns.boxplot(x="modelo", y=metrica, hue="label_gold_binaria", data=df_concat)

    modelos_significativos = estadisticas.get(metrica, set())
    titulo = f"{metrica} (significativo p<0.01 en: {', '.join(modelos_significativos)})" if modelos_significativos else metrica
    plt.title(titulo)
    
    plt.xlabel("Modelo")
    plt.ylabel(metrica)
    plt.legend(title="label_gold_binaria")
    plt.xticks(rotation=30)
    plt.tight_layout()

    filename = f"comparativa_{metrica}.png".replace("/", "_")
    plt.savefig(os.path.join(output_dir_agregado, filename))
    plt.close()

# === Parte 5: Guardar resumen de significancia global con p_value ===
df_resumen = pd.DataFrame(resumen_raw)
df_resumen = df_resumen.sort_values(by=["metrica", "p_value"])
ruta_resumen = os.path.join(output_dir_agregado, "resumen_significancia_mannwhitney.csv")
df_resumen.to_csv(ruta_resumen, index=False)
print(f"? Resumen guardado en: {ruta_resumen}")
