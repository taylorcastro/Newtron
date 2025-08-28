# -*- coding: utf-8 -*-
import os
import json
import pandas as pd
import spacy
from tqdm import tqdm
from scipy.stats import mannwhitneyu

# === 1. Configuración ===
input_dir = "/home/dslab/teilor/newtron/01_respuestas/"
output_dir = "/home/dslab/teilor/newtron/02_analisis_gramatical/"
os.makedirs(output_dir, exist_ok=True)

nlp = spacy.load("en_core_web_sm")  # usa modelo en inglés

# === 2. Archivos a procesar ===
archivos = [f for f in os.listdir(input_dir) if f.startswith("respuestas_gold_") and f.endswith(".json")]

# === 3. Función de análisis ===
def analizar_respuesta(respuesta):
    doc = nlp(respuesta)
    return {
        "num_tokens": len(doc),
        "num_palabras": len([t for t in doc if t.is_alpha]),
        "num_oraciones": len(list(doc.sents)),
        "num_verbs": len([t for t in doc if t.pos_ == "VERB"]),
        "num_nouns": len([t for t in doc if t.pos_ == "NOUN"]),
        "num_adjs": len([t for t in doc if t.pos_ == "ADJ"]),
        "num_advs": len([t for t in doc if t.pos_ == "ADV"]),
        "num_entidades": len(doc.ents),
        "longitud_promedio_oracion": sum(len(sent) for sent in doc.sents) / max(len(list(doc.sents)), 1),
        "densidad_verbos": len([t for t in doc if t.pos_ == "VERB"]) / max(len(doc), 1),
        "densidad_sustantivos": len([t for t in doc if t.pos_ == "NOUN"]) / max(len(doc), 1),
        "densidad_entidades": len(doc.ents) / max(len(doc), 1)
    }

# === 4. Procesar por archivo/modelo ===
for archivo in archivos:
    modelo = archivo.replace("respuestas_gold_", "").replace(".json", "")
    ruta_in = os.path.join(input_dir, archivo)
    ruta_out_csv = os.path.join(output_dir, f"analisis_gramatical_{modelo}.csv")
    ruta_out_test = os.path.join(output_dir, f"estadisticas_comparativas_{modelo}.csv")

    with open(ruta_in, "r") as f:
        datos = json.load(f)

    registros = []
    for entrada in tqdm(datos, desc=f"Procesando {modelo}"):
        analisis = analizar_respuesta(entrada["respuesta"])
        analisis.update({
            "modelo": modelo,
            "benchmark": entrada["benchmark"],
            "label_gold_binaria": entrada["label_gold_binaria"],
            "prompt": entrada["prompt"]
        })
        registros.append(analisis)

    df = pd.DataFrame(registros)
    df.to_csv(ruta_out_csv, index=False)
    print(f"Guardado: {ruta_out_csv}")

    # === 5. Comparación entre clases (0 vs 1) ===
    comparaciones = []
    columnas_metrica = [col for col in df.columns if col.startswith("num_") or col.startswith("densidad_") or col.startswith("longitud_")]
    clase0 = df[df["label_gold_binaria"] == 0]
    clase1 = df[df["label_gold_binaria"] == 1]

    for col in columnas_metrica:
        try:
            stat, p_value = mannwhitneyu(clase0[col], clase1[col], alternative='two-sided')
            comparaciones.append({
                "metrica": col,
                "media_clase_0": clase0[col].mean(),
                "media_clase_1": clase1[col].mean(),
                "p_value": p_value
            })
        except:
            pass  # en caso de columnas con NaNs

    df_comp = pd.DataFrame(comparaciones)
    df_comp.to_csv(ruta_out_test, index=False)
    print(f"Estadisticas comparativas: {ruta_out_test}")
