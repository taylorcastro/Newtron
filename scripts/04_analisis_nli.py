# -*- coding: utf-8 -*-

import os
import json
from tqdm import tqdm
from transformers import pipeline

# === 1. Rutas de entrada y salida ===
input_dir = "/home/dslab/teilor/newtron/01_respuestas/"
output_dir = "/home/dslab/teilor/newtron/04_analisis_nli/"
os.makedirs(output_dir, exist_ok=True)

# === 2. Cargar clasificadores NLI ===
print("Cargando modelos NLI...")
nli_roberta = pipeline("text-classification", model="roberta-large-mnli", tokenizer="roberta-large-mnli")
nli_deberta = pipeline("text-classification", model="microsoft/deberta-large-mnli", tokenizer="microsoft/deberta-large-mnli")
nli_distil = pipeline("text-classification", model="microsoft/deberta-base-mnli", tokenizer="microsoft/deberta-base-mnli")

# === 3. Funcion para obtener etiqueta NLI ===
def predecir_nli(pipeline_nli, premisa, hipotesis):
    try:
        entrada = f"{premisa} </s></s> {hipotesis}"
        salida = pipeline_nli(entrada, truncation=True)
        return salida[0]["label"].lower()
    except Exception as e:
        return "error"

# === 4. Archivos de entrada ===
archivos = [f for f in os.listdir(input_dir) if f.startswith("respuestas_gold_") and f.endswith(".json")]

if not archivos:
    print("No se encontraron archivos de entrada en", input_dir)
else:
    for archivo in archivos:
        modelo = archivo.replace("respuestas_gold_", "").replace(".json", "")
        ruta_in = os.path.join(input_dir, archivo)
        ruta_out = os.path.join(output_dir, f"respuestas_nli_{modelo}.json")

        print(f"Procesando modelo: {modelo}")

        with open(ruta_in, "r", encoding="utf-8") as f:
            datos = json.load(f)

        resultados = []

        for entrada in tqdm(datos, desc=f"Procesando {modelo}"):
            prompt = entrada.get("prompt", "")
            respuesta = entrada.get("respuesta", "")

            label_r = predecir_nli(nli_roberta, prompt, respuesta)
            label_d = predecir_nli(nli_deberta, prompt, respuesta)
            label_distil = predecir_nli(nli_distil, prompt, respuesta)

            etiquetas = [label_r, label_d, label_distil]
            contradicciones = etiquetas.count("contradiction")
            es_alucinacion = 1 if contradicciones >= 2 else 0
            nli_valido = 1 if "error" not in etiquetas else 0

            entrada.update({
                "label_roberta": label_r,
                "label_deberta": label_d,
                "label_distilroberta": label_distil,
                "es_alucinacion": es_alucinacion,
                "n_contradicciones": contradicciones,
                "nli_valido": nli_valido
            })

            resultados.append(entrada)

        with open(ruta_out, "w", encoding="utf-8") as f_out:
            json.dump(resultados, f_out, ensure_ascii=False, indent=2)

        print(f"Guardado: {ruta_out}")
