# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
import pandas as pd
import json
import os

# === 1. Definir rutas ===
benchmarks_dir = "/home/dslab/teilor/atom/data/benchmarks/"
resultados_dir = "/home/dslab/teilor/atom/data/01_respuesta/"
output_path_dir = "/home/dslab/teilor/newtron/01_respuestas/"

# === 2. Cargar prompts con etiquetas y metadata ===
prompts_paths = {
    "halueval": os.path.join(benchmarks_dir, "halueval", "prompts_halueval.csv"),
    "fever": os.path.join(benchmarks_dir, "fever", "prompts_fever.csv"),
    "advglue": os.path.join(benchmarks_dir, "advglue", "prompts_advglue.csv")
}

df_prompts = []

for benchmark, path in prompts_paths.items():
    df = pd.read_csv(path)
    df["benchmark"] = benchmark

    # Estandarizar label_gold
    if benchmark == "halueval":
        df = df[df["label"].isin(["PASS", "FAIL"])]
        df["label_gold"] = df["label"].map({"PASS": "factual", "FAIL": "hallucinated"})

    elif benchmark == "fever":
        df = df[df["label"].isin(["SUPPORTS", "REFUTES"])]
        df["label_gold"] = df["label"]

    elif benchmark == "advglue":
        df = df[df["label"].isin([0, 2])]
        df["label_gold"] = df["label"].map({0: "entailment", 2: "contradiction"})

    # Crear etiqueta binaria invertida
    map_binaria = {
        "hallucinated": 1,
        "REFUTES": 1,
        "contradiction": 1,
        "factual": 0,
        "SUPPORTS": 0,
        "entailment": 0
    }
    df["label_gold_binaria"] = df["label_gold"].map(map_binaria)

    df_prompts.append(df)

df_prompts_all = pd.concat(df_prompts, ignore_index=True)

# === 3. Cargar resultados de modelos ===
resultados_files = [f for f in os.listdir(resultados_dir) if f.startswith("results_") and f.endswith(".json")]

for file in resultados_files:
    modelo = file.replace("results_", "").replace(".json", "")
    with open(os.path.join(resultados_dir, file), "r") as f:
        data = json.load(f)
        df_modelo = pd.DataFrame(data)
        df_modelo["modelo"] = modelo
        df_modelo = df_modelo[["prompt", "benchmark", "respuesta", "modelo"]]

        # === 4. Unir con prompts ===
        df_merged = pd.merge(df_prompts_all, df_modelo, on=["prompt", "benchmark"], how="inner")

        # === 5. Exportar como JSON ===
        output_model_path = os.path.join(output_path_dir, f"respuestas_gold_{modelo}.json")
        with open(output_model_path, "w", encoding="utf-8") as out_f:
            json.dump(df_merged.to_dict(orient="records"), out_f, ensure_ascii=False, indent=2)

        print(f"✓ Archivo generado: {output_model_path}")
