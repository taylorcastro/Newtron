#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
05_metricas_modelos.py
Fusiona GOLD+NLI y calcula métricas (MCC, Balanced Accuracy, CI bootstrap, Kappa, Mann–Whitney).

Notas Newtron:
  - Unidad = PROMPT (1 respuesta por prompt). NO se deduplica por prompt (solo se evita duplicación de PK).
  - es_alucinacion = consenso ESTRICTO de NLI:
      * 3 válidos  -> ≥2 "contradiction" (2/3)
      * 2 válidos  -> 2/2
      * 0–1 válidos -> NaN

Salida: CSVs en --output_dir usados por 06_visualizaciones.py
"""

import os, re, json, glob, argparse, math, hashlib
from typing import List, Dict, Any, Tuple
import numpy as np
import pandas as pd

# ==== RUTAS POR DEFECTO ====
DEFAULT_INPUT_DIR_NLI  = "/home/dslab/teilor/newtron/04_analisis_nli/"
DEFAULT_INPUT_DIR_GOLD = "/home/dslab/teilor/newtron/01_respuestas/"
DEFAULT_OUTPUT_DIR     = "/home/dslab/teilor/newtron/05_metricas/"
DEFAULT_GRAM_DIR       = "/home/dslab/teilor/newtron/02_analisis_gramatical/"
# ===========================

LABEL_MAP = {
    "llama3-latest": "llama3:8b",
    "llama3-8b": "llama3:8b",
    "mistral-7b": "mistral0.2:7b",
    "mistral-0.2-7b": "mistral0.2:7b",
    "phi-latest": "phi3:3.8b",
    "phi3-3.8b": "phi3:3.8b",
    "gemma-7b": "gemma3:4b",
    "gemma-2-9b": "gemma3:4b",
    "qwen-latest": "qwen2.5:7b",
    "qwen2.5-7b": "qwen2.5:7b",
    "deepseek-coder-6.7b": "deepseek:6.7b",
    "deepseek-coder:6.7b": "deepseek:6.7b",
}

# Kappa opcional (solo si está sklearn)
try:
    from sklearn.metrics import cohen_kappa_score
except Exception:
    cohen_kappa_score = None

# Mann–Whitney opcional (solo si está scipy)
try:
    from scipy.stats import mannwhitneyu
except Exception:
    mannwhitneyu = None

# -----------------------
# Utilidades IO
# -----------------------
def ensure_dir(p: str): os.makedirs(p, exist_ok=True)

def descubrir(input_dir: str, patrones: List[str]) -> List[str]:
    files = []
    for pat in patrones:
        files += glob.glob(os.path.join(input_dir, "**", pat), recursive=True)
    return sorted(set(files))

def leer_json_list(path: str) -> List[Dict[str, Any]]:
    """Lector robusto: JSON, JSON con {'data': [...]}, y JSONL línea a línea."""
    try:
        txt = open(path, "r", encoding="utf-8-sig").read()
    except Exception:
        return []
    if not txt.strip():
        return []
    # 1) JSON estándar
    try:
        obj = json.loads(txt)
        if isinstance(obj, list):
            return obj
        if isinstance(obj, dict):
            if "data" in obj and isinstance(obj["data"], list):
                return obj["data"]
            return [obj]
    except Exception:
        pass
    # 2) JSONL (un objeto por línea)
    out = []
    for ln in txt.splitlines():
        ln = ln.strip()
        if not ln:
            continue
        try:
            out.append(json.loads(ln))
        except Exception:
            continue
    return out

def extraer_modelo(path: str) -> str:
    base = os.path.basename(path).lower()
    m = re.match(r"respuestas_(?:nli|gold)_(.+)\.jsonl?$", base)
    name = m.group(1) if m else os.path.splitext(base)[0]
    return LABEL_MAP.get(name, name)

# -----------------------
# Normalización de claves
# -----------------------
def _norm_keys(df: pd.DataFrame) -> pd.DataFrame:
    """Normaliza claves; evita FutureWarning forzando dtype 'object' antes de asignar."""
    if df is None or df.empty:
        return df
    out = df.copy()

    # Asegura dtype object para columnas clave
    for c in ("modelo", "idx", "id", "prompt", "benchmark"):
        if c in out.columns:
            out[c] = out[c].astype("object")

    # modelo/idx/id -> str limpio
    for c in ("modelo", "idx", "id"):
        if c in out.columns:
            s = out[c]
            mask = s.notna()
            out.loc[mask, c] = s.loc[mask].astype(str).str.strip()

    # prompt/benchmark: normaliza solo valores no nulos; deja NaN como NaN
    if "prompt" in out.columns:
        s = out["prompt"]
        mask = s.notna()
        out.loc[mask, "prompt"] = (
            s.loc[mask].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
        )
    if "benchmark" in out.columns:
        s = out["benchmark"]
        mask = s.notna()
        out.loc[mask, "benchmark"] = (
            s.loc[mask].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
        )
    return out

# -----------------------
# Limpieza / normalización de NLI
# -----------------------
def normalizar_labels_nli(df: pd.DataFrame) -> pd.DataFrame:
    valid = {"contradiction","entailment","neutral"}
    syn = {
        "contradiccion":"contradiction","contra":"contradiction",
        "entail":"entailment","entailmento":"entailment",
        "neutralidad":"neutral","c":"contradiction","e":"entailment","n":"neutral"
    }
    for col in ["label_roberta","label_deberta","label_distilroberta"]:
        if col in df.columns:
            s = df[col].astype(str).str.strip().str.lower()
            s = s.replace({"nan":np.nan,"<na>":np.nan,"none":np.nan,"":np.nan})
            s = s.replace(syn)
            df[col] = s.where(s.isin(valid), np.nan)
        else:
            df[col] = np.nan
    return df

def recomputar_mayoria(df: pd.DataFrame) -> pd.DataFrame:
    """Consenso ESTRICTO: 3→2/3; 2→2/2; <=1→NaN."""
    cols = ["label_roberta","label_deberta","label_distilroberta"]
    df["n_nli_validos"] = df[cols].notna().sum(axis=1)
    df["n_contradictions_calc"] = (df[cols] == "contradiction").sum(axis=1)
    thr = df["n_nli_validos"].map({3: 2, 2: 2}).astype(float)
    df["es_alucinacion_calc"] = np.where(
        df["n_nli_validos"].isin([2,3]),
        (df["n_contradictions_calc"] >= thr).astype(int),
        np.nan
    )
    if "es_alucinacion" in df.columns:
        mask = df["es_alucinacion"].notna() & df["es_alucinacion_calc"].notna()
        df["consistencia_mayoria"] = np.nan
        df.loc[mask,"consistencia_mayoria"] = (
            df.loc[mask,"es_alucinacion"].astype(int) == df.loc[mask,"es_alucinacion_calc"].astype(int)
        ).astype(int)
        df["es_alucinacion"] = df["es_alucinacion"].where(df["es_alucinacion"].notna(), df["es_alucinacion_calc"])
    else:
        df["consistencia_mayoria"] = np.nan
        df["es_alucinacion"] = df["es_alucinacion_calc"]
    return df

def homogenizar_meta(df: pd.DataFrame) -> pd.DataFrame:
    for col in ["length_class","has_numbers","is_question","question_type","benchmark","prompt","id","idx"]:
        if col not in df.columns:
            df[col] = np.nan
    for col in ["has_numbers","is_question"]:
        if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
            s = df[col].astype(str).str.lower().str.strip().replace(
                {"true":1,"false":0,"si":1,"sí":1,"no":0,"nan":np.nan,"none":np.nan,"":np.nan}
            )
            df[col] = pd.to_numeric(s, errors="coerce")
    return df

# -----------------------
# Métricas base
# -----------------------
def exactitud(y_true, y_pred):
    return float((y_true == y_pred).mean()) if len(y_true) else np.nan

def prf1_counts(y_true, y_pred, positive=1):
    tp = int(((y_pred==positive) & (y_true==positive)).sum())
    fp = int(((y_pred==positive) & (y_true!=positive)).sum())
    fn = int(((y_pred!=positive) & (y_true==positive)).sum())
    tn = int(((y_pred!=positive) & (y_true!=positive)).sum())
    precision = tp/(tp+fp) if (tp+fp)>0 else np.nan
    recall    = tp/(tp+fn) if (tp+fn)>0 else np.nan
    f1 = 2*precision*recall/(precision+recall) if (precision and recall and (precision+recall)>0) else np.nan
    return precision, recall, f1, tp, fp, fn, tn

def balanced_accuracy_from_counts(tp, fp, fn, tn):
    tpr = tp/(tp+fn) if (tp+fn)>0 else np.nan
    tnr = tn/(tn+fp) if (tn+fp)>0 else np.nan
    if np.isnan(tpr) or np.isnan(tnr): return np.nan
    return (tpr + tnr) / 2.0

def mcc_from_counts(tp, fp, fn, tn):
    num = tp*tn - fp*fn
    den = (tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)
    if den == 0:
        return np.nan
    try:
        return float(num) / math.sqrt(float(den))
    except Exception:
        return np.nan

def kappa_por_modelo(df_model: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str,float]]:
    if cohen_kappa_score is None or df_model.empty:
        return pd.DataFrame(columns=["modelo","comparacion","kappa"]), {}
    name = df_model["modelo"].iloc[0]
    rows = []
    def pair(ca, cb, na, nb):
        mask = df_model[ca].notna() & df_model[cb].notna()
        if mask.sum() < 2: return None
        a = (df_model.loc[mask, ca] == "contradiction").astype(int)
        b = (df_model.loc[mask, cb] == "contradiction").astype(int)
        if a.nunique()<2 or b.nunique()<2: return None
        return {"modelo":name,"comparacion":f"{na} vs {nb}","kappa":round(float(cohen_kappa_score(a,b)),6)}
    for (ca,cb,na,nb) in [
        ("label_roberta","label_distilroberta","RoBERTa","DistilRoBERTa"),
        ("label_roberta","label_deberta","RoBERTa","DeBERTa"),
        ("label_distilroberta","label_deberta","DistilRoBERTa","DeBERTa"),
    ]:
        r = pair(ca,cb,na,nb)
        if r: rows.append(r)
    rango = {}
    if rows:
        ks = [r["kappa"] for r in rows]
        rango = {"modelo":name,"max_kappa":float(np.max(ks)),"min_kappa":float(np.min(ks))}
    return pd.DataFrame(rows), rango

# -----------------------
# Bootstrap (IC 95%)
# -----------------------
def _bootstrap_indices(n, rng, size=None):
    size = n if size is None else size
    return rng.choice(np.arange(n), size=size, replace=True)

def bootstrap_mean_ci(values, n_boot=1000, ci=95, seed=42):
    vals = pd.Series(values).dropna().astype(float).to_numpy()
    if len(vals) == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = _bootstrap_indices(len(vals), rng)
        boots[i] = np.mean(vals[idx])
    lower = np.percentile(boots, (100-ci)/2)
    upper = np.percentile(boots, 100-(100-ci)/2)
    return float(np.mean(vals)), float(lower), float(upper)

def _metric_from_labels(y_true, y_pred):
    prec, rec, f1, tp, fp, fn, tn = prf1_counts(y_true, y_pred, 1)
    acc = exactitud(y_true, y_pred)
    bal = balanced_accuracy_from_counts(tp, fp, fn, tn)
    mcc = mcc_from_counts(tp, fp, fn, tn)
    return {
        "accuracy": acc, "precision": prec, "recall": rec, "f1": f1,
        "balanced_accuracy": bal, "mcc": mcc,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn
    }

def bootstrap_metrics_ci(y_true, y_pred, n_boot=1000, ci=95, seed=42):
    """Bootstrap percentil de accuracy, precision, recall, f1, balanced_accuracy, mcc."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    n = len(y_true)
    if n == 0:
        empty = {k:(np.nan, np.nan, np.nan) for k in
                 ["accuracy","precision","recall","f1","balanced_accuracy","mcc"]}
        return empty
    rng = np.random.default_rng(seed)
    acc_b, prec_b, rec_b, f1_b, bal_b, mcc_b = ([] for _ in range(6))
    idx_all = np.arange(n)

    def once(idx):
        met = _metric_from_labels(y_true[idx], y_pred[idx])
        return (met["accuracy"], met["precision"], met["recall"], met["f1"],
                met["balanced_accuracy"], met["mcc"])

    for _ in range(n_boot):
        idx = rng.choice(idx_all, size=n, replace=True)
        a, p, r, f, b, m = once(idx)
        acc_b.append(a); prec_b.append(p); rec_b.append(r)
        f1_b.append(f); bal_b.append(b);  mcc_b.append(m)

    def _ci(arr):
        arr = np.array(arr, dtype=float)
        if np.isnan(arr).all():
            return (np.nan, np.nan, np.nan)
        low = (100 - ci) / 2
        high = 100 - low
        return (
            float(np.nanmean(arr)),
            float(np.nanpercentile(arr, low)),
            float(np.nanpercentile(arr, high)),
        )

    return {
        "accuracy": _ci(acc_b),
        "precision": _ci(prec_b),
        "recall": _ci(rec_b),
        "f1": _ci(f1_b),
        "balanced_accuracy": _ci(bal_b),
        "mcc": _ci(mcc_b),
    }

# -----------------------
# Mann–Whitney + FDR (opcional)
# -----------------------
GRAM_COL_MAP = {
    "num_tokens": "longitud_tokens",
    "num_palabras": "num_palabras",
    "num_oraciones": "num_oraciones",
    "num_verbs": "num_verbos",
    "num_nouns": "num_sustantivos",
    "num_adjs": "num_adjetivos",
    "num_advs": "num_adverbios",
    "num_entidades": "num_entidades",
    "longitud_promedio_oracion": "longitud_promedio_oracion",
    "densidad_verbos": "densidad_verbos",
    "densidad_sustantivos": "densidad_sustantivos",
    "densidad_entidades": "densidad_entidades",
    "modelo": "modelo",
    "benchmark": "benchmark",
    "label_gold_binaria": "label_gold_binaria",
    "prompt": "prompt",
}
VAR_LING_DEFAULT = [
    "longitud_tokens","num_verbos","num_sustantivos","num_adjetivos","num_adverbios",
    "num_entidades","longitud_promedio_oracion","densidad_verbos","densidad_sustantivos","densidad_entidades",
]

def cargar_gram_csvs(gram_dir: str) -> pd.DataFrame:
    import glob as _glob
    if not gram_dir or not os.path.isdir(gram_dir):
        return pd.DataFrame()
    csv_files = sorted(_glob.glob(os.path.join(gram_dir, "*.csv")))
    if not csv_files:
        return pd.DataFrame()
    frames = []
    for f in csv_files:
        try:
            df = pd.read_csv(f)
        except Exception:
            continue
        df.columns = [str(c).strip() for c in df.columns]
        synonyms = {
            "model": "modelo", "modelo_nombre": "modelo",
            "dataset": "benchmark", "benchmark_name": "benchmark",
            "prompt_text": "prompt", "prompt_original": "prompt",
            "pregunta": "prompt", "texto_prompt": "prompt",
            "gold_binaria": "label_gold_binaria", "labelgold_binaria": "label_gold_binaria",
            "prompt_id": "id", "id_prompt": "id", "Id": "id", "ID": "id",
        }
        for src, dst in synonyms.items():
            if src in df.columns and dst not in df.columns:
                df.rename(columns={src: dst}, inplace=True)
        cols_presentes = {c: GRAM_COL_MAP[c] for c in df.columns if c in GRAM_COL_MAP}
        if cols_presentes:
            df.rename(columns=cols_presentes, inplace=True)
        for c in ["modelo", "benchmark", "prompt", "label_gold_binaria", "id"]:
            if c not in df.columns:
                df[c] = np.nan
        key_cols = {"modelo","benchmark","prompt","label_gold_binaria","id","__file__"}
        numeric_cols = [c for c in df.columns if c not in key_cols and pd.api.types.is_numeric_dtype(df[c])]
        keep = ["modelo","benchmark","prompt","label_gold_binaria","id"] + numeric_cols
        keep = [c for c in keep if c in df.columns]
        if not keep: continue
        df = df[keep]
        df["__file__"] = os.path.basename(f)
        if "modelo" in df.columns:
            mapped = df["modelo"].map(LABEL_MAP)
            df["modelo"] = mapped.where(mapped.notna(), df["modelo"])
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)

def unir_gram_con_json(df_gram: pd.DataFrame, df_json_all: pd.DataFrame) -> pd.DataFrame:
    if df_gram.empty or df_json_all.empty:
        return pd.DataFrame()
    for c in ["modelo","benchmark","prompt","id","idx","es_alucinacion"]:
        if c not in df_json_all.columns:
            df_json_all[c] = np.nan
    for c in ["modelo","benchmark","prompt","id","idx"]:
        if c not in df_gram.columns:
            df_gram[c] = np.nan
    df_gram = _norm_keys(df_gram); df_json_all = _norm_keys(df_json_all)
    candidates = [
        (["modelo","idx"], ["modelo","idx"]),
        (["modelo","benchmark","prompt"], ["modelo","benchmark","prompt"]),
        (["modelo","prompt"], ["modelo","prompt"]),
        (["modelo","id"], ["modelo","id"]),
        (["id"], ["id"]),
    ]
    right = df_json_all[["modelo","benchmark","prompt","id","idx","es_alucinacion"]].drop_duplicates()
    best, best_null = None, 1.1
    for lk, rk in candidates:
        if any(k not in df_gram.columns for k in lk) or any(k not in right.columns for k in rk):
            continue
        merged = df_gram.merge(right, how="left", left_on=lk, right_on=rk, suffixes=("", "_json"))
        null_rate = merged["es_alucinacion"].isna().mean() if "es_alucinacion" in merged.columns else 1.0
        if null_rate < best_null:
            best_null = null_rate
            best = merged
        if best_null < 0.2:
            break
    return best if best is not None else df_gram

def _bh_fdr(pvals: np.ndarray) -> np.ndarray:
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    order = np.argsort(p)
    ranked = p[order]
    adj = np.empty(m, dtype=float)
    cummin = 1.0
    for i in range(m-1, -1, -1):
        rank = i + 1
        val = ranked[i] * m / rank
        if val < cummin:
            cummin = val
        adj[i] = min(cummin, 1.0)
    out = np.empty(m, dtype=float)
    out[order] = adj
    return out

def mann_whitney_desde_gram_unido(df_gram_joined: pd.DataFrame, variables: List[str] = None) -> pd.DataFrame:
    if mannwhitneyu is None or df_gram_joined.empty:
        return pd.DataFrame(columns=[
            "modelo","benchmark","variable",
            "grupo_a_media","grupo_b_media",
            "U_statistic","p_value","p_adj_bh","significativo","significativo_fdr"
        ])
    variables = variables or VAR_LING_DEFAULT
    required_cols = {"modelo","benchmark","es_alucinacion"}
    if not required_cols.issubset(set(df_gram_joined.columns)):
        return pd.DataFrame(columns=[
            "modelo","benchmark","variable",
            "grupo_a_media","grupo_b_media",
            "U_statistic","p_value","p_adj_bh","significativo","significativo_fdr"
        ])
    results = []
    df = df_gram_joined.copy()
    df = df[df["es_alucinacion"].isin([0,1])]
    for (modelo, bmk), sub in df.groupby(["modelo","benchmark"]):
        for var in variables:
            if var not in sub.columns: continue
            a = sub[sub["es_alucinacion"]==1][var].dropna()
            b = sub[sub["es_alucinacion"]==0][var].dropna()
            if len(a)==0 or len(b)==0: continue
            try:
                stat, p = mannwhitneyu(a, b, alternative="two-sided")
            except Exception:
                continue
            results.append({
                "modelo": modelo, "benchmark": bmk, "variable": var,
                "grupo_a_media": float(a.mean()), "grupo_b_media": float(b.mean()),
                "U_statistic": float(stat), "p_value": float(np.round(p, 6))
            })
    out = pd.DataFrame(results)
    if out.empty:
        return pd.DataFrame(columns=[
            "modelo","benchmark","variable",
            "grupo_a_media","grupo_b_media","U_statistic",
            "p_value","p_adj_bh","significativo","significativo_fdr"
        ])
    out["p_adj_bh"] = _bh_fdr(out["p_value"].to_numpy())
    out["significativo"] = out["p_value"] < 0.05
    out["significativo_fdr"] = out["p_adj_bh"] < 0.05
    return out[[
        "modelo","benchmark","variable",
        "grupo_a_media","grupo_b_media","U_statistic","p_value","p_adj_bh",
        "significativo","significativo_fdr"
    ]]

# -----------------------
# CLAVE PRIMARIA y MERGE 1–A–1
# -----------------------
def build_primary_key(df: pd.DataFrame) -> pd.Series:
    """Prioriza id -> idx -> hash(modelo||benchmark||prompt).
    Rellena NaN con placeholders únicos por fila SIN desalinear longitudes."""
    df = _norm_keys(df)

    # 1) por id (si existe y ~completo)
    if "id" in df.columns and df["id"].notna().mean() > 0.9:
        pk = df["modelo"].astype(str).str.strip() + "||" + df["id"].astype(str).str.strip()
        print(f"[PK] filas por estrategia -> id:{len(df)} (100.0%), idx:0 (0.0%), texto:0 (0.0%)")
        return pk

    # 2) por idx (si hay cobertura razonable)
    if "idx" in df.columns and df["idx"].notna().mean() > 0.1:
        pk = df["modelo"].astype(str).str.strip() + "||" + df["idx"].astype(str).str.strip()
        print(f"[PK] filas por estrategia -> id:0 (0.0%), idx:{len(df)} (100.0%), texto:0 (0.0%)")
        return pk

    # 3) por hash de (benchmark, prompt) con placeholders únicos para NaN
    bm = df["benchmark"] if "benchmark" in df.columns else pd.Series(index=df.index, dtype="object")
    pr = df["prompt"]    if "prompt"    in df.columns else pd.Series(index=df.index, dtype="object")

    bm = bm.astype("object")
    pr = pr.astype("object")

    bmask = bm.notna()
    pmask = pr.notna()

    idx_str = pd.Series(df.index.astype(str), index=df.index)

    bm.loc[~bmask] = ("__NA_BENCH__" + idx_str.loc[~bmask]).to_numpy()
    pr.loc[~pmask] = ("__NA_PROMPT__" + idx_str.loc[~pmask]).to_numpy()

    base = bm.astype(str) + "||" + pr.astype(str)
    h = base.apply(lambda x: hashlib.sha1(x.encode("utf-8")).hexdigest())
    pk = df["modelo"].astype(str).str.strip() + "||" + h

    print(f"[PK] filas por estrategia -> id:0 (0.0%), idx:0 (0.0%), texto:{len(df)} (100.0%)")
    return pk

def unir_nli_gold(df_nli: pd.DataFrame, df_gold: pd.DataFrame) -> pd.DataFrame:
    """Merge 1–a–1 garantizado mediante clave primaria __pk__."""
    if df_nli.empty and df_gold.empty:
        return pd.DataFrame()
    if df_gold.empty:
        return df_nli.copy()
    if df_nli.empty:
        return df_gold.copy()

    df_nli  = _norm_keys(df_nli)
    df_gold = _norm_keys(df_gold)

    df_nli["__pk__"]  = build_primary_key(df_nli)
    df_gold["__pk__"] = build_primary_key(df_gold)

    # Diagnóstico de duplicados
    for tag, df_ in [("NLI", df_nli), ("GOLD", df_gold)]:
        dups = df_.duplicated(subset="__pk__", keep=False)
        n_dup = int(dups.sum())
        if n_dup:
            print(f"[WARN] {tag} con {n_dup} claves duplicadas; se deduplican por __pk__ (keep=first).")
            top = (df_.loc[dups].groupby(["modelo","__pk__"]).size()
                   .sort_values(ascending=False).head(3))
            if len(top):
                print(f"[WARN] {tag} top PK duplicadas:\n{top}\n")
            df_ = df_.drop_duplicates(subset="__pk__", keep="first")
        if tag == "NLI": df_nli = df_
        else:            df_gold = df_

    left_n = len(df_nli)
    merged = df_nli.merge(
        df_gold[["__pk__","label_gold_binaria"]],
        how="left", on="__pk__", validate="one_to_one", suffixes=("", "_gold")
    )
    if len(merged) != left_n:
        print(f"[ERROR] Merge cambió el tamaño: NLI={left_n} -> merged={len(merged)}")

    return merged

def colapsar_duplicados_1d(df: pd.DataFrame, nombre: str) -> pd.DataFrame:
    idxs = [i for i, c in enumerate(df.columns) if c == nombre]
    if len(idxs) <= 1:
        if nombre in df.columns and isinstance(df[nombre], pd.DataFrame):
            sub = df[nombre]
            df[nombre] = sub.bfill(axis=1).iloc[:, 0]
        return df
    sub = df.iloc[:, idxs]
    serie = sub.bfill(axis=1).iloc[:, 0]
    df[nombre] = serie
    drop_by_pos = idxs[1:]
    df.drop(df.columns[drop_by_pos], axis=1, inplace=True)
    return df

def canonizar_cols_gold(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    for base in ["label_gold", "label_gold_binaria"]:
        if base in df.columns:
            df = colapsar_duplicados_1d(df, base)
    if "label_gold_gold" in df.columns:
        if "label_gold" not in df.columns:
            df.rename(columns={"label_gold_gold": "label_gold"}, inplace=True)
        else:
            df["label_gold"] = df["label_gold"].where(df["label_gold"].notna(), df["label_gold_gold"])
    if "label_gold_binaria_gold" in df.columns:
        if "label_gold_binaria" not in df.columns:
            df.rename(columns={"label_gold_binaria_gold": "label_gold_binaria"}, inplace=True)
        else:
            df["label_gold_binaria"] = df["label_gold_binaria"].where(
                pd.notna(df["label_gold_binaria"]), df["label_gold_binaria_gold"]
            )
    if "label_gold_binaria" in df.columns:
        df["label_gold_binaria"] = pd.to_numeric(df["label_gold_binaria"], errors="coerce")
    cols_drop = [c for c in df.columns if c.endswith("_gold") and c not in ("benchmark_gold",)]
    if cols_drop:
        df.drop(columns=cols_drop, inplace=True, errors="ignore")
    for base in ["label_gold", "label_gold_binaria"]:
        if base in df.columns and isinstance(df[base], pd.DataFrame):
            df = colapsar_duplicados_1d(df, base)
    return df

def tasa_por(df: pd.DataFrame, col: str) -> pd.DataFrame:
    if col not in df.columns:
        return pd.DataFrame(columns=[col,"rate","n"])
    g = (df.groupby(col, dropna=False)["es_alucinacion"]
           .agg(rate="mean", n="count")
           .reset_index())
    g["rate"] = g["rate"].astype(float)
    g["n"] = g["n"].astype(int)
    return g.sort_values(["rate","n"], ascending=[False,False])

def cargar_df_nli(paths: List[str]) -> pd.DataFrame:
    frames = []
    for p in paths:
        rows = leer_json_list(p)
        if not rows: continue
        df = pd.DataFrame(rows)
        df.drop(columns=["label_gold_binaria","label_gold"], errors="ignore", inplace=True)
        df = normalizar_labels_nli(df)
        df = recomputar_mayoria(df)
        df = homogenizar_meta(df)
        df["modelo"] = extraer_modelo(p)
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

def cargar_df_gold(paths: List[str]) -> pd.DataFrame:
    frames = []
    for p in paths:
        rows = leer_json_list(p)
        if not rows: continue
        df = pd.DataFrame(rows)
        syn = {"gold_binaria":"label_gold_binaria","labelgold_binaria":"label_gold_binaria"}
        for src, dst in syn.items():
            if src in df.columns and dst not in df.columns:
                df.rename(columns={src: dst}, inplace=True)
        if "label_gold_binaria" not in df.columns and "label_gold" in df.columns:
            s = df["label_gold"].astype(str).str.lower().str.strip().replace(
                {"true":"1","false":"0","si":"1","sí":"1","no":"0",
                 "alucinacion":"1","no_alucinacion":"0",
                 "contradiction":"1","entailment":"0","neutral":np.nan,
                 "nan":np.nan,"<na>":np.nan,"none":np.nan,"":np.nan}
            )
            df["label_gold_binaria"] = pd.to_numeric(s, errors="coerce")
        df = homogenizar_meta(df)
        df["modelo"] = extraer_modelo(p)
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

# -----------------------
# Main
# -----------------------
def main(input_dir_nli: str, input_dir_gold: str, output_dir: str, gram_dir: str = None):
    input_dir_nli  = os.path.abspath(input_dir_nli)
    input_dir_gold = os.path.abspath(input_dir_gold)
    output_dir     = os.path.abspath(output_dir)
    ensure_dir(output_dir)

    nli_files  = descubrir(input_dir_nli,  ["respuestas_nli_*.json","respuestas_nli_*.jsonl"])
    gold_files = descubrir(input_dir_gold, ["respuestas_gold_*.json","respuestas_gold_*.jsonl"])

    print(f"[NLI] {len(nli_files)} archivos en {input_dir_nli}")
    print(f"[GOLD] {len(gold_files)} archivos en {input_dir_gold}")

    df_nli  = _norm_keys(cargar_df_nli(nli_files))
    df_gold = _norm_keys(cargar_df_gold(gold_files))

    # --- CHECK inicial (antes de procesar) ---
    try:
        nli_counts  = df_nli.groupby("modelo").size().to_dict()
        gold_counts = df_gold.groupby("modelo").size().to_dict()
        print("[CHECK] Filas leídas por modelo (antes de procesar):")
        for m in sorted(nli_counts.keys()):
            print(f"  - {m}: prompts_en_archivos(NLI)={nli_counts[m]} | respuestas_en_archivos(GOLD)={gold_counts.get(m,0)}")
    except Exception:
        pass

    # --- DEBUG tamaños leídos antes de merges ---
    try:
        print("[DEBUG] NLI filas totales:", len(df_nli),
              "| por modelo:", df_nli.groupby("modelo").size().to_dict())
    except Exception:
        pass
    try:
        print("[DEBUG] GOLD filas totales:", len(df_gold),
              "| por modelo:", df_gold.groupby("modelo").size().to_dict())
    except Exception:
        pass

    if df_nli.empty:
        print("[ERROR] No hay NLI; no se pueden calcular tasas por consenso.")
        return

    # --- MERGE 1–A–1 ---
    df_all = unir_nli_gold(df_nli, df_gold)
    df_all = canonizar_cols_gold(df_all)

    # --- por modelo ---
    overview_rows, dist_global = [], []
    seg_len_all, seg_num_all, seg_isq_all, seg_qtype_all, seg_bench_all, seg_gold_all = [], [], [], [], [], []
    kappa_all_rows_list, kappa_range_rows, ranking_rows = [], [], []
    metricas_gold_rows, pct_clf_rows = [], []

    for modelo, dfm in df_all.groupby("modelo", dropna=False):
        dfm = dfm.copy()

        # 1 fila = 1 prompt (no colapsamos)
        n_prompts    = int(len(dfm))
        n_respuestas = n_prompts

        # disponibilidad NLI
        n_valid_prom = int(dfm["es_alucinacion"].notna().sum())
        pct_nli_prom = (n_valid_prom / n_prompts) if n_prompts else np.nan

        # tasa e IC
        mean_rate, lo_rate, hi_rate = bootstrap_mean_ci(dfm["es_alucinacion"])

        # consistencia de mayoría (si venía en datos originales)
        consist = float(dfm.get("consistencia_mayoria", pd.Series(dtype=float)).mean()) \
                  if "consistencia_mayoria" in dfm.columns else np.nan

        overview_rows.append({
            "modelo": modelo,
            "N_prompts": n_prompts,
            "N_respuestas": n_respuestas,
            "n_prompts_con_NLI": n_valid_prom,
            "n_respuestas_con_NLI": n_valid_prom,
            "pct_con_NLI_prompts": pct_nli_prom,
            "pct_con_NLI_respuestas": pct_nli_prom,
            # compat antigua
            "n": n_prompts,
            "n_con_NLI": n_valid_prom,
            "pct_con_NLI": pct_nli_prom,
            # tasas
            "hallucination_rate": mean_rate,
            "hallucination_rate_ci_lower": lo_rate,
            "hallucination_rate_ci_upper": hi_rate,
            "consistencia_mayoria": consist
        })

        # LOG por modelo (prompts/respuestas)
        try:
            pct_txt = f"{pct_nli_prom:.1%}" if not np.isnan(pct_nli_prom) else "NaN"
            rate_txt = f"{mean_rate:.1%}" if not np.isnan(mean_rate) else "NaN"
            print(f"[STATS] {modelo}: N_prompts={n_prompts}, N_respuestas={n_respuestas}, "
                  f"con_NLI={n_valid_prom} ({pct_txt}), halluc_rate={rate_txt}")
        except Exception:
            pass

        # distribución NLI (por clasificador)
        for (name,col) in [("roberta","label_roberta"),("deberta","label_deberta"),("distilroberta","label_distilroberta")]:
            vc = dfm[col].value_counts(dropna=False)
            for lab,cnt in vc.items():
                dist_global.append({"modelo":modelo,"clasificador_nli":name,"label":str(lab),"count":int(cnt)})

        # métricas vs gold + CI
        if "label_gold_binaria" in dfm.columns:
            sub = dfm[dfm["label_gold_binaria"].notna() & dfm["es_alucinacion"].notna()]
            if not sub.empty:
                y_true = sub["label_gold_binaria"].astype(int).to_numpy()
                y_pred = sub["es_alucinacion"].astype(int).to_numpy()
                met = _metric_from_labels(y_true, y_pred)
                boots = bootstrap_metrics_ci(y_true, y_pred)
                metricas_gold_rows.append({
                    "modelo":modelo,"n_gold":int(len(sub)),
                    "accuracy_gold":met["accuracy"],
                    "accuracy_ci_lower":boots["accuracy"][1],
                    "accuracy_ci_upper":boots["accuracy"][2],
                    "precision_gold":met["precision"],
                    "precision_ci_lower":boots["precision"][1],
                    "precision_ci_upper":boots["precision"][2],
                    "recall_gold":met["recall"],
                    "recall_ci_lower":boots["recall"][1],
                    "recall_ci_upper":boots["recall"][2],
                    "f1_gold":met["f1"],
                    "f1_ci_lower":boots["f1"][1],
                    "f1_ci_upper":boots["f1"][2],
                    "balanced_accuracy_gold":met["balanced_accuracy"],
                    "balanced_accuracy_ci_lower":boots["balanced_accuracy"][1],
                    "balanced_accuracy_ci_upper":boots["balanced_accuracy"][2],
                    "mcc_gold":met["mcc"],
                    "mcc_ci_lower":boots["mcc"][1],
                    "mcc_ci_upper":boots["mcc"][2],
                    "tp":met["tp"],"fp":met["fp"],"fn":met["fn"],"tn":met["tn"]
                })

        # segmentaciones
        def _seg(df, col):
            if col not in df.columns:
                return None
            g = df.groupby(col, dropna=False)["es_alucinacion"].agg(["mean","count"]).reset_index()
            g.rename(columns={"mean":"rate","count":"n"}, inplace=True)
            g.insert(0,"modelo",modelo)
            return g

        for col, bucket in [
            ("length_class",seg_len_all),("has_numbers",seg_num_all),("is_question",seg_isq_all),
            ("question_type",seg_qtype_all),("benchmark",seg_bench_all),("label_gold_binaria",seg_gold_all)
        ]:
            s = _seg(dfm, col)
            if s is not None:
                bucket.append(s)

        # kappa entre clasificadores NLI
        kdf, kr = kappa_por_modelo(dfm)
        if not kdf.empty: kappa_all_rows_list.append(kdf)
        if kr: kappa_range_rows.append(kr)

        # ranking y porcentajes por clasificador + consenso
        ranking_rows.append({
            "modelo":modelo,"porcentaje_alucinacion":(mean_rate*100.0 if mean_rate==mean_rate else np.nan),
            "n":n_prompts,"n_con_NLI":n_valid_prom
        })
        for name,col in [("roberta","label_roberta"),("deberta","label_deberta"),("distilroberta","label_distilroberta")]:
            pct_clf_rows.append({
                "modelo":modelo,"clasificador":name,
                "porcentaje": float((dfm[col]=="contradiction").mean()*100.0) if dfm[col].notna().any() else np.nan
            })
        pct_clf_rows.append({
            "modelo":modelo,"clasificador":"consenso",
            "porcentaje": float(dfm["es_alucinacion"].mean()*100.0) if dfm["es_alucinacion"].notna().any() else np.nan
        })

    # --- ensamblado global ---
    overview_df = pd.DataFrame(overview_rows).sort_values(
        ["hallucination_rate","n_con_NLI","n"], ascending=[False,False,False]
    )
    if not overview_df.empty:
        overview_df["pct_alucinadas_ci95_txt"] = (
            (overview_df["hallucination_rate"]*100).round(1).astype(str)
            + " [" + (overview_df["hallucination_rate_ci_lower"]*100).round(1).astype(str)
            + ", "   + (overview_df["hallucination_rate_ci_upper"]*100).round(1).astype(str) + "]"
        )

    # LOG resumen total prompts/respuestas
    try:
        tot_prompts = int(overview_df["N_prompts"].sum())
        print(f"[TOTAL] Prompts totales: {tot_prompts} | Respuestas totales: {tot_prompts}")
    except Exception:
        pass

    dist_df     = pd.DataFrame(dist_global, columns=["modelo","clasificador_nli","label","count"])

    seg_len_df   = pd.concat(seg_len_all,  ignore_index=True) if seg_len_all  else pd.DataFrame(columns=["modelo","length_class","rate","n"])
    seg_num_df   = pd.concat(seg_num_all,  ignore_index=True) if seg_num_all  else pd.DataFrame(columns=["modelo","has_numbers","rate","n"])
    seg_isq_df   = pd.concat(seg_isq_all,  ignore_index=True) if seg_isq_all  else pd.DataFrame(columns=["modelo","is_question","rate","n"])
    seg_qtype_df = pd.concat(seg_qtype_all,ignore_index=True) if seg_qtype_all else pd.DataFrame(columns=["modelo","question_type","rate","n"])
    seg_bench_df = pd.concat(seg_bench_all,ignore_index=True) if seg_bench_all else pd.DataFrame(columns=["modelo","benchmark","rate","n"])
    seg_gold_df  = pd.concat(seg_gold_all, ignore_index=True) if seg_gold_all else pd.DataFrame(columns=["modelo","label_gold_binaria","rate","n"])

    kappa_rows_df  = pd.concat(kappa_all_rows_list, ignore_index=True) if kappa_all_rows_list else pd.DataFrame(columns=["modelo","comparacion","kappa"])
    kappa_range_df = pd.DataFrame(kappa_range_rows) if kappa_range_rows else pd.DataFrame(columns=["modelo","max_kappa","min_kappa"])

    kappa_pivot = (kappa_rows_df.pivot_table(index="modelo", columns="comparacion", values="kappa").reset_index()
                   if not kappa_rows_df.empty else pd.DataFrame())
    heatmap_mb = seg_bench_df.copy()
    if not heatmap_mb.empty:
        heatmap_mb["porcentaje"] = heatmap_mb["rate"]*100.0
        heatmap_mb = heatmap_mb[["modelo","benchmark","porcentaje","n"]]
    else:
        heatmap_mb = pd.DataFrame(columns=["modelo","benchmark","porcentaje","n"])

    df_pct_clf = pd.DataFrame(pct_clf_rows, columns=["modelo","clasificador","porcentaje"])
    ranking_df = pd.DataFrame(sorted(
        ranking_rows, key=lambda x: (x["porcentaje_alucinacion"] if x["porcentaje_alucinacion"]==x["porcentaje_alucinacion"] else 1e9)
    ))
    metricas_gold_df = pd.DataFrame(metricas_gold_rows) if metricas_gold_rows else pd.DataFrame(
        columns=[
            "modelo","n_gold",
            "accuracy_gold","accuracy_ci_lower","accuracy_ci_upper",
            "precision_gold","precision_ci_lower","precision_ci_upper",
            "recall_gold","recall_ci_lower","recall_ci_upper",
            "f1_gold","f1_ci_lower","f1_ci_upper",
            "balanced_accuracy_gold","balanced_accuracy_ci_lower","balanced_accuracy_ci_upper",
            "mcc_gold","mcc_ci_lower","mcc_ci_upper",
            "tp","fp","fn","tn"
        ]
    )

    # Mann–Whitney (opcional)
    if gram_dir:
        df_gram_raw = cargar_gram_csvs(gram_dir)
        if not df_gram_raw.empty:
            cols = [c for c in ["modelo","benchmark","prompt","id","idx","es_alucinacion"] if c in df_all.columns]
            gram_join = unir_gram_con_json(df_gram_raw, df_all[cols].copy())
            mw_df = mann_whitney_desde_gram_unido(gram_join)
        else:
            mw_df = pd.DataFrame(columns=[
                "modelo","benchmark","variable","grupo_a_media","grupo_b_media",
                "U_statistic","p_value","p_adj_bh","significativo","significativo_fdr"
            ])
    else:
        mw_df = pd.DataFrame(columns=[
            "modelo","benchmark","variable","grupo_a_media","grupo_b_media",
            "U_statistic","p_value","p_adj_bh","significativo","significativo_fdr"
        ])

    # exportar
    def save_csv(df, name):
        path = os.path.join(output_dir, name)
        df.to_csv(path, index=False)

    save_csv(overview_df, "05_overview_por_modelo.csv")
    save_csv(metricas_gold_df, "05_metricas_vs_gold_por_modelo.csv")
    save_csv(metricas_gold_df, "05_metricas_consenso_vs_gold.csv")  # alias compat
    save_csv(dist_df, "05_distribucion_nli_por_modelo.csv")
    save_csv(seg_len_df, "05_tasa_por_length_class.csv")
    save_csv(seg_num_df, "05_tasa_por_has_numbers.csv")
    save_csv(seg_isq_df, "05_tasa_por_is_question.csv")
    save_csv(seg_qtype_df, "05_tasa_por_question_type.csv")
    save_csv(seg_bench_df, "05_tasa_por_benchmark.csv")
    save_csv(seg_gold_df, "05_tasa_por_label_gold_binaria.csv")
    save_csv(pd.concat([df_nli, df_gold], ignore_index=True, sort=False), "05_union_todos_los_modelos.csv")
    save_csv(kappa_rows_df, "05_cohen_kappa_por_modelo.csv")
    save_csv(kappa_range_df, "05_cohen_kappa_rango.csv")
    save_csv(kappa_pivot, "05_cohen_kappa_pivot.csv")
    save_csv(heatmap_mb, "05_heatmap_modelo_benchmark_consenso.csv")
    save_csv(df_pct_clf, "05_porcentaje_alucinaciones_por_modelo_y_clasificador.csv")
    save_csv(ranking_df, "05_ranking_porcentaje_alucinacion.csv")
    save_csv(mw_df, "05_mannwhitney_por_modelo_benchmark.csv")

    print("Archivos generados en:", output_dir)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Metricas por modelo fusionando NLI (04_) y GOLD (01_).")
    ap.add_argument("--input_dir_nli",  type=str, default=DEFAULT_INPUT_DIR_NLI,
                    help="Directorio base con respuestas_nli_*.json(.jsonl)")
    ap.add_argument("--input_dir_gold", type=str, default=DEFAULT_INPUT_DIR_GOLD,
                    help="Directorio base con respuestas_gold_*.json(.jsonl)")
    ap.add_argument("--output_dir",     type=str, default=DEFAULT_OUTPUT_DIR,
                    help="Directorio de salida para CSVs 05_")
    ap.add_argument("--gram_dir",       type=str, default=DEFAULT_GRAM_DIR,
                    help="(Opcional) Directorio CSVs gramaticales para Mann–Whitney")
    args = ap.parse_args()
    main(args.input_dir_nli, args.input_dir_gold, args.output_dir, args.gram_dir)
