#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
06_visualizaciones.py
Lee los CSVs generados por 05_metricas_modelos.py y produce todas las gráficas
en una sola carpeta de salida (06_visualizaciones) + un PDF con gráficas y tablas.

Entradas esperadas (archivos en --input_dir_05):
- 05_overview_por_modelo.csv
- 05_metricas_vs_gold_por_modelo.csv   (alias: 05_metricas_consenso_vs_gold.csv)
- 05_distribucion_nli_por_modelo.csv
- 05_heatmap_modelo_benchmark_consenso.csv
- 05_porcentaje_alucinaciones_por_modelo_y_clasificador.csv
- 05_ranking_porcentaje_alucinacion.csv
- 05_cohen_kappa_pivot.csv  (y/o 05_cohen_kappa_por_modelo.csv)
- 05_mannwhitney_por_modelo_benchmark.csv   (opcional)

Salidas (imágenes .png) en --output_dir:
- 06_heatmap_modelo_benchmark.png
- 06_ranking_modelos.png
- 06_porcentajes_por_clasificador.png
- 06_hallucination_rate_ci.png
- 06_metricas_vs_gold_ci.png
- 06_kappa_heatmap.png
- 06_stack_nli_labels_roberta.png
- 06_stack_nli_labels_deberta.png
- 06_stack_nli_labels_distilroberta.png

Reporte PDF:
- 06_reporte_visualizaciones.pdf
  (portada + todas las imágenes + tablas transpuestas: overview, métricas GOLD, matriz de confusión,
   y páginas de Mann–Whitney: SIEMPRE muestra un resumen; además
   - si hay FDR-significativos: páginas con significativos, y
   - siempre: Top-20 global por p_value,
   - NUEVO: Top-10 por cada modelo por p_value.)
"""

import os
import argparse
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# ==== rutas por defecto (ajusta si hace falta) ====
DEFAULT_05_DIR = "/home/dslab/teilor/newtron/05_metricas/"
DEFAULT_06_DIR = "/home/dslab/teilor/newtron/06_visualizaciones/"
# ==================================================

TOP_N_MW_GLOBAL = 20   # Top-N global por p-value
TOP_N_MW_MODEL  = 10   # Top-N por modelo por p-value

plt.rcParams.update({
    "figure.dpi": 130,
    "savefig.dpi": 130,
    "font.size": 12,
    "axes.spines.right": False,
    "axes.spines.top": False,
})

def ensure_dir(p): os.makedirs(p, exist_ok=True)

def read_csv_safe(path_list):
    """Devuelve el primer CSV que exista en path_list, o None."""
    for p in path_list:
        if os.path.isfile(p):
            try:
                return pd.read_csv(p)
            except Exception:
                pass
    return None

def rename_model_column(df, col="modelo"):
    """deepseek-coder:* -> deepseek:* (mejor legibilidad en todas las salidas)."""
    if df is None or df.empty or col not in df.columns:
        return df
    out = df.copy()
    out[col] = out[col].astype(str).str.replace("deepseek-coder", "deepseek", regex=False)
    return out

def sort_models_index(df, col="modelo"):
    if df is None or df.empty or col not in df.columns:
        return df
    return df.sort_values(col).reset_index(drop=True)

# ------------------ PLOTS ------------------ #

def plot_heatmap_model_benchmark(df, outpath):
    if df is None or df.empty: return
    # Espera columnas: modelo, benchmark, porcentaje, n
    piv = df.pivot_table(index="modelo", columns="benchmark", values="porcentaje", aggfunc="mean")
    if piv.empty: return
    plt.figure(figsize=(11,6))
    im = plt.imshow(piv.values, aspect="auto")
    plt.colorbar(im, label="% de Alucinación (consenso)")
    plt.xticks(ticks=np.arange(piv.shape[1]), labels=piv.columns, rotation=15)
    plt.yticks(ticks=np.arange(piv.shape[0]), labels=piv.index)
    plt.title("Hallucination Heatmap by Model and Benchmark")
    # anotar celdas
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            val = piv.values[i, j]
            if not np.isnan(val):
                plt.text(j, i, f"{val:.1f}", ha="center", va="center", fontsize=10, weight="bold")
    plt.tight_layout()
    plt.savefig(outpath, bbox_inches="tight")
    plt.close()

def plot_ranking(df, outpath):
    if df is None or df.empty: return
    # Espera columnas: modelo, porcentaje_alucinacion
    df = df.copy()
    df = df.sort_values("porcentaje_alucinacion", ascending=True)
    plt.figure(figsize=(9,5.5))
    y = np.arange(len(df))
    plt.barh(y, df["porcentaje_alucinacion"].values)
    for i, v in enumerate(df["porcentaje_alucinacion"].values):
        if not np.isnan(v):
            plt.text(v + 0.5, i, f"{v:.0f}%", va="center")
    plt.yticks(y, df["modelo"])
    plt.xlabel("% de respuestas alucinadas (consenso)")
    plt.title("Model Ranking by Hallucination Rate")
    plt.tight_layout()
    plt.savefig(outpath, bbox_inches="tight")
    plt.close()

def plot_porcentajes_clasificador(df, outpath):
    if df is None or df.empty: return
    # Espera: modelo, clasificador, porcentaje
    df = df.copy()
    modelos = sorted(df["modelo"].unique().tolist())
    clasificadores = ["roberta","deberta","distilroberta","consenso"]
    plt.figure(figsize=(12,6))
    width = 0.18
    x = np.arange(len(modelos))
    for k, clf in enumerate(clasificadores):
        sub = df[df["clasificador"]==clf].set_index("modelo").reindex(modelos)
        vals = sub["porcentaje"].to_numpy()
        plt.bar(x + (k-1.5)*width, vals, width=width, label=clf.capitalize())
        # anotaciones
        for i, v in enumerate(vals):
            if not np.isnan(v):
                plt.text(x[i] + (k-1.5)*width, v+0.3, f"{v:.0f}%", ha="center", va="bottom", fontsize=9)
    plt.xticks(x, modelos, rotation=20)
    plt.ylabel("%")
    plt.title("Hallucination Rates Across Models and NLI Classifiers")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, bbox_inches="tight")
    plt.close()

def plot_hallucination_rate_ci(df, outpath):
    if df is None or df.empty: return
    # Espera: modelo, hallucination_rate, hallucination_rate_ci_lower, hallucination_rate_ci_upper
    df = df.copy()
    df = df.sort_values("hallucination_rate", ascending=True)
    x = np.arange(len(df))
    y = df["hallucination_rate"].to_numpy()*100
    ylo = df["hallucination_rate_ci_lower"].to_numpy()*100
    yhi = df["hallucination_rate_ci_upper"].to_numpy()*100
    err_low = y - ylo
    err_high = yhi - y
    plt.figure(figsize=(10,5.5))
    plt.errorbar(x, y, yerr=[err_low, err_high], fmt="o", capsize=4)
    for i, v in enumerate(y):
        if not np.isnan(v):
            plt.text(i, v + 1.2, f"{v:.0f}%", ha="center", fontsize=10)
    plt.xticks(x, df["modelo"], rotation=20)
    plt.ylabel("% de respuestas alucinadas (consenso)")
    plt.title("Hallucination Rate (95% CI via bootstrap)")
    plt.tight_layout()
    plt.savefig(outpath, bbox_inches="tight")
    plt.close()

def plot_metricas_vs_gold_ci(df, outpath):
    if df is None or df.empty: return
    # Columnas esperadas con CI:
    # accuracy_gold, *_ci_lower, *_ci_upper, precision_gold, recall_gold, f1_gold,
    # balanced_accuracy_gold, mcc_gold
    df = df.copy()
    modelos = df["modelo"].tolist()
    metrics = [
        ("accuracy_gold","Accuracy"),
        ("precision_gold","Precision"),
        ("recall_gold","Recall"),
        ("f1_gold","F1"),
        ("balanced_accuracy_gold","Balanced Acc."),
        ("mcc_gold","MCC"),
    ]
    plt.figure(figsize=(12,7))
    n = len(modelos)
    x = np.arange(n)
    width = 0.14

    for i, (col, label) in enumerate(metrics):
        y = df[col].to_numpy()*100  # mostramos en %
        lo = df.get(col.replace("_gold","")+"_ci_lower", np.nan).to_numpy()
        hi = df.get(col.replace("_gold","")+"_ci_upper", np.nan).to_numpy()
        if not np.all(np.isnan(lo)) and not np.all(np.isnan(hi)):
            lo = lo*100; hi = hi*100
            err = [y - lo, hi - y]
            plt.errorbar(x + (i-2.5)*width, y, yerr=err, fmt="o", capsize=3, label=label)
        else:
            plt.plot(x + (i-2.5)*width, y, "o", label=label)

    plt.axhline(50, lw=0.8, ls="--")
    plt.xticks(x, modelos, rotation=20)
    plt.ylabel("Porcentaje / Índice (×100)")
    plt.title("Métricas vs GOLD (95% CI)")
    plt.legend(ncol=3, frameon=False)
    plt.tight_layout()
    plt.savefig(outpath, bbox_inches="tight")
    plt.close()

def plot_kappa_heatmap(df_pivot, df_rows, outpath):
    # Preferimos pivot si está; si no, construimos a partir de rows
    if (df_pivot is None or df_pivot.empty) and (df_rows is None or df_rows.empty):
        return
    if df_pivot is None or df_pivot.empty:
        # construir pivot
        tmp = df_rows.pivot_table(index="modelo", columns="comparacion", values="kappa", aggfunc="mean")
    else:
        tmp = df_pivot.set_index("modelo")
    if tmp.empty: return
    tmp = tmp.sort_index()
    plt.figure(figsize=(10,5.2))
    im = plt.imshow(tmp.values, aspect="auto", vmin=-1, vmax=1)
    plt.colorbar(im, label="Cohen's κ (binario por contradiction)")
    plt.xticks(np.arange(tmp.shape[1]), tmp.columns, rotation=15)
    plt.yticks(np.arange(tmp.shape[0]), tmp.index)
    plt.title("Agreement entre clasificadores NLI")
    for i in range(tmp.shape[0]):
        for j in range(tmp.shape[1]):
            v = tmp.values[i, j]
            if not np.isnan(v):
                plt.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=10, weight="bold")
    plt.tight_layout()
    plt.savefig(outpath, bbox_inches="tight")
    plt.close()

def plot_stack_nli_labels_per_clf(df, outdir, prefix="06_stack_nli_labels"):
    """Genera un PNG por clasificador (sin subplots): roberta, deberta, distilroberta."""
    if df is None or df.empty: return []
    df = df.copy()
    modelos = sorted(df["modelo"].unique().tolist())
    clasificadores = ["roberta","deberta","distilroberta"]
    etiquetas = ["contradiction","entailment","neutral", "nan"]
    outputs = []
    for clf in clasificadores:
        sub = df[df["clasificador_nli"]==clf]
        if sub.empty: continue
        piv = sub.pivot_table(index="modelo", columns="label", values="count", aggfunc="sum").reindex(modelos)
        for e in etiquetas:
            if e not in piv.columns:
                piv[e] = 0
        piv = piv[etiquetas]
        tot = piv.sum(axis=1).replace(0, np.nan)
        pct = (piv.T / tot).T * 100.0
        plt.figure(figsize=(12, 4.2))
        bottom = np.zeros(len(modelos))
        for e in etiquetas:
            vals = pct[e].to_numpy()
            plt.bar(modelos, vals, bottom=bottom, label=e.capitalize())
            bottom += np.nan_to_num(vals)
        plt.ylabel("%")
        plt.title(f"Distribución de etiquetas NLI – {clf.capitalize()}")
        plt.xticks(rotation=20)
        plt.legend(ncol=4, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.15))
        plt.tight_layout()
        outpath = os.path.join(outdir, f"{prefix}_{clf}.png")
        plt.savefig(outpath, bbox_inches="tight")
        plt.close()
        outputs.append(outpath)
    return outputs

# ------------------ FORMATEO DE TABLAS (PDF) ------------------ #

def _fmt_pct(x, digits=1):
    try:
        if pd.isna(x): return ""
        return f"{x*100:.{digits}f}"
    except Exception:
        return ""

def _fmt_pct_ci(v, lo, hi, digits=1):
    try:
        if pd.isna(v): return ""
        return f"{v*100:.{digits}f} [{(lo*100 if not pd.isna(lo) else np.nan):.{digits}f}, {(hi*100 if not pd.isna(hi) else np.nan):.{digits}f}]"
    except Exception:
        return ""

def _fmt_num(x, digits=3):
    try:
        if pd.isna(x): return ""
        return f"{x:.{digits}f}"
    except Exception:
        return ""

def _fmt_num_ci(v, lo, hi, digits=3):
    try:
        if pd.isna(v): return ""
        return f"{v:.{digits}f} [{(lo if not pd.isna(lo) else np.nan):.{digits}f}, {(hi if not pd.isna(hi) else np.nan):.{digits}f}]"
    except Exception:
        return ""

def build_transposed_tables(df_overview, df_gold):
    """Devuelve (overview_T, gold_T_metrics, gold_T_counts) transpuestos."""
    overview_T = None
    gold_T_metrics = None
    gold_T_counts = None

    if df_overview is not None and not df_overview.empty:
        d = df_overview.copy().sort_values("modelo")
        models = d["modelo"].tolist()
        overview_rows = {
            "N": d["n"].astype(int).tolist(),
            "Con NLI": d["n_con_NLI"].astype(int).tolist(),
            "% con NLI": [_fmt_pct(x) for x in d["pct_con_NLI"].tolist()],
            "% resp. alucinadas (CI95)": [_fmt_pct_ci(v, lo, hi) for v,lo,hi in zip(d["hallucination_rate"], d["hallucination_rate_ci_lower"], d["hallucination_rate_ci_upper"])],
            "Consistencia mayoría": [_fmt_num(x,2) for x in d["consistencia_mayoria"].tolist()],
        }
        overview_T = pd.DataFrame(overview_rows, index=models).T.reset_index().rename(columns={"index":"Métrica"})

    if df_gold is not None and not df_gold.empty:
        g = df_gold.copy().sort_values("modelo")
        models = g["modelo"].tolist()
        def pack_pct(metric):
            return [_fmt_pct_ci(v, lo, hi) for v, lo, hi in zip(
                g[f"{metric}_gold"], g[f"{metric}_ci_lower"], g[f"{metric}_ci_upper"]
            )]
        def pack_num(metric):
            return [_fmt_num_ci(v, lo, hi, 3) for v, lo, hi in zip(
                g[f"{metric}_gold"], g[f"{metric}_ci_lower"], g[f"{metric}_ci_upper"]
            )]
        gold_T_metrics = pd.DataFrame({
            "Métrica": ["Accuracy % (CI95)","Precision % (CI95)","Recall % (CI95)","F1 % (CI95)","Balanced Acc. % (CI95)","MCC (CI95)"]
        })
        cols = {
            "Accuracy % (CI95)": pack_pct("accuracy"),
            "Precision % (CI95)": pack_pct("precision"),
            "Recall % (CI95)": pack_pct("recall"),
            "F1 % (CI95)": pack_pct("f1"),
            "Balanced Acc. % (CI95)": pack_pct("balanced_accuracy"),
            "MCC (CI95)": pack_num("mcc"),
        }
        # añadir cada modelo como columna
        for model, acc, prec, rec, f1, bal, mcc in zip(models, cols["Accuracy % (CI95)"], cols["Precision % (CI95)"], cols["Recall % (CI95)"], cols["F1 % (CI95)"], cols["Balanced Acc. % (CI95)"], cols["MCC (CI95)"]):
            gold_T_metrics[model] = [acc, prec, rec, f1, bal, mcc]

        # Conteos matriz de confusión
        gold_T_counts = pd.DataFrame({"Métrica": ["TP","FP","FN","TN"]})
        for model, tp, fp, fn, tn in zip(models, g["tp"].astype(int), g["fp"].astype(int), g["fn"].astype(int), g["tn"].astype(int)):
            gold_T_counts[model] = [tp, fp, fn, tn]

    return overview_T, gold_T_metrics, gold_T_counts

# ------------------ PDF ------------------ #

def build_pdf_report(output_dir, image_files, df_overview=None, df_gold=None, df_mw=None):
    """Crea el PDF consolidado con tablas transpuestas y páginas de Mann–Whitney."""
    pdf_path = os.path.join(output_dir, "06_reporte_visualizaciones.pdf")
    overview_T, gold_T_metrics, gold_T_counts = build_transposed_tables(df_overview, df_gold)

    with PdfPages(pdf_path) as pdf:
        # Portada
        plt.figure(figsize=(11.69, 8.27))  # A4 apaisado aprox
        plt.axis("off")
        plt.title("Reporte de Visualizaciones – 06_", fontsize=20, pad=20)
        plt.text(0.02, 0.84, f"Generado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", fontsize=12)
        plt.text(0.02, 0.78, f"Carpeta de salida: {output_dir}", fontsize=10)
        plt.text(0.02, 0.72, "Nota: porcentajes ⇒ % de respuestas alucinadas (consenso). Δ en Mann–Whitney: μ(aluc.) − μ(no aluc.)", fontsize=10)
        pdf.savefig(bbox_inches="tight"); plt.close()

        # Cada imagen, una página
        for img_path in image_files:
            if not os.path.isfile(img_path):
                continue
            img = plt.imread(img_path)
            plt.figure(figsize=(11.69, 8.27))
            plt.imshow(img)
            plt.axis("off")
            plt.title(os.path.basename(img_path), fontsize=11, pad=6)
            pdf.savefig(bbox_inches="tight"); plt.close()

        # Tablas transpuestas con fuente reducida
        def render_table(df, title, fontsize=7, scale_y=1.05):
            if df is None or df.empty: return
            fig = plt.figure(figsize=(11.69, 8.27)); ax = fig.add_subplot(111); ax.axis('off')
            ax.set_title(title, fontsize=13, pad=8)
            table = ax.table(cellText=df.values, colLabels=df.columns, loc='center')
            table.auto_set_font_size(False); table.set_fontsize(fontsize); table.scale(1, scale_y)
            pdf.savefig(bbox_inches="tight"); plt.close(fig)

        render_table(overview_T, "Overview por modelo", fontsize=7, scale_y=1.05)
        render_table(gold_T_metrics, "Métricas vs GOLD por modelo", fontsize=7, scale_y=1.05)
        render_table(gold_T_counts, "Matriz de confusión vs GOLD", fontsize=7, scale_y=1.05)

        # Mann–Whitney (resumen + significativos + Top-N)
        if df_mw is not None and not df_mw.empty:
            m = df_mw.copy()
            # Tipos
            for c in ["grupo_a_media","grupo_b_media","U_statistic","p_value","p_adj_bh"]:
                if c in m.columns:
                    m[c] = pd.to_numeric(m[c], errors="coerce")
            if "significativo_fdr" in m.columns:
                m["significativo_fdr"] = m["significativo_fdr"].astype(str).str.lower().eq("true")
            else:
                m["significativo_fdr"] = False

            m["delta"] = m["grupo_a_media"] - m["grupo_b_media"]

            # Resumen por modelo
            summary = (m.groupby("modelo")
                         .agg(n_tests=("variable","count"),
                              n_sig_fdr=("significativo_fdr","sum"),
                              min_p=("p_value","min"),
                              min_p_adj=("p_adj_bh","min"))
                         .reset_index()
                         .sort_values(["n_sig_fdr","n_tests"], ascending=[False, False]))
            for c in ["min_p","min_p_adj"]:
                summary[c] = summary[c].round(6)
            render_table(summary, "Mann–Whitney: resumen por modelo", fontsize=8, scale_y=1.10)

            # Páginas de significativos (FDR)
            m_sig = m[m["significativo_fdr"] == True].copy()
            if not m_sig.empty:
                m_sig = m_sig.sort_values(["p_adj_bh","modelo","benchmark","variable"]).reset_index(drop=True)
                ms = m_sig.copy()
                ms["μ(aluc.)"] = ms["grupo_a_media"].round(3)
                ms["μ(no aluc.)"] = ms["grupo_b_media"].round(3)
                ms["Δ"] = ms["delta"].round(3)
                ms["p_adj_bh"] = ms["p_adj_bh"].round(6)
                tbl = ms[["modelo","benchmark","variable","μ(aluc.)","μ(no aluc.)","Δ","p_adj_bh"]]

                rows_per_page = 22
                for start in range(0, len(tbl), rows_per_page):
                    chunk = tbl.iloc[start:start+rows_per_page]
                    render_table(chunk, "Mann–Whitney (FDR): resultados significativos", fontsize=7, scale_y=1.05)

            # Top-N global por p_value (siempre)
            m_top = m.sort_values("p_value", ascending=True).head(TOP_N_MW_GLOBAL).copy()
            if not m_top.empty:
                m_top["μ(aluc.)"] = m_top["grupo_a_media"].round(3)
                m_top["μ(no aluc.)"] = m_top["grupo_b_media"].round(3)
                m_top["Δ"] = m_top["delta"].round(3)
                m_top["p_value"] = m_top["p_value"].round(6)
                m_top["p_adj_bh"] = m_top["p_adj_bh"].round(6)
                tbl_top = m_top[["modelo","benchmark","variable","μ(aluc.)","μ(no aluc.)","Δ","p_value","p_adj_bh"]]
                render_table(tbl_top, f"Mann–Whitney: Top-{TOP_N_MW_GLOBAL} por p-value (global)", fontsize=7, scale_y=1.05)

            # NUEVO: Top-N por modelo por p-value (siempre)
            modelos = [x for x in m["modelo"].dropna().unique().tolist()]
            for mod in sorted(modelos):
                sub = m[m["modelo"] == mod].sort_values("p_value", ascending=True).head(TOP_N_MW_MODEL).copy()
                if sub.empty: 
                    continue
                sub["μ(aluc.)"] = sub["grupo_a_media"].round(3)
                sub["μ(no aluc.)"] = sub["grupo_b_media"].round(3)
                sub["Δ"] = (sub["grupo_a_media"] - sub["grupo_b_media"]).round(3)
                sub["p_value"] = sub["p_value"].round(6)
                sub["p_adj_bh"] = sub["p_adj_bh"].round(6)
                tbl_mod = sub[["benchmark","variable","μ(aluc.)","μ(no aluc.)","Δ","p_value","p_adj_bh"]]
                render_table(tbl_mod, f"Mann–Whitney: Top-{TOP_N_MW_MODEL} por p-value – {mod}", fontsize=7, scale_y=1.05)

    return pdf_path

# ------------------ MAIN ------------------ #

def main(input_dir_05: str, output_dir: str):
    input_dir_05 = os.path.abspath(input_dir_05)
    output_dir   = os.path.abspath(output_dir)
    ensure_dir(output_dir)

    # Cargar CSVs
    df_overview = read_csv_safe([os.path.join(input_dir_05, "05_overview_por_modelo.csv")])
    df_gold     = read_csv_safe([
        os.path.join(input_dir_05, "05_metricas_vs_gold_por_modelo.csv"),
        os.path.join(input_dir_05, "05_metricas_consenso_vs_gold.csv"),
    ])
    df_dist     = read_csv_safe([os.path.join(input_dir_05, "05_distribucion_nli_por_modelo.csv")])
    df_heat     = read_csv_safe([os.path.join(input_dir_05, "05_heatmap_modelo_benchmark_consenso.csv")])
    df_pctclf   = read_csv_safe([os.path.join(input_dir_05, "05_porcentaje_alucinaciones_por_modelo_y_clasificador.csv")])
    df_rank     = read_csv_safe([os.path.join(input_dir_05, "05_ranking_porcentaje_alucinacion.csv")])
    df_kpivot   = read_csv_safe([os.path.join(input_dir_05, "05_cohen_kappa_pivot.csv")])
    df_krows    = read_csv_safe([os.path.join(input_dir_05, "05_cohen_kappa_por_modelo.csv")])
    df_mw       = read_csv_safe([os.path.join(input_dir_05, "05_mannwhitney_por_modelo_benchmark.csv")])

    # Renombrado deepseek-coder -> deepseek en todas las tablas
    for name in ["df_overview","df_gold","df_dist","df_heat","df_pctclf","df_rank","df_kpivot","df_krows","df_mw"]:
        obj = locals().get(name)
        if isinstance(obj, pd.DataFrame):
            locals()[name] = rename_model_column(obj)

    # Orden básico
    df_overview = sort_models_index(df_overview)
    df_gold     = sort_models_index(df_gold)
    df_pctclf   = sort_models_index(df_pctclf)
    df_heat     = sort_models_index(df_heat)
    df_rank     = sort_models_index(df_rank)
    df_dist     = sort_models_index(df_dist)
    if df_kpivot is not None:
        df_kpivot = sort_models_index(df_kpivot)

    # Generar plots (una figura por gráfico; sin colores explícitos)
    created = []

    f = os.path.join(output_dir, "06_heatmap_modelo_benchmark.png")
    plot_heatmap_model_benchmark(df_heat, f)
    if os.path.isfile(f): created.append(f)

    f = os.path.join(output_dir, "06_ranking_modelos.png")
    plot_ranking(df_rank, f)
    if os.path.isfile(f): created.append(f)

    f = os.path.join(output_dir, "06_porcentajes_por_clasificador.png")
    plot_porcentajes_clasificador(df_pctclf, f)
    if os.path.isfile(f): created.append(f)

    f = os.path.join(output_dir, "06_hallucination_rate_ci.png")
    plot_hallucination_rate_ci(df_overview, f)
    if os.path.isfile(f): created.append(f)

    f = os.path.join(output_dir, "06_metricas_vs_gold_ci.png")
    plot_metricas_vs_gold_ci(df_gold, f)
    if os.path.isfile(f): created.append(f)

    f = os.path.join(output_dir, "06_kappa_heatmap.png")
    plot_kappa_heatmap(df_kpivot, df_krows, f)
    if os.path.isfile(f): created.append(f)

    # Distribución de etiquetas NLI: un PNG por clasificador
    created += plot_stack_nli_labels_per_clf(df_dist, output_dir, "06_stack_nli_labels")

    # PDF consolidado con tablas transpuestas y MW (con Top-20 global + Top-10 por modelo)
    pdf_path = build_pdf_report(output_dir, created, df_overview, df_gold, df_mw)

    print("Imagenes guardadas en:", output_dir)
    print("PDF generado:", pdf_path)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Graficador 06_: visualizaciones + PDF (tablas transpuestas y Mann–Whitney).")
    ap.add_argument("--input_dir_05", type=str, default=DEFAULT_05_DIR,
                    help="Carpeta con los CSVs de 05_metricas")
    ap.add_argument("--output_dir", type=str, default=DEFAULT_06_DIR,
                    help="Carpeta de salida para las imágenes y el PDF")
    args = ap.parse_args()
    main(args.input_dir_05, args.output_dir)
