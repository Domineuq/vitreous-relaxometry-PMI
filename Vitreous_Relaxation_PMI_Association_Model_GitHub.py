# -*- coding: utf-8 -*-
"""
@author: Dr. Dominique Neuhaus

For each relaxation metric (T1, T2, T2s), fit:
    PMI ~ Temp_metric + Relax_metric
using OLS with intercept.

Outputs:
- master_table.csv (merged)
- model_summary.csv (one row per metric with coefficients, p-values, R2, etc.)
- {metric}_PMI_base_summary.txt (full statsmodels summary)
"""

PMI_XLSX  = r"C:path/to/excel_sheet/mean_PMI.xlsx"  # columns: Case, PMI (hours)
TEMP_XLSX = r"C:path/to/excel_sheet/mean_forhead_temperatures.xlsx"  # columns: Case, temp_T1, temp_T2, temp_T2s
CSV_DIR   = r"path/to/directory/MeanRelaxations"  # directory of per-case CSV files containing relaxation values
OUT_DIR   = r"path/to/directory/out_dir"  # output directory to save results

import os, glob, sys
import numpy as np
import pandas as pd
import statsmodels.api as sm


# I/O helpers
# ----------------------------
def read_pmi_xlsx(path: str) -> pd.DataFrame:
    df = pd.read_excel(path)
    df.columns = [c.strip() for c in df.columns]

    if "Case" not in df.columns:
        raise ValueError('PMI file must have column "Case"')

    pmi_col = None
    for c in df.columns:
        if str(c).lower().startswith("pmi"):
            pmi_col = c
            break
    if pmi_col is None:
        raise ValueError('PMI file must have "PMI" column (hours)')

    out = df[["Case", pmi_col]].rename(columns={pmi_col: "PMI"}).copy()
    out["Case"] = out["Case"].astype(str).str.strip()
    out["PMI"] = pd.to_numeric(out["PMI"], errors="coerce")
    return out


def read_temp_xlsx(path: str) -> pd.DataFrame:
    df = pd.read_excel(path)
    df.columns = [c.strip() for c in df.columns]

    need = ["Case", "temp_T1", "temp_T2", "temp_T2s"]
    for n in need:
        if n not in df.columns:
            raise ValueError(f'Missing column "{n}" in temperature file')

    out = df[need].copy()
    out["Case"] = out["Case"].astype(str).str.strip()
    for c in ["temp_T1", "temp_T2", "temp_T2s"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def read_case_csvs(csv_dir: str) -> pd.DataFrame:
    rows = []
    csvs = sorted(glob.glob(os.path.join(csv_dir, "*.csv")))
    if not csvs:
        raise ValueError(f"No CSV files found in {csv_dir}")

    for p in csvs:
        try:
            df = pd.read_csv(p)
        except Exception as e:
            print(f"[WARN] {p}: {e}", file=sys.stderr)
            continue

        df.columns = [c.strip() for c in df.columns]
        required = ["case", "metric", "value"]
        if not all(r in df.columns for r in required):
            print(f"[WARN] {p} missing one of {required}", file=sys.stderr)
            continue

        # Keep first occurrence per metric to avoid duplicates
        firsts = df.drop_duplicates(subset=["metric"], keep="first")
        wide = (
            firsts.pivot_table(index="case", columns="metric", values="value", aggfunc="first")
            .reset_index()
        )

        ren = {}
        for c in wide.columns:
            cl = str(c).strip().lower()
            if cl == "t1":
                ren[c] = "T1"
            elif cl == "t2":
                ren[c] = "T2"
            elif cl in ("t2s", "t2*", "t2star", "t2-star") or ("t2" in cl and "*" in cl):
                ren[c] = "T2s"
        wide = wide.rename(columns=ren)

        for _, r in wide.iterrows():
            rows.append({
                "Case": str(r.get("case", np.nan) or r.get("Case", np.nan)).strip(),
                "T1":  pd.to_numeric(r.get("T1",  np.nan), errors="coerce"),
                "T2":  pd.to_numeric(r.get("T2",  np.nan), errors="coerce"),
                "T2s": pd.to_numeric(r.get("T2s", np.nan), errors="coerce"),
            })

    out = pd.DataFrame(rows).drop_duplicates(subset=["Case"], keep="first")
    if out.empty:
        raise ValueError("No valid metric rows from CSVs.")
    return out


def build_master(pmi_path, temp_path, csv_dir) -> pd.DataFrame:
    pmi  = read_pmi_xlsx(pmi_path)
    temp = read_temp_xlsx(temp_path)
    mets = read_case_csvs(csv_dir)

    df = pmi.merge(temp, on="Case", how="inner").merge(mets, on="Case", how="inner")
    return df


# Modeling: PMI as outcome
# ----------------------------
def fit_base(sub: pd.DataFrame, metric: str):
    """PMI ~ Temp(metric) + Relaxation(metric)"""
    temp_col = {"T1": "temp_T1", "T2": "temp_T2", "T2s": "temp_T2s"}[metric]
    y = sub["PMI"]
    X = sm.add_constant(sub[[temp_col, metric]], has_constant="add")
    return sm.OLS(y, X).fit()


# Main
# ----------------------------
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    df = build_master(PMI_XLSX, TEMP_XLSX, CSV_DIR)
    df.to_csv(os.path.join(OUT_DIR, "master_table.csv"), index=False)
    print("[OK] merged rows:", len(df))

    rows = []
    for metric in ["T1", "T2", "T2s"]:
        temp_col = {"T1": "temp_T1", "T2": "temp_T2", "T2s": "temp_T2s"}[metric]

        # Need PMI + Temp + Relaxation(metric)
        sub = df[["Case", "PMI", metric, temp_col]].dropna().copy()
        if sub.empty or len(sub) < 5:
            print("[WARN] not enough data for", metric, "(n<5 after dropna)")
            continue

        model = fit_base(sub, metric)

        rec = {
            "metric": metric,
            "n": int(model.nobs),
            "AIC": float(model.aic),
            "BIC": float(model.bic),
            "R2": float(model.rsquared),
            "R2_adj": float(model.rsquared_adj),
            "F_p": float(model.f_pvalue),
        }

        # Coefficients + p-values for base model terms
        for term in ["const", temp_col, metric]:
            rec[f"coef_{term}"] = float(model.params.get(term, np.nan))
            rec[f"se_{term}"] = float(model.bse.get(term, np.nan))
            rec[f"t_{term}"] = float(model.tvalues.get(term, np.nan))
            rec[f"p_{term}"] = float(model.pvalues.get(term, np.nan))
        rows.append(rec)

        # Save full statsmodels summary
        with open(os.path.join(OUT_DIR, f"{metric}_PMI_base_summary.txt"), "w", encoding="utf-8") as f:
            f.write(model.summary().as_text())

    if rows:
        pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, "model_summary.csv"), index=False)
        print("[OK] wrote model_summary.csv with", len(rows), "rows")
    else:
        print("[WARN] no models were fitted (no metric had enough data).")

    print("[DONE] Outputs:", OUT_DIR)


if __name__ == "__main__":
    main()