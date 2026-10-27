#!/usr/bin/env python3
"""Test gate for final packaging (Yeepee, Huawei TechArena 2026 CTR).

A submission is zipped as FINAL only if its local test score reaches the bar
(default 0.83) on the primary cut C=25 (staleness-reweighted AUC when given,
otherwise overall AUC) AND it passes the format checks. Otherwise nothing is
written and the script exits with code 1.

Usage:
  python3 package_final.py --csv sub.csv --metrics metrics.json [--name v10] [--bar 0.83]
  python3 package_final.py --csv sub.csv --auc25 0.8312 [--rw25 0.8305]

metrics.json: either {"cut":25,"auc":..,"rw":..} or a list / jsonl of such rows
(the row with cut 25 is used).
"""
import argparse, hashlib, json, math, os, sys, zipfile
import pandas as pd

REF = "/workspace/huawei/v2work/out/submission_v8.csv"   # reference rowid set
N_ROWS = 529390

def load_c25(path):
    txt = open(path).read().strip()
    rows = []
    try:
        j = json.loads(txt); rows = j if isinstance(j, list) else [j]
    except json.JSONDecodeError:
        rows = [json.loads(l) for l in txt.splitlines() if l.strip()]
    c25 = [r for r in rows if int(r.get("cut", 25)) == 25]
    if not c25:
        sys.exit("GATE FAIL: no C=25 row in metrics file")
    return c25[-1]

def num(x):
    try:
        x = float(x); return None if math.isnan(x) else x
    except (TypeError, ValueError):
        return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--metrics"); ap.add_argument("--auc25", type=float); ap.add_argument("--rw25", type=float)
    ap.add_argument("--bar", type=float, default=0.83)
    ap.add_argument("--name"); ap.add_argument("--outdir")
    a = ap.parse_args()

    if a.metrics:
        r = load_c25(a.metrics); auc, rw = num(r.get("auc")), num(r.get("rw"))
    else:
        auc, rw = a.auc25, a.rw25
    score = rw if rw is not None else auc
    if score is None:
        sys.exit("GATE FAIL: no local C=25 score given; refusing to package")
    label = "C=25 reweighted AUC" if rw is not None else "C=25 AUC"
    print(f"{label}: {score:.5f}  (bar {a.bar})")
    if score < a.bar:
        print(f"GATE FAIL: {score:.5f} < {a.bar}. Not packaged."); sys.exit(1)

    df = pd.read_csv(a.csv)
    errs = []
    if list(df.columns) != ["rowid", "predicted_probability"]: errs.append(f"columns {list(df.columns)}")
    if len(df) != N_ROWS: errs.append(f"{len(df)} rows, expected {N_ROWS}")
    if df["rowid"].duplicated().any(): errs.append("duplicate rowids")
    p = df["predicted_probability"]
    if p.isna().any() or (p < 0).any() or (p > 1).any(): errs.append("probabilities missing or outside [0,1]")
    if os.path.exists(REF):
        ref = pd.read_csv(REF, usecols=["rowid"])["rowid"]
        if set(ref) != set(df["rowid"]): errs.append("rowid set differs from eval set")
    if errs:
        print("GATE FAIL (format): " + "; ".join(errs)); sys.exit(1)

    name = a.name or os.path.splitext(os.path.basename(a.csv))[0]
    outdir = a.outdir or os.path.dirname(os.path.abspath(a.csv))
    zp = os.path.join(outdir, f"{name}_FINAL.csv.zip")
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(a.csv, arcname=f"{name}.csv")
    sha = hashlib.sha256(open(zp, "rb").read()).hexdigest()
    with open(zp + ".gate.json", "w") as f:
        json.dump({"csv": os.path.abspath(a.csv), "zip": zp, "sha256": sha, "score_c25": score,
                   "metric": label, "bar": a.bar, "auc25": auc, "rw25": rw}, f, indent=1)
    print(f"GATE PASS: packaged {zp}\nsha256 {sha}")

if __name__ == "__main__":
    main()
