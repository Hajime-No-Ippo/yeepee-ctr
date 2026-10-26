"""Time-based splits that mirror the official evaluation.

Evaluation = each user's FIRST event at/after the cutoff. So for a cutoff C:
  fit events  = all train events with ts < C            (labels usable for aggregates)
  target/val  = for each user, the first train event with C <= ts < C+W (W=None -> until end of train)
Whole events always stay together (split is by event, rows follow their event).
"""
import polars as pl

J20 = 1737331200          # 2025-01-20 00:00 UTC (start of train window; earlier clicks = warm-up)
DAY = 86400
C_MAIN = J20 + 5 * DAY    # 2025-01-25
C_SECOND = J20 + 6 * DAY  # 2025-01-26
C_FINAL = J20 + 7 * DAY   # 2025-01-27 (official cutoff)

def fit_events(TE: pl.DataFrame, C: int) -> pl.DataFrame:
    return TE.filter(pl.col('ts') < C)

def first_events(TE: pl.DataFrame, C: int, W: int | None = None) -> pl.DataFrame:
    """First event per user in [C, C+W). Ties on ts broken by eid (deterministic)."""
    m = pl.col('ts') >= C
    if W is not None:
        m = m & (pl.col('ts') < C + W)
    return TE.filter(m).sort('ts', 'eid').group_by('u', maintain_order=True).first()

def split(TE: pl.DataFrame, TR: pl.DataFrame, C: int, W: int | None = None):
    """Return (fit_ev, fit_rows, val_ev, val_rows)."""
    fe = fit_events(TE, C)
    ve = first_events(TE, C, W)
    fr = TR.join(fe.select('eid'), on='eid', how='semi')
    vr = TR.join(ve.select('eid'), on='eid', how='semi')
    return fe, fr, ve, vr

def train_fold_cutoffs(C_val: int):
    """Daily cutoffs strictly before C_val used to build model-training rows (each fold: first event/user in [Ck, Ck+1d))."""
    return [J20 + k * DAY for k in range(1, (C_val - J20) // DAY)]
