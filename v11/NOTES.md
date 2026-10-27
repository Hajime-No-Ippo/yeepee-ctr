# v11 notes (quality first; runtime recorded, not constrained). Times Europe/Dublin (UTC+0).
Harness: copy of v10 (cl.py D=v11, ens.py +--save rounds, e.sh runner 8 threads, cmp11.py NAME SPEC -> cmp11_results.jsonl with d8_ (vs v8 trim) and d10_ (vs v10B)).
Feature dirs symlinked read-only (feat_sq10 -> v10/feat_sq10). seq11.py = seq10.py + env knobs SQ_D/SQ_DH/SQ_H/SQ_HID/SQ_HL/SQ_SEED/SQ_TEMB.
Refs: v8 trim C25 .81321 C23 .79056 C26 .82188 | v10B C25 .81388 C23 .79102 (rw .78984) C26 .82305. Accept: +C25, C23 not < -0.0002.
## Checkpoints
- 17:55 b1.sh: LightGBM capacity (nf1 = negfrac 1.0 for b and s; big1 = lr .015 leaves127 ff.5 l2 30 maxbin255, 3000 rounds) on cuts 25,23.
- queued sw1.sh: raw seq-model sweep on val25 (base, ep5, d64, temb, hl50, np6, h4, seed1) -> sw/*
