# dw_lb findings — what could close 0.818 → 0.83? (Grok Bot / Eric, 2026-10-28)
All Δ = global AUC vs a control model (same residual LGB on v12 logits with z-derived context only), user-grouped 5-fold CV, event bootstrap 95% CI. v12 val logits reproduced exactly (C25 0.81723, C23 0.79566). Null-noise feature: Δ −0.00006 (C25), 0.00000 (C23) → harness false-positive floor ≈ ±0.0001.

## 1. Public intel (honest: nothing concrete exists)
- Official description only: "Competing in the Intelligent Marketplace: Generative Sequential Recommendation Under Constraints" — feedback loop, multiple agents competing, compute/runtime constraints. https://huaweiireland.teamtailor.com/jobs/8447633-huawei-ireland-tech-arena-2026 , https://www.siliconrepublic.com/machines/how-huaweis-tech-arena-2026-aims-to-develop-irelands-ai-talent , https://www.hackathon.com/event/huawei-ireland-tech-arena-2026-6ac54367612836bf413df387 (Phase 1 Oct 21–Nov 4, top 15 announced Nov 9).
- No public Phase-1 solutions, leaderboard write-ups or CTR/AUC specifics found. The 0.83 figure is hearsay; unverifiable. Older edition posts (LinkedIn, e.g. https://www.linkedin.com/posts/rahulraj1406_huawei-techarena-llm-activity-7397952145213292544-JDot) concern a different task.

## 2. Ranked ΔAUC (C25 / C23)
| # | lever | C25 | C23 | verdict |
|---|---|---|---|---|
| 1 | User click-propensity (28 feats, feat_prop.py) | +0.00020 [+0.00008,+0.00033] | +0.00026 [+0.00012,+0.00043] | real but tiny |
| 2 | Transductive family f2 co-occurrence (VAL only) | +0.00013 [+0.00004,+0.00023] (seed1 +0.00012) | −0.00005 | ≈0 |
| 3 | Transductive f3 history popularity | +0.00010 / +0.00013 | +0.00009 | ≈0 (CI spans 0) |
| 4 | Transductive all 60 | +0.00010 [−0.00008,+0.00024] | +0.00012 [−0.00004,+0.00028] | ≈0 |
| 5 | Transductive f1 exposure | −0.00002 / −0.00005 | +0.00008 | 0 |
| 6 | Transductive f4 timing | +0.00003 / +0.00005 | +0.00007 | 0 |
| 7 | transductive + propensity (88) | +0.00018 [−0.00001,+0.00033] | +0.00029 [+0.00013,+0.00047] | = propensity alone |
| 8 | Two-stage event model P(ev)·softmax (ctx / ctx+prop) | glob 0.81707 / 0.81720 vs z 0.81723 | 0.79508 / 0.79525 vs 0.79566 | negative |
| 9 | More training events (learning curve, 1→4 pseudo-folds, same LGB, cut 25) | 0.8142 / 0.8158 / 0.8162 / 0.8160 | – | saturated after ~2–3 folds → all-events rebuild est. ≤ +0.001 |
| 10 | From-scratch softmax+outside-option latent model (torch, all events) | alone 0.58–0.60 | – | useless: sparse (50k items, 53k users, 687k events), overfits |

## 3. Simulator facts learned
- **Each event has at most ONE click** (0.366 no-click, 0.634 one click, nothing else) → single-choice softmax with an outside option; v12's binary logits sum to 0.98 per slate vs 0.63 observed (calibrated within-event ranking is what matters).
- P(no click) falls with slate logsumexp but is more consistent with a taste-dependent outside option than a constant one: constant-outside prediction overshoots P0 (e.g. top decile pred 0.31 vs obs 0.17).
- Event "any click" AUC: lse 0.668 → +z-context LGB 0.671 → + user propensity 0.672 (C25); C23 0.644→0.649→0.652. Oracle event label would give 0.894 (C25). So the 0.83 gap would have to come from event-level prediction ≈0.74+ (headroom sim: event AUC 0.76 ⇒ +0.0075), which is not reachable from user propensity (split-half user corr only 0.39, event click rate flat in hour/gap) nor from slate/exposure aggregates.
- Transductive univariates ≤0.53 within-event AUC; stationary data; exposure unrelated to click; post-cutoff clicks in histories ≈0.500.

## 4. Which lever most likely explains 0.83?
None of the five measured levers can: sum of all positive finds ≈ +0.0005–0.001. The gap (if real) is most plausibly better **item–user taste modelling** (a better preference model improves both within-event ranking and event-level lse), not data volume, propensity, or eval aggregates. My own end-to-end latent softmax failed because of sparsity, so any win there needs the strong CF-style features v12 already has (+ ensembling of many diverse models). Cheap remaining idea within v12's paradigm: more diverse seeds/learners (blend gains in v12 notes were ~+0.001 each).

## 5. Value of transductive features (information only, for Eric)
Measured cost of the eval-aggregate ban: **≈ +0.0001 (CI spans 0) on C25/C23** — not worth lifting the ban. v14 README written in /workspace/huawei/v14_transductive/, submission NOT built (nothing gained).

## 6. Code paths (all in /workspace/huawei/dw_lb/)
lib.py (z12, load_cut, oof, boot_delta, report) · feat_prop.py → cache_prop_c{25,23}.npy (propensity, ready to hand to v12 as new feature family; leak-safe: snapshot ts<C + own history) · feat_tx.py (transductive, `build_tx`) · run_tx.py / run_prop.py / run_p4.py (tests) · p4_softmax.py (torch single-choice model) · lc/lc.sh (learning curve) · results*.jsonl · logs.
