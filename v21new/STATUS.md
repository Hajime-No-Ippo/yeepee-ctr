# v21new STATUS (started 2026-11-02 ~12:40 Dublin)
Goal: genuinely new train-only model; bar +2e-4 on C23&C25 over base(.4z+.6M)+G8. Cores 4-7 (taskset), RSS<6-7GB, venv_v17. No uploads/pushes.
Data tables: /workspace/huawei/v21graph2/out/tab_{val23,val25,val26,eval}.npz (user,item,ts,hist newest-first pad 50000,hts,y), slates.npz (train events), g_k.npz edges. G8 fold preds v21graph2/out/graph2s3b_f6_c{23,25,26}.npy.
## Log
- 13:10 t_rhythm: user time-of-day rhythm adds nothing to event-click AUC (dead). Built ae.py (history-only ID-free next-click autoencoder, sampled softmax, k5 test). Untied d256: overfit (.645). Tied d64 lr2e-3 6ep: .707 still rising. Running t4 (d64 lr5e-3 20ep) and t5 (d128 lr4e-3 20ep) on k5.
- 13:45 AE history-only (tied d64/d128): plateau .71 val25 then overfit -> DEAD (info gap: lacks user's full click union). Next: ae2.py = Mult-DAE style set autoencoder: input = user's all cutoff clicks minus held-out target (+dropout), at inference input = union(user graph clicks, event hist).
- 13:30 ae2 (set DAE, tied d64/d128, k5): plateau .703-.704 val25 ep4 then overfit -> DEAD. Next: spectral GF-CF/PureSVD fold-in (svd.py), last family.
- 13:50 svd.py spectral fold-in (PureSVD/GF-CF style, K32-512, gamma 0-3, a/b norm): best val25 .727 (K128, gam0), K256 .707, K512 .66 -> DEAD. All three new families cap .70-.73 vs G8 .80. STOP; no gain.py run, no file built, no fit extension. Nothing uploaded/pushed.
