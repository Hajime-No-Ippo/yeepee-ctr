# v21seq STATUS (auto-generated 2026-11-01 05:22 Europe/Dublin)
ctl6 = noseq listwise MLP (v21list recipe + 6 epochs + BCE alpha .3); big = v21seq_big (trf history encoder, DIN attn, slate attn, 6ep, alpha .3, aug .3, frozen emb). base = .4z+.6L (z=v19 stack at C25, dwbig at C23; L=v21list s0 logit).

## var A C23
z 0.79679 | v21list A (1 seeds) 0.79799 | base .4z+.6L 0.79865

| model | n | seed-avg AUC at k=1/2/4/8/all | last-seed-singles | blend .4z+.6M (all) | delta vs base |
|---|---|---|---|---|---|
| ctl6 | 8 | 0.79824 / 0.79845 / 0.79857 / 0.79862 | 0.79824 0.79836 0.79845 0.79847 0.79830 0.79841 0.79848 0.79849 | 0.79903 | +0.00038 |
| v21seq_big | 3 | 0.79828 / 0.79846 / 0.79851 | 0.79828 0.79837 0.79835 | 0.79902 | +0.00037 |
| ALL | 11 | 0.79864 | | 0.79905 | +0.00040 |

## var A C25
z 0.81829 | v21list A (1 seeds) 0.81857 | base .4z+.6L 0.81894

| model | n | seed-avg AUC at k=1/2/4/8/all | last-seed-singles | blend .4z+.6M (all) | delta vs base |
|---|---|---|---|---|---|
| ctl6 | 8 | 0.81890 / 0.81900 / 0.81899 / 0.81904 | 0.81890 0.81893 0.81880 0.81879 0.81900 0.81888 0.81900 0.81891 | 0.81921 | +0.00028 |
| v21seq_big | 3 | 0.81904 / 0.81907 / 0.81916 | 0.81904 0.81892 0.81905 | 0.81930 | +0.00036 |
| ALL | 11 | 0.81910 | | 0.81925 | +0.00031 |

## var A C26

| model | n | seed-avg AUC at k=1/2/4/8/all | last-seed-singles | blend .4z+.6M (all) | delta vs base |
|---|---|---|---|---|---|
| ctl6 | 8 | 0.82783 / 0.82787 / 0.82793 / 0.82797 | 0.82783 0.82775 0.82783 0.82785 0.82793 0.82782 0.82787 0.82779 | - | - |
| v21seq_big | 2 | 0.82779 / 0.82791 | 0.82779 0.82784 | - | - |
| ALL | 10 | 0.82798 | | - | - |

## var B C25 (TRANSDUCTIVE)
z 0.81830 | v21list B (3 seeds) 0.81856 | base .4z+.6L 0.81892

| model | n | seed-avg AUC at k=1/2/4/8/all | last-seed-singles | blend .4z+.6M (all) | delta vs base |
|---|---|---|---|---|---|
| ctl6 | 3 | 0.81893 / 0.81897 / 0.81899 | 0.81893 0.81885 0.81886 | 0.81920 | +0.00028 |

## C27 (eval) prediction files
- A ctl6: 8 seeds
- A v21seq_big: 3 seeds
- B ctl6: 8 seeds

## Other results (C23, A)
- slate correction (hlen/staleness logistic offset, cross-fit C25<->C23): -0.00005/-0.00006 (null)
- GBDT lambdarank 0.79108 (500r) standalone; xendcg .79127; binary+slate-relative .79687 (vs binary .79679); blends with base: <= +0.00003
- seq with learned item embeddings overfits (.7937->.7439); frozen embeddings: no gain over noseq in a like-for-like control (C23 big .79828 vs ctl .79824; C25 .81904 vs .81890)
- blend weight w in (1-w)z+wM: optimum ~.7 (C23 .79907, C25 .81924), w=.6 within .00003

## Probes in out/ (sha256; ctl8 already copied to the Mac - do not touch)
### out/SHA256SUMS_all.txt
```
0292ac62461c3b9b7e40d8438322285d9797595494c5fce9e0c251290f7fc84f  submission_v21seq_A_trainonly_all.csv.zip
c0565700cb03963fa4821c7cf6c6cd3c0e061bc9d288e2ac915c09941b2ef1c8  probe_v21seq_A_trainonly_p33_w80_all.csv.zip
34c3eed57f18775323ce5d5ca76825e2ec463a7462aa92938c11cc99dc9d56c7  probe_ebnew_v21seq_A_w90_10_all.csv.zip
4f44b197e5c42d9cc2b284e228113363bec798950616c5f450b19abf30c8748a  probe_ebnew_v21seq_A_w85_15_all.csv.zip
```
### out/SHA256SUMS_ctl8.txt
```
55a20fa8676ae4983c21db6b97af2237fa0707e71e0069934fe166ccb20ae0e8  submission_v21seq_A_trainonly_ctl8.csv.zip
454018cec7bd661458f19f804fd7e90581c57cc1d3524b68c06fe933cd9be730  probe_v21seq_A_trainonly_p33_w80_ctl8.csv.zip
4d52c8226584201ccadea66d097288e8a6e0b48ee623d6530406db3a3434211f  probe_ebnew_v21seq_A_w90_10_ctl8.csv.zip
220cd4109ebdee1bedb6f05544b7c20f59912b5e5368927aabbf37f7b190f162  probe_ebnew_v21seq_A_w85_15_ctl8.csv.zip
```
### out/SHA256SUMS_ctl8B.txt
```
43fa439ecad711983e5ecea220184a39aaa5c7db680220ecfd109f3faa9fc971  submission_v21seq_B_TRANSDUCTIVE_ctl8B.csv.zip
368616f401e35b3a73b99a06982dbc67a3452e37b1ed747917fc0b8412d1ad7e  probe_v21seq_B_TRANSDUCTIVE_p33_w80_ctl8B.csv.zip
5ff6a3e84df728205efaa6f04233c18f5109eece4121f54792c7d84945646920  probe_ebnew_v21seq_B_w90_10_ctl8B.csv.zip
efb6e13880fc44e07c0f83e22bd8d907ec02a23a253dae842322d9643038a9fc  probe_ebnew_v21seq_B_w85_15_ctl8B.csv.zip
```
### out/SHA256SUMS_pre3.txt
```
920d19b8d8a23a41bb0387b314a453b141431ab037de45e463513ca1deb2f55e  submission_v21seq_A_trainonly_pre3.csv.zip
f3912765b39093882e09b9a78bba6ca5f2ef8ed69018978944982f0993cd4887  probe_v21seq_A_trainonly_p33_w80_pre3.csv.zip
e5c544b05870bc2cea6f504bd1a9d961148feb64ee5d9172b9c0ea0dbeaa30c4  probe_ebnew_v21seq_A_w90_10_pre3.csv.zip
048fedaaf3e537b94bce51b888ee91507ab7b7b6f137b362df55b5a518369ba3  probe_ebnew_v21seq_A_w85_15_pre3.csv.zip
```
