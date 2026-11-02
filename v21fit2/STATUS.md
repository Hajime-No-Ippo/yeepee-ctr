# v21fit2 STATUS (started 07:40 Nov 2, Dublin) - refit E0/v8/v5 jointly with graph
- fit2.py: 12-comp binormal (10 base + G7 + G8), obs Q2,R1,S1,T1,G1,G2,G3; SIG .00006 tau .00005; posterior saved post2.npz. opt2.py: re-optimises (c,g,q,v8,v5,w8) vs G3. local.py: C23/C25 D-share x graph weight check.
- No uploads, no pushes, no file built yet. Cores 4-7.
## 08:05 RESULT: bar NOT met -> NO FILE built, nothing uploaded/pushed.
Main fit (SIG .00006,tau .00005; logs fit2.log opt2.log): ESS 2509; G7 AUC .8057 (5/95 .8033-.8082), G8 .8084 (.8056-.8113), dloc .0027 (-.0011..+.0064) (was .0049 assumed); aD .8182, ours .8194, v5 .7923, v8r .8147.
Residuals obs-fit (e-5): Q2 +4.4 R1 +0.6 S1 0 T1 -1.5 G1 -1.7 G2 -1.6 G3 +0.1.
Re-opt vs G3 (unshrunk) (c,g,q,v8,v5,w8)=(.128,.186,.484,-.126,-.042,.060): +4.8e-5, 5/95 -5.2e-5..+14.7e-5, P>0 .79, P>5e-5 .48. 50% shrink +3.6e-5 P .90. Graph weight alone around S1 flat (w8 .035-.05 best, +0..+.5e-5). SIG .00004 variant (opt2_s4.log): optimum (.142,.173,.73,-.109,-.039,.061) +3.9e-5 P .80 -> q not identified (.48 vs .73).
Local (local.py): with graph, best D share q drops to ~.33-.5 (C23,C25); q .67 costs ~7e-5 (C25) / 25e-5 (C23) at w .2.
## 08:30 Three directions vs G3 (dirs.py/dirs.log, local2.py/local2.log; posterior post2.npz): NO candidate meets bar -> NO FILE, no upload/push.
D1 graph mix (S1+w7*G7+w8*G8): best tot .04 all-7s +1.6e-5 [-10.6,+13.4]e-5 P .60 P>5e-5 .32; .04 half/half +1.2e-5 P .63. tot>=.07 negative. Local: G8 alone > mix > G7 alone on C23/C25 (mix hurts locally).
D2 trade ours total for graph: plain (S1 E0/v8/v5) all negative (-0.4e-5 .. -24e-5); lightly refit g .10 (c .124,v8 -.070,v5 -.039,w8 .046) +1.0e-5 [-4.4,+6.7]e-5 P .62 P>5e-5 .12.
D3 rank-avg with Q2/R1 (m .1-.5) +/- extra G8: all <=0 (best G3+R1 m.1 +G8 .03 +0.2e-5 P .53); Q2 mixes -0.6..-7e-5.
