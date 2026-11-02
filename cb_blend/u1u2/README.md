# u1u2 = v21graph LightGCN configs (u1, u1b; u2 not scoreable) -- local test, Code Buddy 2026-11-02 ~03:10 Dublin
Read-only on v21graph. Files: gfeat_chunk.py (chunked identical re-impl of gfeat.feats; max diff 1e-6), build_members.py (per-seed feat caches, rowid/slate checks -> align.json),
test_graph.py/test_graph.log (reproduction, held-out, seed sensitivity), heldout.json, graph_members.npz (frozen G23/G25/G26), graph_member.py.
blend_oof.py: added member 'graphU1' (backup blend_oof.py.bak_pregraph = state before my edit). Existing 'graph' (interim frozen copy, C23/25 only) left untouched.
