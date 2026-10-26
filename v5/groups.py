import json,sys,re
sel=[s[4] for s in json.load(open('out/SEL.json'))]
G={'A1':'^(%s)(_z|_rk)?$'%'|'.join(sel) if sel else '^$NONE',
   'A2':'^x_(icon05_k32|icon1_k32|icos_k32|icon025_k64)(_z|_rk)?$',
   'A3':'^x_con05_k32_(norm|dot|dmax)$|^x_n(clk_cl|skip_p)$'+('|^%s_(norm|dot)$'%sel[0] if sel else '')}
def drop_except(keep): return '|'.join(v for k,v in G.items() if k not in keep) or '^$NONE'
if __name__=='__main__':
    keep=sys.argv[1].split(',') if len(sys.argv)>1 and sys.argv[1] else []
    print(drop_except(keep))
