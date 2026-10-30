exec(open('bl.py').read().split('for w in (90,80,70)')[0].split('# slate mean of ranks')[0])
import zipfile, hashlib, os
exec('def zipit'+open('bl.py').read().split('def zipit')[1].split('for w in (90,80,70)')[0])
for w in (60,50,40): mk(f'probe_eb_v21list_B_w{w}',w/100*r['eb']+(1-w/100)*r['ours'])
