import sys; sys.dont_write_bytecode=True
from ana import *
for tag in ('u12','u13'):
  for st in (500,650,800,900): line(25,f'{tag} s0 {st}',sum3(avg(tag,25,[0],st)))
