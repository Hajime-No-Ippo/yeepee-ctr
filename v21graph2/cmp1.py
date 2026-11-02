import sys; sys.dont_write_bytecode=True
from ana import *
c=25
line(c,'u1 d64L3 s0 350',sum3(avg('u1',c,[0],350)))
for st in (250,350,500): line(c,f'u3 d128L3 s0 {st}',sum3(avg('u3',c,[0],st)))
