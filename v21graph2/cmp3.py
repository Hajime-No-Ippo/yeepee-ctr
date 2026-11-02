import sys; sys.dont_write_bytecode=True
from ana import *
c=25; sd=[0,1]
line(c,'u1 d64L3 2s 350',sum3(avg('u1',c,sd,350)))
for st in (250,350,500): line(c,f'u6 d64L4 2s {st}',sum3(avg('u6',c,sd,st)))
