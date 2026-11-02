import sys; sys.dont_write_bytecode=True
from ana import *
c=25
line(c,'u1 d64L3 3seeds 350',sum3(avg('u1',c,[0,1,2],350)))
for st in (250,350,500): line(c,f'u4 d64L2 3seeds {st}',sum3(avg('u4',c,[0,1,2],st)))
