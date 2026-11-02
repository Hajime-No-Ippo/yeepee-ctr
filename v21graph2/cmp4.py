import sys; sys.dont_write_bytecode=True
from ana import *
c=25
line(c,'u1 L3 s0 350',sum3(avg('u1',c,[0],350)))
line(c,'u6 L4 s0 350',sum3(avg('u6',c,[0],350)))
line(c,'u7 L3 tau5 s0 350',sum3(avg('u7',c,[0],350)))
for st in (250,350,500): line(c,f'u8 L5 s0 {st}',sum3(avg('u8',c,[0],st)))
line(c,'u6 L4 3s 350',sum3(avg('u6',c,[0,1,2],350)))
line(c,'u6 L4 3s 500',sum3(avg('u6',c,[0,1,2],500)))
line(c,'u1 L3 3s 350',sum3(avg('u1',c,[0,1,2],350)))
