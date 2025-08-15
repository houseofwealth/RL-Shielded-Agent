from z3       import *

pos, vel  = Reals('pos vel')
force = Real('force')

s = Solver()

f = Not(Not(306 <=
        2000/3*pos +
        2000*vel +
        15/2*(pos + vel)*(pos + vel) +
        -45/8*
        (pos + vel)*
        (pos + vel)*
        (pos + vel)*
        (pos + vel) +
        2*force +
        15*pos*pos +
        -45/4*pos*pos*pos*pos))
h = [Not(9/20 <= pos), 
    Not(9/20 <= pos + vel), 
    Not(908/3 <=
    2000/3*pos +
    15/2*pos*pos +
    -45/8*pos*pos*pos*pos +
    4000/3*vel), 
    Not(154 <=
    1000/3*pos +
    15/2*pos*pos +
    -45/8*pos*pos*pos*pos +
    1000*vel +
    15/4*(pos + vel)*(pos + vel) +
    -45/16*(pos + vel)*(pos + vel)*(pos + vel)*(pos + vel)), force >= -1, force <= 1, 
    Not(181/400 <=
    pos +
    2*vel +
    3/2000*force +
    9/800*pos*pos +
    -27/3200*pos*pos*pos*pos), 
    Not(313/2 <=
    1000/3*pos +
    4000/3*vel +
    15/2*(pos + vel)*(pos + vel) +
    -45/8*(pos + vel)*(pos + vel)*(pos + vel)*(pos + vel) +
    3/2*force +
    45/4*pos*pos +
    -135/16*pos*pos*pos*pos +
    15/4*
    (-1/400 +
     pos +
     2*vel +
     3/2000*force +
     9/800*pos*pos +
     -27/3200*pos*pos*pos*pos)*
    (-1/400 +
     pos +
     2*vel +
     3/2000*force +
     9/800*pos*pos +
     -27/3200*pos*pos*pos*pos) +
    -45/16*
    (-1/400 +
     pos +
     2*vel +
     3/2000*force +
     9/800*pos*pos +
     -27/3200*pos*pos*pos*pos)*
    (-1/400 +
     pos +
     2*vel +
     3/2000*force +
     9/800*pos*pos +
     -27/3200*pos*pos*pos*pos)*
    (-1/400 +
     pos +
     2*vel +
     3/2000*force +
     9/800*pos*pos +
     -27/3200*pos*pos*pos*pos)*
    (-1/400 +
     pos +
     2*vel +
     3/2000*force +
     9/800*pos*pos +
     -27/3200*pos*pos*pos*pos))]
s.add(f)
s.add(h)
print('checking')
s.check()

