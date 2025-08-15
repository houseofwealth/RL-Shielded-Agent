""" # model of water tank taken from "Safe Reinforcement Learning via Shielding """
from z3       import *

'''state  variables'''
pos, posX, vel, velX  = Reals('pos posX vel velX')
state                 = [pos, vel] 
stateX                = [posX,velX]
subst                 = list(zip(state,stateX))
stateInv              = True 

'''control var'''
force = Real('force')

'''misc'''
POWER = 0.0015
K = 0.0025
GOAL = 0.45
x = Real('x')

PI = 3.142
SQRT_2 = 1.414
def cosF(x):
   #  return 1 - x*x/2 + x*x*x*x/24

   if x == -3*PI/2 or x == -PI/2 or x == PI/2: return 0
   if x == -PI or x == 0 or x == PI: return 1
   if -3*PI/2 < x and x < -PI:
      f = 0
   else:
      f = 1
   return f
   # else if -P! 
   # if -PI/2 < x and x < 0:
   #    f = 1
   # else if 

# Nodes
singleton  = {'name'         : 'singleton',
              'globals'      : [],
              'vars'         : state,
              'postVars'     : stateX,    
              'subst'        : subst,    
              'invariant'    : True}  

action = {'name'       : "action",
          'actionPred' : And( #moved -> U: open,      
                              # ForAll(x, cosF(x) == 1 - x*x/2 + x*x*x*x/24),      #req NRA
                              # velX == vel + force*POWER - K*cosF(3*pos),
                              Implies(Or(3*pos == -3*PI/2, 3*pos == -PI/2, 3*pos == PI/2),
                                      velX == vel + force*POWER),
                              Implies(Or(3*pos == -PI, 3*pos == PI),
                                      velX == vel + force*POWER - -K),
                              Implies(pos == 0,
                                      velX == vel + force*POWER - K),
                              Implies(And(-3*PI/2 < 3*pos, 3*pos < -PI), 
                                      velX == vel + force*POWER - K*cos1(3*pos))
                              # Implies(And(-0.07 <= tempX, tempX <= 0.07), velX == tempX)
                              # Implies(Not(And(-0.07 <= tempX, tempX <= 0.07)), velX == vel)
                              posX == pos + vel), 
          'invariant'  :  #True,
                          And(#may need to move these into the actionPred
                              -1.2 <= pos, pos <= 0.6,
                              -0.07 <= vel, vel <= 0.07),
          'controlVar' : [force],
          'controlPred': And(-1 <= force, force <= 1),     
          'precNode'   : singleton,
          'postNode'   : singleton
                }

model       = {'name'        : 'Continuous Mountain Car',
               'initNode'    : singleton,
               'nodes'       : [singleton],
               'transitions' : [action]
              }

""" # Required Properties """
safetyProps = [ pos < GOAL]

def fnOr(xs): return any(xs)
def fnAnd(xs): return all(xs)
def fnNot(x): return not x

# def solnExists(tank_level, t):
#   return fnAnd( tank_level >= 0,
#                 tank_level <= 100,
#                 t >= 0,
#                 t <= 3,
#                 Or(Not(t == 3), 1 <= tank),
#                 Or(Not(t == 2), 2 <= tank))

# def OK( open_valve, tank_level, t):
#     return OKProp1(open_valve, tank_level, t) #and OKProp2(open_valve, tank_level)
    
# #if an action choice fails, agent needs to retry with the negated choce, otherwise the t's wont be corectly updated
# def OKProp1(open_valve, tank_level, t):
#   #the first 3 conds below are the starting constraint for control pred above
#   if open_valve:
#     res = fnAnd(Not(t == 2), Not(t == 3), tank_level <= 98)
#     if res:     #don't modify the t if this choice fails as it will retry
#       match t:
#         case 0: t=1
#         case 1: t=0
#         case 2: t=3
#         case 3: t=0
#   else:
#     res = fnOr(t == 2, t == 3, fnAnd(1 <= tank, t == 0), 3 <= tank)
#     if res:
#       match t:
#         case 0: t=0
#         case 1: t=2
#         case 2: t=3
#         case 3: t=0
#   return (res,t)


"""
produces:
stateInv
 And(tank >= 0,
    tank <= 100,
    t >= 0,
    t <= 3,
    Or(Not(t == 3), 1 <= tank),
    Or(Not(t == 2), 2 <= tank))
newInvDelta:
 []
no change to invariant on singleton

----------------
Final Model - fixpoint at iteration 4
Node: singleton
  Vars: [t, T0, T1, T2, T3, tank]
  Node Invariant: True
Transition: openAction
   0   open
   1   Not(t == 2)
   2   Not(t == 3)
   3   tank <= 98
   4   inn >= 1
   5   inn <= 2
   6   out >= 0
   7   out <= 1
   8   tankX == tank + inn + -1*out
   9   Or(Not(open), tX == 1, Not(t == 0))
   10   Or(open, tX == 0, Not(t == 0))
   11   Or(open, tX == 2, Not(t == 1))
   12   Or(Not(open), tX == 0, Not(t == 1))
   13   Or(Not(t == 2), tX == 3)
   14   Or(Not(t == 3), tX == 0)
Transition: closeAction
   0   Not(open)
   1   Or(t == 2, t == 3, And(1 <= tank, t == 0), 3 <= tank)
   2   inn >= 1
   3   inn <= 2
   4   out >= 0
   5   out <= 1
   6   tankX == tank + -1*out
   7   Or(Not(open), tX == 1, Not(t == 0))
   8   Or(open, tX == 0, Not(t == 0))
   9   Or(open, tX == 2, Not(t == 1))
   10   Or(Not(open), tX == 0, Not(t == 1))
   11   Or(Not(t == 2), tX == 3)
   12   Or(Not(t == 3), tX == 0)


#safetyProps= [out >= 0, out <= 4,
#              Not(tank + -1*out <= -1), Not(-1*tank + out <= -17),
#              -1*tank + 3*out <= 3,   tank + -3*out <= 11,
#              tank + -4*out <= 10,  -1*tank + 4*out <= 6]
#
#      out + u >= 0
#      out + u <= 4
#      Not(tank + -1*u + -2*out <= -1)
#      Not(-1*tank + u + 2*out <= -13)
#      Not(tank + -3*u + -4*out <= -4)
#      Not(-1*tank + 3*u + 4*out <= -8)
#      Not(-1*tank + 4*u + 5*out <= -7)
#      Not(tank + -4*u + -5*out <= -7)

# equivalent solution:
#         Not(tank + -4*u + -5*out <= -7),  Not(-1*tank + 4*u + 5*out <= -7),
#         -1 <= tank + -2*u + -3*out,       tank + -2*u + -3*out <= 9
"""