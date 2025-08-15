# model of water tank taken from "Safe Reinforcement Learning via Shielding", handcrafted means that these constraints were done by inspection and not by any systematic way of deriving them from the property spec. But I'm not sure they actually work

from z3       import *

Time = IntSort() 

#  state  variables
t, tX = Consts('t tX', Time)
T1, T2, T3 =     Consts('T1 T2 T3', BoolSort())
T1X, T2X, T3X =  Consts('T1X T2X T3X', BoolSort())

tank, tankX =           Reals('tank tankX')
wasOpen, wasOpenX =     Bools('wasOpen wasOpenX')
wasClosed, wasClosedX = Bools('wasClosed wasClosedX')
globals     = []
state       = [t, T1, T2, T3,  tank,  wasOpen,  wasClosed]
stateX      = [tX,T1X,T2X,T3X, tankX, wasOpenX, wasClosedX]
subst       = list(zip(state,stateX))
stateInv    = True 

# Env variables
inn = Real('inn')
out = Real('out')

# Control var
# Alt: open = Int('open')
open = Bool('open')

# Nodes
singleton  = {'name'         : 'singleton',
              'globals'      : globals,
              'vars'         : state,
              'postVars'     : stateX,    
              'subst'        : subst,    
              # alt: make open an int: 'invariant'    : And(t >= 0, T1 >= 0, 0 <= open, open <=1)}
              'invariant'    : And(t >= 0)} #, T1 >= 0, T2 >= 0)}

updateAction = {'name'       : "updateOpen",
                'actionPred' : And( open,      
                                    tankX == tank + inn - out, 
                                    tX == t+1,
                                    T1X),
                'envVar'     : [inn, out],
                'envPred'    : And(1 <= inn, inn <= 2, 0 <= out, out <= 1),
                'controlVar' : [open],
                'controlPred': And(Implies(T2, Not(open)), Implies(T3, Not(open))),
                'precNode'   : singleton,
                'postNode'   : singleton
                }

updateAction2 = {'name'       : "updateClosed",
                'actionPred' : And( Not(open), 
                                    tankX == tank - out,
                                    tX == t+1,
                                    T1X == T1),
                'envVar'     : [inn, out],
                'envPred'    : And(1 <= inn, inn <= 2, 0 <= out, out <= 1),
                'controlVar' : [open],
                'controlPred': And(Implies(T2, Not(open)), Implies(T3, Not(open))),
                'precNode'   : singleton,
                'postNode'   : singleton
                }

model       = {'name'        : 'Water Tank',
               'initNode'    : singleton,
               'nodes'       : [singleton],
               'transitions' : [updateAction, updateAction2]
              }

# Required Properties
TANK_CAPACITY = 100
initProps   = [tank == 0, out == 0]
safetyProps = [ 0 <= tank, tank <= TANK_CAPACITY,
                # turn off flags once next has passed'''
                Implies(T1, T1X==False),
                Implies(T2, T2X==False),
                Implies(T3, T3X==False),
                # track the first X.. which comes when open is true'''
                Implies(open, T1X),
                Implies(Not(open), T1X==T1),                            #frame assump
                # track that open /\ Xclosed requires XX..'''
                Implies(And(T1, Not(open)), T2X),
                Implies(And(Not(And(T1, Not(open))), Not(T2)), T2X==T2),#frame assmp
                # track that open /\ Xclosed requires XXX..'''
                Implies(T2, T3X),
                Implies(And(Not(T2), Not(T3)), T3X==T3)                 #frame assump
              ]
'''
Final Model - fixpoint at iteration 2
Node: singleton
  Vars: [t, T1, T2, T3, tank, wasOpen, wasClosed]
  Node Invariant: And(t >= 0)
Transition: updateOpen
   0   Or(Not(T2), Not(open))
   1   Or(Not(T3), Not(open))
   2   Or(T1, Not(open), tank <= 98)
   3   inn >= 1
   4   inn <= 2
   5   out >= 0
   6   out <= 1
   7   tankX == tank + inn + -1*out
   8   tX == 1 + t
   9   Not(T1)
   10   Not(T2X)
   11   Or(Not(T3), Not(T3X))
   12   T1X
   13   open
   14   Or(T3X, Not(T2))
   15   Or(T2, T3, Not(T3X))
Transition: updateClosed
   0   Or(Not(T2), Not(open))
   1   Or(Not(T3), Not(open))
   2   Or(T1, open, And(T2, T3), 1 <= tank)
   3   inn >= 1
   4   inn <= 2
   5   out >= 0
   6   out <= 1
   7   tankX == tank + -1*out
   8   tX == 1 + t
   9   Not(T1X)
   10   Not(T2X)
   11   Or(Not(T3), Not(T3X))
   12   Not(open)
   13   T1X == T1
   14   Or(T3X, Not(T2))
   15   Or(T2, T3, Not(T3X))
'''
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
