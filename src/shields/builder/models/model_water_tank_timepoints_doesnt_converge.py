# model of water tank taken from "Safe Reinforcement Learning via Shielding"

from z3       import *

Time = IntSort() 

#  state  variables
# t,  T1,  T2 =        Consts('t  T1  T2',  Time)
t, T1, T2 = Consts('t T1 T2', Time)
tX, T1X, T2X =       Consts('tX T1X T2X', Time)
tank, tankX = Reals('tank tankX')
# wasOpen, wasOpenX = Ints('wasOpen wasOpenX')
wasOpen, wasOpenX = Bools('wasOpen wasOpenX')
wasClosed, wasClosedX = Bools('wasClosed wasClosedX')
globals     = []
state       = [t, T1, T2, tank,  wasOpen, wasClosed]
stateX      = [tX,T1X,T2X, tankX, wasOpenX, wasClosedX]
subst       = list(zip(state,stateX))
stateInv    = True #And(0 <= tank)      #initial state inv, isn't this subsumbed by props?

# Env variables
inn = Real('inn')
out = Real('out')

# Control var
# open = Int('open')
open = Bool('open')

# Nodes
singleton  = {'name'         : 'singleton',
              'globals'      : globals,
              'vars'         : state,
              'postVars'     : stateX,    # optional: for self transitions
              'subst'        : subst,     # optional: for self transitions
              # alt: make open an int: 'invariant'    : And(t >= 0, T1 >= 0, 0 <= open, open <=1)}
            #   'invariant'    : And(t >= 0, T1 >= 0, T2 >= 0)}
              'invariant'    : And(t >= 0)}#, T1 >= 0, T2 >= 0)}

# seperating this action into seperate actions actions will it work? 
# Arcs/Transitions
# orig:
# updateAction = {'name'       : "update",
#                 'actionPred' : Or(And(open,      
#                                       tankX == tank + inn - out, 
#                                       tX == t+1),
#                                     #   T1X == t+1),
#                                       # wasOpenX),  
#                                   And(Not(open), 
#                                       tankX == tank - out,
#                                       tX == t+1)),
#                                     #   T1X == T1)),
#                                       # Not(wasOpenX))),
#                 'envVar'     : [inn, out],
#                 'envPred'    : And(1 <= inn, inn <= 2, 0 <= out, out <= 1),
#                 'controlVar' : [open],
#                 # 'controlPred': Implies(t==T2, Not(open)),
#                 'controlPred': True,
#                 'precNode'   : singleton,
#                 'postNode'   : singleton
#                 }
updateAction = {'name'       : "updateOpen",
                'actionPred' : And(open,      
                                      tankX == tank + inn - out, 
                                      tX == t+1),
                                    #   T1X == t+1),
                                      # wasOpenX),  
                'envVar'     : [inn, out],
                'envPred'    : And(1 <= inn, inn <= 2, 0 <= out, out <= 1),
                'controlVar' : [open],
                # 'controlPred': Implies(t==T2, Not(open)),
                'controlPred': True,
                'precNode'   : singleton,
                'postNode'   : singleton
                }

updateAction2 = {'name'       : "updateClosed",
                'actionPred' : And(Not(open), 
                                      tankX == tank - out,
                                      tX == t+1),
                                    #   T1X == T1)),
                                      # Not(wasOpenX))),
                'envVar'     : [inn, out],
                'envPred'    : And(1 <= inn, inn <= 2, 0 <= out, out <= 1),
                'controlVar' : [open],
                # 'controlPred': Implies(t==T2, Not(open)),
                'controlPred': True,
                'precNode'   : singleton,
                'postNode'   : singleton
                }

model       = {'name'        : 'Water Tank',
               'initNode'    : singleton,
               'nodes'       : [singleton],
               'transitions' : [updateAction, updateAction2]
              }

# Required Properties
initProps   = [tank == 0, out == 0]
safetyProps = [ 0 <= tank, tank <= 5,
                #open /\ X closed --> XX closed
                # Implies(And(wasOpen, Not(open)), wasClosedX),     #.X open /\ closed --> X wC
                # Implies(wasClosed, Not(open)),                    #stay closed one more step
                # Implies(open, Not(wasClosedX))]                   #once opened, turn off wC
                # reinstante these two for timepoint based
                # Implies(And(t == T1, Not(open)), T2X == t+1),
                # Implies(Or(t != T1, open), T2X == T2)
              ]
'''
The basic problem (without the open closed temporal constraints) solution is: (ignore the "t")
(a) when done as one transition with an Or of the two arms:
Final Model - fixpoint at iteration 2
Node: singleton
  Vars: [t, T1, T2, tank, wasOpen, wasClosed]
  Node Invariant: And(t >= 0)
Transition: update
   0   Or(And(open, Not(1 <= tank)),
          And(tank <= 3, 1 <= tank),
          And(Not(open), 1 <= tank))
   1   inn >= 1
   2   inn <= 2
   3   out >= 0
   4   out <= 1
   5   Or(And(open, tankX == tank + inn - out, tX == t + 1),
          And(Not(open), tankX == tank - out, tX == t + 1))

(a) when done as 2 transitions:
Final Model - fixpoint at iteration 2
Node: singleton
  Vars: [t, T1, T2, tank, wasOpen, wasClosed]
  Node Invariant: And(t >= 0)
Transition: updateOpen
   0   Or(Not(open), tank <= 3)
   1   inn >= 1
   2   inn <= 2
   3   out >= 0
   4   out <= 1
   5   open
   6   tankX == tank + inn - out
   7   tX == t + 1
Transition: updateClosed
   0   Or(open, 1 <= tank)
   1   inn >= 1
   2   inn <= 2
   3   out >= 0
   4   out <= 1
   5   Not(open)
   6   tankX == tank - out
   7   tX == t + 1
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
