# model of water tank taken from "Safe Reinforcement Learning via Shielding". This is simplified version of the full problem that only implements open /\ Xclosed => XXclosed. Included is a "hacky" solution to the problem that relised on past time but this seemed difficult to generalize

from z3       import *

Time = IntSort() 

#  state  variables

t, tX = Consts('t tX', Time)
T1, T2 =     Consts('T1 T2', BoolSort())
T1X, T2X =  Consts('T1X T2X', BoolSort())

tank, tankX = Reals('tank tankX')
# wasOpen, wasOpenX = Ints('wasOpen wasOpenX')
wasOpen, wasOpenX = Bools('wasOpen wasOpenX')
wasClosed, wasClosedX = Bools('wasClosed wasClosedX')
globals     = []
state       = [t, T1, T2, tank,  wasOpen, wasClosed]
stateX      = [tX,T1X,T2X, tankX, wasOpenX, wasClosedX]
subst       = list(zip(state,stateX))
stateInv    = True 

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
              'invariant'    : And(t >= 0)} #, T1 >= 0, T2 >= 0)}

#originally had one oombined action with Or of the conditions but was difficult to interpret resul
updateAction = {'name'       : "updateOpen",
                'actionPred' : And(open,      
                                      tankX == tank + inn - out, 
                                      tX == t+1,
                                      T1X),
                                      # wasOpenX),  
                                      # Not(wasOpenX))),
                'envVar'     : [inn, out],
                'envPred'    : And(1 <= inn, inn <= 2, 0 <= out, out <= 1),
                'controlVar' : [open],
                'controlPred': Implies(T2, Not(open)),
                'precNode'   : singleton,
                'postNode'   : singleton
                }

updateAction2 = {'name'       : "updateClosed",
                'actionPred' : And(Not(open), 
                                      tankX == tank - out,
                                      tX == t+1,
                                      T1X == T1),
                                      # Not(wasOpenX))),
                'envVar'     : [inn, out],
                'envPred'    : And(1 <= inn, inn <= 2, 0 <= out, out <= 1),
                'controlVar' : [open],
                'controlPred': Implies(T2, Not(open)),
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
                Implies(T1, T1X==False),
                Implies(T2, T2X==False),
                Implies(And(T1, Not(open)), T2X),]
                # these way of doing the constraints worked but was hacky
                #open /\ X closed --> XX closed
                # Implies(And(wasOpen, Not(open)), wasClosedX),     #.X open /\ closed --> X wC
                # Implies(wasClosed, Not(open)),                    #stay closed one more step
                # Implies(open, Not(wasClosedX))]                   #once opened, turn off wC


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
