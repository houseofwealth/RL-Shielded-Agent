# model of water tank taken from "Safe Reinforcement Learning via Shielding"

from z3       import *

Time = IntSort() 

#  state  variables
t, tX = Consts('t tX', Time)
T0, T1, T2, T3 =     Consts('T0 T1 T2 T3', BoolSort())
T0X, T1X, T2X, T3X =  Consts('T0X T1X T2X T3X', BoolSort())

tank, tankX =           Reals('tank tankX')
wasOpen, wasOpenX =     Bools('wasOpen wasOpenX')
wasClosed, wasClosedX = Bools('wasClosed wasClosedX')
globals     = []
state       = [t, T0, T1, T2, T3,  tank,  wasOpen,  wasClosed]
stateX      = [tX,T0X,T1X,T2X,T3X, tankX, wasOpenX, wasClosedX]
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
              'invariant'    : True}  #And(t >= 0)} #, T1 >= 0, T2 >= 0)}

openAction = {'name'       : "openAction",
                'actionPred' : And( #moved -> U: open,      
                                    tankX == tank + inn - out), 
                'envVar'     : [inn, out],
                'envPred'    : And(1 <= inn, inn <= 2, 0 <= out, out <= 1),
                'invariant'  : True,
                'controlVar' : [open],
                'controlPred': And(open,Implies(T2, Not(open))),     #this is temp hack. Otherwise chooses U = ~open
                'precNode'   : singleton,
                'postNode'   : singleton
                }

closeAction = {'name'       : "closeAction",
                'actionPred' : And( #moved -> U: Not(open), 
                                    tankX == tank - out),
                'envVar'     : [inn, out],
                'envPred'    : And(1 <= inn, inn <= 2, 0 <= out, out <= 1),
                'invariant'  : True,
                'controlVar' : [open],
                'controlPred': And(Not(open),Implies(T2, Not(open))), #this is temp hack otherwise chooses U=open 
                'precNode'   : singleton,
                'postNode'   : singleton
                }

model       = {'name'        : 'Water Tank',
               'initNode'    : singleton,
               'nodes'       : [singleton],
               'transitions' : [openAction, closeAction]
              }

# Required Properties
TANK_CAPACITY = 4
initProps   = [tank == 0, out == 0]
safetyProps = [ 0 <= tank, tank <= TANK_CAPACITY,
                Implies(And(T0,open), And(T0X==False, T1X)),
                #soon as you see an open, and not in the middle of a processing, set 1st flag T1
                # Implies(And(Not(T3), Not(And(T0,open))), T0X==T0),  #frame
                Implies(And(Not(T2), Not(And(T0,open))), T0X==T0),  #frame
                # Implies(And(Not (T1), Not(T2), Not(T3), open), T1X),           
                # Implies(Not(T3), T0X==False),
                # Implies(Not(open), T1X), not sure why it needs to be open and not ~open
                # Implies(open, T1X),
                # turn off flags once next has passed
                # T1 is only set true by above constraint, rest of time its false, kins of frame rule
                Implies(And(T1, Not(open)), And(T1X==False, T2X)),
                # Implies(T1,And(T1X==False, T2X)),  
                Implies(And(Not(And(T1, Not(open))), Not(And(T0,open))), T1X == T1),
                # Implies(T2 ,T2X==False),  
                # Implies(Or(T1, T2, T3, Not(open)),T1X==False),    #frame + turnoff rules
                # Implies(T1, T2X),                                 #turn on
                # Implies(And(Not(T0),Not(And(T1, Not(open)))), And(T1X==T1)),  #frame rule
                # Implies(Not(And(T0,open)),T1X==False),  #frame rule

                # Implies(T2, And(T2X==False, T3X, Not(open))),     #turn off, turn on T3
                # Implies(And(Not(T1), Not(T2)), T2X==T2),          #frame rule
                Implies(T2, And(T2X==False, T0X )),     #turn off, turn on T3
                # moved --> control pred: Implies(T2, Not(open)),     #turn off, turn on T3
                Implies(And(Not(T1), Not(T2)), T2X==T2),          #frame rule
                # Implies(Not(T1), T2X==T2),                     
                
                # Implies(T3, And(T3X==False, Not(open))),          #turn off
                # Implies(And(Not(T2), Not(T3)), T3X==T3),          #frame rule
                # Implies(T3, And(T3X==False, T0X)),
                # Implies(T3, Not(open)),
                # Implies(Not(T2), T3X==T3),                       

                # And(#Not(And(T0,T1)), Not(And(T0,T2)), Not(And(T0,T3)), 
                #                      Not(And(T1,T2)), Not(And(T1,T3)),
                #                                       Not(And(T2,T3))),
                And(#Not(And(T0,T1)), Not(And(T0,T2)), Not(And(T0,T3)), 
                                     Not(And(T1,T2)), Not(And(T1,T0)),
                                                      Not(And(T2,T0))),
                Or(T0,T1,T2)
              ]

""" new ctrl Pred (residue of ctrlPred over Inv) openAction :
And(open, Or(T2, T3, tank <= 98))
guardDisjunction: Or(False, Exists(open, And(open, Or(T2, T3, tank <= 98))))
guardDisjunction after QE [Or(False, Or(tank <= 98, T2, T3))]
simplified guardDisjunction:
 Or(T2, T3, tank <= 98)
...
new ctrl Pred (residue of ctrlPred over Inv) closeAction :
And(Not(open), 1 <= tank)
guardDisjunction: Or(Or(T2, T3, tank <= 98),
   Exists(open, And(Not(open), 1 <= tank)))
guardDisjunction after QE [Or(Or(T2, T3, tank <= 98), 1 <= tank)]
simplified guardDisjunction:
 Or(T2, T3, tank <= 98, 1 <= tank)

- Completed arcs from node singleton. Now checking if invariant ne-eds updating (control preds may also get updated again)
stateInv
 And(tank >= 0,
    tank <= 100,
    Not(And(T1, T2)),
    Not(And(T1, T3)),
    Not(And(T2, T3)))
newInvDelta:
 []
no change to invariant on singleton """


# safetyProps = [ 0 <= tank, tank <= TANK_CAPACITY,
#                 Implies(T0, And(T0X==False, T1X)),
#                 Implies(And(Not(T0),Not(T3)), T0X==T0),
#                 # Implies(Not(open), T1X), not sure why it needs to be open and not ~open
#                 # Implies(open, T1X),
#                 # turn off flags once next has passed'''
#                 Implies(T1, And(T1X==False, T2X)),
#                 # Implies(And(Not(open),Not(T1)), And(T1X==T1)),  #frame rule
#                 Implies(And(Not(T0),Not(T1)), And(T1X==T1)),  #frame rule

#                 Implies(T2, And(T2X==False, T3X)),
#                 Implies(And(Not(T1), Not(T2)), And(T2X==T2)),
                
#                 Implies(T3, And(T3X==False, T0X)),
#                 Implies(And(Not(T2), Not(T3)), And(T3X==T3)),

#                 And(Not(And(T0,T1)), Not(And(T0,T2)), Not(And(T0,T3)), 
#                                      Not(And(T1,T2)), Not(And(T1,T3)),
#                                                       Not(And(T2,T3))),
#                     #not needed: Not(And(T3,T0)), Not(And(T3,T1)), Not(And(T3,T2))),
#                 # Implies(T1, And(Not(T2), Not(T3))),
#                 # Implies(T2, And(Not(T1), Not(T3))),
#                 # Implies(T3, And(Not(T2), Not(T1)))
#               ]

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
