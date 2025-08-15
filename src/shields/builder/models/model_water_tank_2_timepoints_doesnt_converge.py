# model of water tank taken from "Safe Reinforcement Learning via Shielding", uses 2 timers t but doesnt converge even after 50 iterations

from z3       import *

Time = IntSort() 

#  state  variables
t, tX       = Consts('t tX', Time)
u, uX       = Consts('u uX', Time)
tank, tankX = Reals('tank tankX')
globals     = []
state       = [t, tank] 
stateX      = [tX,tankX]
subst       = list(zip(state,stateX))
stateInv    = True 

# Env variables
inn = Real('inn')
out = Real('out')

# Control var
open = Bool('open')

# Nodes
singleton  = {'name'         : 'singleton',
              'globals'      : globals,
              'vars'         : state,
              'postVars'     : stateX,    
              'subst'        : subst,    
              'invariant'    : True}  

openAction = {'name'       : "openAction",
                'actionPred' : And( #moved -> U: open,      
                                    tankX == tank + inn - out), 
                'envVar'     : [inn, out],
                'envPred'    : And(1 <= inn, inn <= 2, 0 <= out, out <= 1),
                'invariant'  : True,
                'controlVar' : [open],
                #'controlPred': And(open,Implies(t==2, Not(open))),    
                #this is a hack b/c of entire guard not being updated. w/o it, chooses U = ~open
                #this is t=2 -> ~open /\ t=2 -> ~open 
                'controlPred': And(open,Implies(Or(t==2, t==3), Not(open)), 
                                        Implies(Or(u==2, u==3), Not(open)), ),     
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
                'controlPred': Not(open),  
                'precNode'   : singleton,
                'postNode'   : singleton
                }

model       = {'name'        : 'Water Tank',
               'initNode'    : singleton,
               'nodes'       : [singleton],
               'transitions' : [openAction, closeAction]
              }

# Required Properties
""" open /\ Xclosed --> XXclosed /\ XXXclosed == 
    ~open \/ X~closed \/ (XX~open /\ XXX~open) ===
    (~open \/ Xopen \/ XXclosed) /\ (~open \/ Xopen \/ XXXclosed)
"""
TANK_CAPACITY = 100
NESTING_LEVEL = 3
initProps   = [tank == 0, out == 0]
safetyProps_t = [ 0 <= tank, tank <= TANK_CAPACITY,
                  0 <= t, t <= NESTING_LEVEL,
                  Implies(And(t==0, open), tX==1),        #at the top level, if open, inc the counter
                  Implies(And(t==0, Not(open)), tX==0),   #if its not open, reset

                  Implies(And(t==1, Not(open)), tX==2),   #if its closed at level 2, inc the counter
                  Implies(And(t==1, open), tX==0),        #if its open, reset

                  Implies(t==2, tX==3),                   #inc the counter
                  Implies(t==3, tX==0),                   #end of cycle, reset
                  #for the reqts on t==2 and t==3 see openActions's control pred
                ]
safetyProps_u = [ 0 <= tank, tank <= TANK_CAPACITY,
                  0 <= u, u <= NESTING_LEVEL,
                  Implies(And(t==1, u==0, open), uX==1),        #at the top level, if open, inc the counter
                  Implies(And(t==1, u==0, Not(open)), uX==0),   #if its not open, reset

                  Implies(And(u==1, Not(open)), uX==2),   #if its closed at level 2, inc the counter
                  Implies(And(u==1, open), uX==0),        #if its open, reset

                  Implies(u==2, uX==3),                   #inc the counter
                  Implies(u==3, uX==0),                   #end of cycle, reset
                  #for the reqts on u==2 and u==3 see openActions's control pred
                ]
safetyProps = safetyProps_t + safetyProps_u
              #   0 <= s, s <= NESTING_LEVEL,
              #   Implies(And(s==0, Not(open)), sX==1),   #at the top level, if not open, inc the counter
              #   Implies(And(s==0, open), sX==0),        #if its  open, no change
              #   Implies(And(s==1, open), sX==2),        #if its open at level 2, inc the counter
              #   Implies(And(s==1, Not(open)), sX==0),   #if its closed, reset
              #  #  Implies(s==2, sX==0)                  #end of cycle, resetj
              #   Implies(s==2, sX==3),                   #inc the counter
              #   Implies(s==3, sX==0)                    #end of cycle, reset

def fnOr(xs): return any(xs)
def fnAnd(xs): return all(xs)
def fnNot(x): return not x

def solnExists(tank_level, t):
  return fnAnd( tank_level >= 0,
                tank_level <= 100,
                t >= 0,
                t <= 3,
                Or(Not(t == 3), 1 <= tank),
                Or(Not(t == 2), 2 <= tank))

def OK( open_valve, tank_level, t):
    return OKProp1(open_valve, tank_level, t) #and OKProp2(open_valve, tank_level)
    
