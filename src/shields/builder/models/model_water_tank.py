# model of water tank taken from "Safe Reinforcement Learning via Shielding"

from z3       import *

Time = IntSort() 

#  state  variables
t, tX       = Consts('t tX', Time)
s, sX       = Consts('s sX', Time)
tank, tankX = Reals('tank tankX')
globals     = []
# state       = [t, tank] 
# stateX      = [tX,tankX]
state       = [t, s, tank] 
stateX      = [tX,sX,tankX]
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
                #this is open /\ (t=2 -> ~open /\ t=2 -> ~open )
                #ie And(open,Implies(Or(t==2, t==3), Not(open))), which simplifies to:
                'controlPred': And(open, t!=2, t!=3),   
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
                # why does this: 'controlPred': And(Not(open), s!=2, s!=3), instead of the one below lead to a *more* complex guard on the two actions? two extraneous conditions show up
                'controlPred': And(Not(open), Implies(Or(s==2, s==3), open)),
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
safetyPropsOpen = [ 0 <= tank, tank <= TANK_CAPACITY,
                    0 <= t, t <= NESTING_LEVEL,
                    Implies(And(t==0, open), tX==1),        #at the top level, if open, inc the counter
                    Implies(And(t==0, Not(open)), tX==0),   #if its not open, reset

                    Implies(And(t==1, Not(open)), tX==2),   #if its closed at level 2, inc the counter
                    Implies(And(t==1, open), tX==0),        #if its open, reset

                    Implies(t==2, tX==3),                   #inc the counter
                    Implies(t==3, tX==0),                   #end of cycle, reset
                    #for the reqts on t==2 and t==3 see openActions's control pred
                  ]
safetyPropsClsd = [ 0 <= tank, tank <= TANK_CAPACITY,
                    0 <= s, s <= NESTING_LEVEL,
                    Implies(And(s==0, Not(open)), sX==1),   #at the top level, if closed, inc the counter
                    Implies(And(s==0, open), sX==0),        #if its open, reset

                    Implies(And(s==1, open), sX==2),        #if its open at level 2, inc the counter
                    Implies(And(s==1, Not(open)), sX==0),   #if its closed, reset

                    Implies(s==2, sX==3),                   #inc the counter
                    Implies(s==3, sX==0),                   #end of cycle, reset
                    #for the reqts on t==2 and t==3 see closedActions's control pred
                  ]              #   0 <= s, s <= NESTING_LEVEL,
              #   Implies(And(s==0, Not(open)), sX==1),   #at the top level, if not open, inc the counter
              #   Implies(And(s==0, open), sX==0),        #if its  open, no change
              #   Implies(And(s==1, open), sX==2),        #if its open at level 2, inc the counter
              #   Implies(And(s==1, Not(open)), sX==0),   #if its closed, reset
              #  #  Implies(s==2, sX==0)                  #end of cycle, resetj
              #   Implies(s==2, sX==3),                   #inc the counter
              #   Implies(s==3, sX==0)                    #end of cycle, reset
safetyProps = safetyPropsOpen + safetyPropsClsd

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

t = 0; t1 = 0; t2 = 0; t3 = 0
#***NOTE: if an action choice fails, agent needs to retry with the negated choce, otherwise the t's wont be corectly updated
def OKProp1(open_valve, tank_level): #, t):
  global t,t1,t2,t3
  #the first 3 conds below are the starting constraint for control pred above
  if open_valve:
    res = fnAnd(Not(t == 2), Not(t == 3), tank_level <= 98)
    if res:     #don't modify the t if this choice fails as it will retry
      # match t:
      #   case 0: t=1
      #   case 1: t=0
      #   case 2: t=3
      #   case 3: t=0
      if   t==0: t=1    #open @ t=0
      elif t==1: t=0    #open @ t=1, reset
      elif t==2: t=3
      elif t==3: t=0

  """ if   t==1 and t1==0: t1=1    #open @ t1=0
      elif t1==1: t1=0             #open @ t1=1, reset
      elif t1==2: t1=3
      elif t1==3: t1=0 """

  # else:
  #   res = fnOr(t == 2, t == 3, fnAnd(1 <= tank, t == 0), 3 <= tank)
  #   if res:
  #     match t:
  #       case 0: t=0
  #       case 1: t=2
  #       case 2: t=3
  #       case 3: t=0
  return (res,t)

  # if open_valve and t==0: t = 1
  # return (res, t+1 mod (NESTING_LEVEL+1))

"""
with just safetyPropsOpen produces:
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

------------------ when safetyPropsClsd is added you get: --------------------
stateInv
 And(tank >= 0,
    tank <= 100,
    t >= 0,
    t <= 3,
    s >= 0,
    s <= 3,
    Or(And(tank <= 98, Not(t == 2), Not(t == 3)),
       And(1 <= tank, Not(s == 3), Not(s == 2))),
    Or(And(Not(s == 2), 2 <= tank),
       And(Not(t == 2), tank <= 96)))
newInvDelta:
 []
no change to invariant on singleton

----------------
Final Model - fixpoint at iteration 4
Node: singleton
  Vars: [t, s, tank]
  Node Invariant: True
Transition: openAction
   0   open
   1   Not(t == 2)
   2   Not(t == 3)
   3   Or(s == 2,
   s == 3,
   And(t == 0, -94 <= -1*tank + ToReal(t)),
   tank <= 94,
   Not(2 <= tank),
   And(s == 0, tank <= 98),
   And(t == 0, t == s, -98 <= ToReal(t) + -1*tank))
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
   15   Or(open, Not(s == 0), sX == 1)
   16   Or(Not(s == 0), Not(open), sX == 0)
   17   Or(Not(open), sX == 2, Not(s == 1))
   18   Or(open, sX == 0, Not(s == 1))
   19   Or(Not(s == 2), sX == 3)
   20   Or(Not(s == 3), sX == 0)
Transition: closeAction
   0   Not(open)
   1   Not(s == 2)
   2   Not(s == 3)
   3   Or(t == 2,
   t == 3,
   Not(tank <= 98),
   3 <= tank,
   And(t == 0, 1 <= tank))
   4   inn >= 1
   5   inn <= 2
   6   out >= 0
   7   out <= 1
   8   tankX == tank + -1*out
   9   Or(Not(open), tX == 1, Not(t == 0))
   10   Or(open, tX == 0, Not(t == 0))
   11   Or(open, tX == 2, Not(t == 1))
   12   Or(Not(open), tX == 0, Not(t == 1))
   13   Or(Not(t == 2), tX == 3)
   14   Or(Not(t == 3), tX == 0)
   15   Or(open, Not(s == 0), sX == 1)
   16   Or(Not(s == 0), Not(open), sX == 0)
   17   Or(Not(open), sX == 2, Not(s == 1))
   18   Or(open, sX == 0, Not(s == 1))
   19   Or(Not(s == 2), sX == 3)
   20   Or(Not(s == 3), sX == 0)

..safety shield construction complete

---------------------------------------------------------------------------------
but when you simplify only one of the initial control preds (see above) but not both you get a simpler solution:
----------------
Final Model - fixpoint at iteration 4
Node: singleton
  Vars: [t, s, tank]
  Node Invariant: True
Transition: openAction
   0   open
   1   Not(t == 2)
   2   Not(t == 3)
   3   Or(s == 2, s == 3, tank <= 94, And(s == 0, tank <= 98))
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
   15   Or(open, Not(s == 0), sX == 1)
   16   Or(Not(s == 0), Not(open), sX == 0)
   17   Or(Not(open), sX == 2, Not(s == 1))
   18   Or(open, sX == 0, Not(s == 1))
   19   Or(Not(s == 2), sX == 3)
   20   Or(Not(s == 3), sX == 0)
Transition: closeAction
   0   Not(open)
   1   Not(s == 2)
   2   Not(s == 3)
   3   Or(t == 2,
   t == 3,
   And(t == 0, 1 <= tank),
   Not(tank <= 97),
   3 <= tank)
   4   inn >= 1
   5   inn <= 2
   6   out >= 0
   7   out <= 1
   8   tankX == tank + -1*out
   9   Or(Not(open), tX == 1, Not(t == 0))
   10   Or(open, tX == 0, Not(t == 0))
   11   Or(open, tX == 2, Not(t == 1))
   12   Or(Not(open), tX == 0, Not(t == 1))
   13   Or(Not(t == 2), tX == 3)
   14   Or(Not(t == 3), tX == 0)
   15   Or(open, Not(s == 0), sX == 1)
   16   Or(Not(s == 0), Not(open), sX == 0)
   17   Or(Not(open), sX == 2, Not(s == 1))
   18   Or(open, sX == 0, Not(s == 1))
   19   Or(Not(s == 2), sX == 3)
   20   Or(Not(s == 3), sX == 0)

..safety shield construction complete

"""