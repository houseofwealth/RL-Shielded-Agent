#this isnt currently useable on my m/c becuse config needs env which needs Gym
from z3       import *
from config import DEFAULT_CONFIG
# from ipdb import set_trace

''' ------------------ CONSTANTS ------------------------------'''
_3D = False   #NB has to stay False for evaluation until Python is modified accordinly
DELTA_T = DEFAULT_CONFIG['env_config']['STEP_SIZE']
ws = DEFAULT_CONFIG['env_config']['workspace_size']
print('workspace_size = ', ws)
A_MAX = DEFAULT_CONFIG['env_config']['max_acceleration']
A_MIN = -A_MAX
print('acceleration bounds A_MAX =', A_MAX)
TARGET_RADIUS = DEFAULT_CONFIG['env_config']['AT_TARGET_RADIUS']
print('TARGET_RADIUS', TARGET_RADIUS)

L = -ws; R=ws; B=0; T=ws; C = -ws; F = ws
# LObs = -8; RObs=8; BObs=5; TObs=15            #is 16x10
# LObs = -4; RObs=4; BObs=3; TObs=8            #is 16x10
# x
LObs = DEFAULT_CONFIG['env_config']['LObs']
RObs = DEFAULT_CONFIG['env_config']['RObs']
#y if 3D, ognored otherwise
CObs = DEFAULT_CONFIG['env_config']['CObs']
FObs = DEFAULT_CONFIG['env_config']['FObs']
#z if 3D, y otherwise
TObs = DEFAULT_CONFIG['env_config']['TObs']
BObs = DEFAULT_CONFIG['env_config']['BObs']

DOING_OBSTACLES = DEFAULT_CONFIG['env_config']['DOING_OBSTACLES']
print("DOING_OBSTACLES", DOING_OBSTACLES)
if DOING_OBSTACLES: print('LObs RObs CObs FObs BObs TObs', LObs, RObs, CObs, FObs, BObs, TObs)
GEOFENCING = DEFAULT_CONFIG['env_config']['GEOFENCING']
print('GEOFENCING', GEOFENCING)
DOING_BOUNDED = DEFAULT_CONFIG['env_config']['DOING_BOUNDED']
print('DOING_BOUNDED', DOING_BOUNDED)
STEPS_BOUND = DEFAULT_CONFIG['env_config']['STEPS_BOUND']
print('STEPS_BOUND', STEPS_BOUND)


NONE = -1
# types
Time = IntSort()
'''-------------------------- Z3 VARS -------------------------------------'''
#  state variables - creates python vars whose values are the internal Z3 structs representing same
x, xX, v_x, v_xX, a_x  = Reals('x xX v_x v_xX a_x')
y, yX, v_y, v_yX, a_y  = Reals('y yX v_y v_yX a_y')
z, zX, v_z, v_zX, a_z  = Reals('z zX v_z v_zX a_z')

prey_x, prey_y, prey_z, prey_vx, prey_vy, prey_vz = Reals('prey_x prey_y prey_z prey_vx prey_vy prey_vz')

t,tX = Consts('t tX', Time)  # k>=0, t=now >= 0.
K = Const('K', Time) 
deadline, deadlineX       = Consts('deadline deadlineX', Time)


if _3D: zstate = [z, v_z] 
else:   zstate = []
if _3D: zXstate = [zX, v_zX] 
else:   zXstate = []

globals     = []
state       = [x, y, v_x, v_y, t, deadline] + zstate #,a_x,a_y]     #not a list of the vars but a list of the corresp Z3 structs!
stateX      = [xX,yX,v_xX,v_yX,tX,deadlineX] + zXstate
subst       = list(zip(state,stateX)) 
inv_subst   = list(zip(stateX,state))

# Env, Sys Control variables
e,u = Ints('e u')

# boundedReachabilityProp = And(x==R, y==T)
# boundedReachabilityProp = And(x==prey_x, y==prey_y)  <-- to hit exactly
# Z3 has no abs fn: boundedReachabilityProp = And(abs(x - prey_x) < TARGET_RADIUS, abs(y, prey_y) <= TARGET_RADIUS)
boundedReachabilityProp = And(-TARGET_RADIUS <= x - prey_x, x - prey_x <= TARGET_RADIUS, 
                              -TARGET_RADIUS <= y - prey_y, y - prey_y <= TARGET_RADIUS)
# Nodes
singleton  = {'name'         : 'main',
              'globals'      : globals,
              'vars'         : state,
              'postVars'     : stateX,    # optional: for self transitions
              'subst'        : subst,     # optional: for self transitions
              'invariant'    : t >= 0     # was True
              }

def delta_v(a):
  global DELTA_T
  return a*DELTA_T      #this is Z3's multiply

def delta_s(v,a):
  global DELTA_T
  deltav = delta_v(a)
  # wrong! return v*DELTA_T + (deltav/2)*DELTA_T*DELTA_T #dist due to current vel + dist due to accel
  return v*DELTA_T + (deltav/2)*DELTA_T #dist due to current vel + dist due to accel

if _3D: 
  zMove = zX == z+delta_s(v_z,a_z)
  zMovev = v_zX == v_z+delta_v(a_z) 
  zAccMin = A_MIN <= a_z
  zAccMax = a_z <= A_MAX
  zCtrl = [a_z]
else:
  zMove = True; zMovev = True; zAccMin = True; zAccMax = True; zCtrl = []

# Arcs/Transitions 
#only one action corresponding to a choice made by the RL agent
action = {'name'       : "accelerate",
          'actionPred' : And(xX == x+delta_s(v_x,a_x), 
                             v_xX == v_x+delta_v(a_x), 
                             yX == y+delta_s(v_y,a_y), v_yX == v_y+delta_v(a_y),
                             zMove, zMovev,
                             tX == t+1),
          # why is env var and pred on an action and not a node?
          'envVar'      : [],
          'controlVar' : [a_x,a_y] + zCtrl,
          'controlPred': And(A_MIN<=a_x, a_x<=A_MAX, 
                             A_MIN<=a_y, a_y<=A_MAX,
                             zAccMin, zAccMax),
          'blPred'       : boundedReachabilityProp, #And(x==R-1, y==B+1)
          # 'assumption' : True,
          # 'assumpVars' : [a_x,a_y],
          'precNode'   : singleton,
          'postNode'   : singleton
          }
transitions = [action]

model       = {'name'        : '1 point predator, 2-D',
               'initNode'    : singleton,
               'nodes'       : [singleton],
               'transitions' : transitions
               }

# Required Properties
initProps   = [deadline==0, t==0]

if DOING_OBSTACLES:  #[Or(x < LObs, x > RObs), Or(y < BObs, y > TObs)]
  if _3D:
    ObstacleAvoid = [Or(x < LObs, x > RObs, y < CObs, y > FObs, z < BObs, z > TObs)]
  else:
    ObstacleAvoid = [Or(x < LObs, x > RObs, y < BObs, y > TObs)]
else: 
  ObstacleAvoid = []

if GEOFENCING: 
  if _3D:
    Geofencing = [L <= x, x <= R, C <= y, y <= F, B <= z, T >= z]
  else:
    Geofencing = [L <= x, x <= R, B <= y, y <= T] 
else: 
  Geofencing = []

at_tgt = And(x == prey_x, y == prey_y)
if DOING_BOUNDED:
  #from 2.1.2 when unlock happens can remove the deadline
  BR1 = Implies(at_tgt, deadlineX == NONE)     
  #from 2.1.3 that says t <= min{Deadlines}
  BR2 = Implies(deadline != NONE, t <= deadline)
  #tfrom 2.2 replacement of <>_k phi in formula by deadlineX == t+K
  BR3 = Implies(t==0, deadlineX == t+STEPS_BOUND)
  BR = [BR1,BR2,BR3]
else:
  BR = []

# safetyProps = [L <= x, x <= R, B <= y, y <= T] + ObstacleAvoid     #the invariant to be maintained
safetyProps = Geofencing + ObstacleAvoid # + BR    #the invariant to be maintained

# model of single point agent moving in rectangular geofence - control signal is acceleration
def fnOr(xs): return any(xs)
def fnAnd(xs): return all(xs)
def fnNot(x): return not x

def OK(
  agent_action,
  curr_st,
  prey_st,
  bound,
  doing_bounded=DOING_BOUNDED,
  doing_obstacles=DOING_OBSTACLES,
  geofencing=GEOFENCING
  ):

  for constraint in [doing_bounded, doing_obstacles, geofencing]:
    assert isinstance(constraint, bool), 'Constraints must be boolean: ' + str(constraint)

  ok = True
  if doing_obstacles:
    ok = ok and OKObs(agent_action, curr_st)
  if geofencing:
    ok = ok and OKGeo(agent_action, curr_st)
  if doing_bounded:
    if ok: ok = OKBR(agent_action, curr_st, prey_st, bound) #this fn may return a list too
  return ok

# workspace_size =  10
# DOING_OBSTACLES False
# acceleration bounds A_MAX = 10
# GEOFENCING True
def OKGeo(agent_action, curr_st):
  # print('quickOK: agent_action', agent_action, 'curr_st', curr_st)
  a_x,a_y = agent_action
  x,y, v_x,v_y = curr_st
  return\
    fnAnd([ y + v_y + 1/2*a_y >= 0,
            x + v_x + 1/2*a_x <= 10,
            y + v_y + 1/2*a_y <= 10,
            x + v_x + 1/2*a_x >= -10,
            2*x + 4*v_x + 3*a_x <= 30,
            -1*x + -2*v_x + -3/2*a_x <= 15,
            -1*y + -2*v_y + -3/2*a_y <= 5,
            2*y + 4*v_y + 3*a_y <= 30])
""" # acceleration bounds A_MAX = 1
   4   x + v_x + 1/2*a_x >= -10
   5   x + v_x + 1/2*a_x <= 10
   6   y + v_y + 1/2*a_y >= 0
   7   y + v_y + 1/2*a_y <= 10
   8   2*y + 4*v_y + 3*a_y <= 21
   9   2*x + 4*v_x + 3*a_x <= 21
   10   -1*x + -2*v_x + -3/2*a_x <= 21/2
   11   -1*y + -2*v_y + -3/2*a_y <= 1/2
   12   -1*x + -3*v_x + -5/2*a_x <= 12
   13   -1*y + -3*v_y + -5/2*a_y <= 2
   14   2/3*y + 2*v_y + 5/3*a_y <= 8
   15   2/3*x + 2*v_x + 5/3*a_x <= 8
   16   -1*y + -4*v_y + -7/2*a_y <= 9/2
   17   -1*x + -4*v_x + -7/2*a_x <= 29/2
   18   2/5*x + 8/5*v_x + 7/5*a_x <= 29/5
   19   2/5*y + 8/5*v_y + 7/5*a_y <= 29/5
   20   -1*x + -5*v_x + -9/2*a_x <= 18
   21   -1*y + -5*v_y + -9/2*a_y <= 8
   22   2/7*x + 10/7*v_x + 9/7*a_x <= 36/7
   23   2/7*y + 10/7*v_y + 9/7*a_y <= 36/7
   24   -1*x + -6*v_x + -11/2*a_x <= 45/2
   25   2/9*x + 4/3*v_x + 11/9*a_x <= 5
   26   xX == x + v_x*1 + ((a_x*1)/2)*1
   27   v_xX == v_x + a_x*1
   28   yX == y + v_y*1 + ((a_y*1)/2)*1
   29   v_yX == v_y + a_y*1 """



# workspace_size =  10
# DOING_OBSTACLES True
# LObs RObs BObs TObs -4 4 3 8
# acceleration bounds A_MAX = 10
# GEOFENCING True
def OKObs(agent_action, curr_st):
  a_x,a_y = agent_action
  x,y, v_x,v_y = curr_st
  return\
   fnOr([ fnNot(y + v_y + 1/2*a_y <= 8),
          fnNot(x + v_x + 1/2*a_x <= 4),
          fnNot(-4 <= x + v_x + 1/2*a_x),
          fnNot(3 <= y + v_y + 1/2*a_y)])

""" A_MAX = 1
   Not(y + v_y + 1/2*a_y <= 8),
   Not(-4 <= x + v_x + 1/2*a_x),
   Not(3 <= y + v_y + 1/2*a_y))
   5   Or(Not(-15/2 <= -1*y + -2*v_y + -3/2*a_y),
   Not(7 <= 2*y + 4*v_y + 3*a_y),
   Not(-7 <= 2*x + 4*v_x + 3*a_x),
   Not(-7/2 <= -1*x + -2*v_x + -3/2*a_x))
   6   Or(Not(-2 <= -1*x + -3*v_x + -5/2*a_x),
   Not(-6 <= -1*y + -3*v_y + -5/2*a_y),
   Not(10/3 <= 2/3*y + 2*v_y + 5/3*a_y),
   Not(-4/3 <= 2/3*x + 2*v_x + 5/3*a_x))
   7   Or(Not(-15/2 <= -1*y + -2*v_y + -3/2*a_y),
   -7/2 <= -1*x + -2*v_x + -3/2*a_x,
   Not(7 <= 2*y + 4*v_y + 3*a_y),
   Not(-2 <= -1*x + -3*v_x + -5/2*a_x),
   Not(-6 <= -1*y + -3*v_y + -5/2*a_y),
   Not(-31/6 <= -2/3*x + -1*v_x + -2/3*a_x),
   Not(10/3 <= 2/3*y + 2*v_y + 5/3*a_y))
   8   Or(Not(-15/2 <= -1*y + -2*v_y + -3/2*a_y),
   Not(-7 <= 2*x + 4*v_x + 3*a_x),
   Not(-7/2 <= -1*x + -2*v_x + -3/2*a_x),
   Not(-2 <= -1*x + -3*v_x + -5/2*a_x),
   Not(3/2 <= 2*y + 3*v_y + 2*a_y),
   Not(10/3 <= 2/3*y + 2*v_y + 5/3*a_y),
   Not(-4/3 <= 2/3*x + 2*v_x + 5/3*a_x))
   9   Or(-15/2 <= -1*y + -2*v_y + -3/2*a_y,
   Not(-7 <= 2*x + 4*v_x + 3*a_x),
   Not(-7/2 <= -1*x + -2*v_x + -3/2*a_x),
   Not(-2 <= -1*x + -3*v_x + -5/2*a_x),
   Not(-41/6 <= -2/3*y + -1*v_y + -2/3*a_y),
   Not(-6 <= -1*y + -3*v_y + -5/2*a_y),
   Not(-4/3 <= 2/3*x + 2*v_x + 5/3*a_x))
   10   Or(Not(-15/2 <= -1*y + -2*v_y + -3/2*a_y),
   Not(7 <= 2*y + 4*v_y + 3*a_y),
   Not(-7/2 <= -1*x + -2*v_x + -3/2*a_x),
   Not(-31/2 <= 2*x + 3*v_x + 2*a_x),
   -2 <= -1*x + -3*v_x + -5/2*a_x,
   Not(-6 <= -1*y + -3*v_y + -5/2*a_y),
   Not(10/3 <= 2/3*y + 2*v_y + 5/3*a_y)) """

def OKBR(agent_action, curr_st, prey_st, bound):
  # print('bound', bound)
  a_x,a_y = agent_action
  # x,y, v_x,v_y = curr_st
  # prey_x, prey_y = prey_pos
  x,y, v_x_abs,v_y_abs = curr_st
  prey_x, prey_y, prey_vx, prey_vy = prey_st
  v_x, v_y = v_x_abs - prey_vx, v_y_abs - prey_vy  #use relative velocities
  
  at_tgt = x == prey_x and y == prey_y
  if TARGET_RADIUS == 1:
    one_step = fnAnd([y + v_y + 1/2*a_y - prey_y >= -1,
                      y + v_y + 1/2*a_y - prey_y <= 1,
                      x + v_x + 1/2*a_x - prey_x >= -1,
                      x + v_x + 1/2*a_x - prey_x <= 1])
    two_steps = fnAnd([-1*x + -2*v_x + -3/2*a_x + prey_x <= 6,
                      x + 2*v_x + 3/2*a_x + -1*prey_x <= 6,
                      y + 2*v_y + 3/2*a_y + -1*prey_y <= 6,
                      -1*y + -2*v_y + -3/2*a_y + prey_y <= 6])
    three_steps= fnAnd([2/3*y + 2*v_y + 5/3*a_y + -2/3*prey_y <= 14,
                        2/3*x + 2*v_x + 5/3*a_x + -2/3*prey_x <= 14,
                        -1*y + -3*v_y + -5/2*a_y + prey_y <= 21,
                        -1*x + -3*v_x + -5/2*a_x + prey_x <= 21])
  elif TARGET_RADIUS == 0:
    one_step = fnAnd([y + v_y + 1/2*a_y == prey_y,
                      x + v_x + 1/2*a_x == prey_x])
    two_steps = fnAnd([-1*x + -2*v_x + -3/2*a_x + prey_x <= 5,
                      x + 2*v_x + 3/2*a_x + -1*prey_x <= 5,
                      y + 2*v_y + 3/2*a_y + -1*prey_y <= 5,
                      -1*y + -2*v_y + -3/2*a_y + prey_y <= 5])
    three_steps= fnAnd([2/3*y + 2*v_y + 5/3*a_y + -2/3*prey_y <= 40/3,
                        2/3*x + 2*v_x + 5/3*a_x + -2/3*prey_x <= 40/3,
                        -1*y + -3*v_y + -5/2*a_y + prey_y <= 20,
                        -1*x + -3*v_x + -5/2*a_x + prey_x <= 20])
  else:
    print('***ERROR: TARGET_RADIUS can currently only be 0 or 1')
  if bound > 1:
    res = fnOr([#bound==1 and (one_step or at_tgt),
                bound==2 and two_steps,
                bound==3 and three_steps 
               ])
    # thes weakenings aren't correct for same reason as for solnExists
    # res = fnOr([#bound==1 and (one_step or at_tgt),
    #             bound==2 and (two_steps or one_step or at_tgt),
    #             bound==3 and (three_steps or two_steps or one_step or at_tgt)
    #            ])
    
    '''when one staep away from tgt, agent unlikely to discover exact value so supply the correct value provided its not going to violate some other constraint
    '''
  elif bound==1:
    new_acc = [(prey_x - x - v_x)*2, 
               (prey_y - y - v_y)*2] + new_acc_z
    # new_pos = [x + delta_s(v_x,new_acc[0]),
    #            y + delta_s(v_y,new_acc[1])] + new_z
    # print('new_acc', new_acc)
    if implies(DOING_OBSTACLES, OKObs(new_acc, curr_st)) and implies(GEOFENCING, OKGeo(new_acc, curr_st)):
      res = new_acc
    else:
      res = False
  else:
    res = res =at_tgt
  return res

if _3D: 
  new_acc_z = [(prey_z - z - v_z)*2]
else:
  new_acc_z = []

# -----------
def solnExists(curr_st, prey_st, bound):
  ok = True
  # ok = implies(DOING_OBSTACLES, solnExistsObsPy(curr_st))
  # ok = ok and implies(GEOFENCING, solnExistsGeoPy(curr_st))
  # ok = ok and implies(DOING_BOUNDED, solnExistsBRPy(curr_st, prey_st, bound))
  if DOING_OBSTACLES:
    ok = solnExistsObsPy(curr_st)
  if GEOFENCING:
    ok = ok and solnExistsGeoPy(curr_st)
  if DOING_BOUNDED:
    which_ones = solnExistsBRPy(curr_st, prey_st, bound) 
    if not any(which_ones) or not ok:
      return False
    else:
      return which_ones
  return ok

def solnExistsObsPy(curr_st):
  #fixpoint after 2 iterations, simple inv because obst is small enough it can accelerate in the other dimension if about to hit ob
  x,y, _,_ = curr_st
  return\
    fnOr([fnNot(-4 <= x), fnNot(x <= 4), fnNot(3 <= y), fnNot(y <= 8)])

def solnExistsGeoPy(curr_st):
  #fixpoint after 3 iterations
  x,y, v_x,v_y = curr_st
  return \
    fnAnd([x >= -10,
          x <= 10,
          y >= 0,
          y <= 10,
          -1*y + -1*v_y <= 5,
          2*x + 2*v_x <= 30,
          2*y + 2*v_y <= 30,
          -1*x + -1*v_x <= 15,
          2/3*x + 4/3*v_x <= 20,
          -1*x + -2*v_x <= 30])

def solnExistsBRPy(curr_st, prey_st, bound):
  if TARGET_RADIUS == 1: return solnExistsBR_TGT_RAD_1Py(curr_st, prey_st, bound)
  else: 
    assert TARGET_RADIUS < 1, "TARGET_RADIUS can no more than 1"
    return solnExistsBR_TGT_RAD_0Py(curr_st, prey_st, bound)

def solnExistsBR_TGT_RAD_0Py(curr_st, prey_st, bound):
  x,y, v_x_abs,v_y_abs = curr_st
  # prey_x, prey_y = prey_pos
  prey_x, prey_y, prey_vx, prey_vy = prey_st
  v_x, v_y = v_x_abs - prey_vx, v_y_abs - prey_vy  #use relative velocities
  at_tgt = x == prey_x and y == prey_y
  one_step = \
          fnAnd([-1*x + -1*v_x + prey_x <= 5,
                y + v_y + -1*prey_y <= 5,
                x + v_x + -1*prey_x <= 5,
                -1*y + -1*v_y + prey_y <= 5])
  two_step = \
        fnAnd([2/3*x + 4/3*v_x + -2/3*prey_x <= 40/3,
                2/3*y + 4/3*v_y + -2/3*prey_y <= 40/3,
                -1*y + -2*v_y + prey_y <= 20,
                -1*x + -2*v_x + prey_x <= 20])
  three_step = \
        fnAnd([2/5*x + 6/5*v_x + -2/5*prey_x <= 18,
                -1*y + -3*v_y + prey_y <= 45,
                2/5*y + 6/5*v_y + -2/5*prey_y <= 18,
                -1*x + -3*v_x + prey_x <= 45])
  # if one_step: print('one_Step')
  # if two_step: print('two_Step')
  # if three_step: print('three_Step')

  which_ones = [at_tgt, one_step, two_step, three_step]

  return which_ones
    # dont think these weakenings will work, each step condition is based on the variant and the variant is specific to the bound length
    # (bound==1 and (one_step or at_tgt)) or\
    # (bound==2 and (two_step or one_step or at_tgt)) or\
    # (bound==3 and (three_step or two_step or one_step or at_tgt))

def solnExistsBR_TGT_RAD_1Py(curr_st, prey_st, bound):
  x,y, v_x_abs,v_y_abs = curr_st
  # prey_x, prey_y = prey_pos
  prey_x, prey_y, prey_vx, prey_vy = prey_st
  v_x, v_y = v_x_abs - prey_vx, v_y_abs - prey_vy  #use relative velocities
  at_tgt = x == prey_x and y == prey_y
  one_step = \
          fnAnd([-1*x + -1*v_x + prey_x <= 6,
                y + v_y + -1*prey_y <= 6,
                x + v_x + -1*prey_x <= 6,
                -1*y + -1*v_y + prey_y <= 6])
  two_step = \
        fnAnd([2/3*x + 4/3*v_x + -2/3*prey_x <= 14,
                2/3*y + 4/3*v_y + -2/3*prey_y <= 14,
                -1*y + -2*v_y + prey_y <= 21,
                -1*x + -2*v_x + prey_x <= 21])
  three_step = \
        fnAnd([2/5*x + 6/5*v_x + -2/5*prey_x <= 92/5,
                -1*y + -3*v_y + prey_y <= 46,
                2/5*y + 6/5*v_y + -2/5*prey_y <= 92/5,
                -1*x + -3*v_x + prey_x <= 46])
  # if one_step: print('one_Step')
  # if two_step: print('two_Step')
  # if three_step: print('three_Step')

  which_ones = [at_tgt, one_step, two_step, three_step]

  return which_ones


def implies(p,q):
  return not p or q
# -------------------------- Test --------------------------------------------
BOUND = 3
if __name__ == "__main__":
  curr_st = [9.0, 7.59532356262207, 1.7186055183410645, 4.43661117553711] #[5.0, 0.0, 0.0, 0.0]
  # prey_pos = [0.0, 7.0] #[-1.0, 2.0]
  prey_st = [0.0, 10.0, 0.0, -5.0] #[-1.0, 2.0]
  which_ones = solnExists(curr_st, prey_st, BOUND)
  print(which_ones)
  if which_ones and which_ones != True:
    for bound,bound_solvable in enumerate(which_ones):
      if bound_solvable:
    # if soln:
        action = [-5.0, -5.0] #[ 8.67989302, -4.5718205 ]
        actionOK = OKBR(action, curr_st, prey_st, bound)
        print('actionOK', actionOK)







#------------------------ polygonal geofence -----------------------------------------------------
BOUNDARY = [(0,0), (0,5), (10,5), (10,0)]
def isInside(x,y): 
  num_intersections = 0
  for edge in edgesOf(BOUNDARY):                
    if intersectsLeftRay(edge, x, y):
      num_intersections = num_intersections+1
  if isOdd(num_intersections):
    return True
  else:
    return False

def edgesOf(boundary):
  num_points = len(boundary)
  return [(boundary[i],boundary[(i+1) % num_points]) for i in range(num_points)]

def intersectsLeftRay(edge, x, y):
  # x,y = (RealVal(x), RealVal(y))
  p,q = edge[0], edge[1]                            #edge is pair ((1,2), (3,4))
  px,py,qx,qy = p[0], p[1], q[0], q[1]              #edge is from p to q
  px,py,qx,qy = RealVal(px), RealVal(py), RealVal(qx), RealVal(qy)
  # x,y = toFloat(x), toFloat(y)                      #convert to Python floats
  # if py == qy and qy == x: return True              #edge is horizontal and point lies on it
  # if (py < y and qy < y) or (py > y and qy > y): 
  #have to say Or(And(..), Not(Or(And(..))))
  if And(py == qy, qy == x): return True              #edge is horizontal and point lies on it
  if Or(And(py < y, qy < y), And(py > y, qy > y)): 
    return False                                    #point is below or above edge
  intersection = calcIntersection(edge, x, y)
  if intersection[0] <= x: return True               #ray from left hits edge before point
  else: return False

def calcIntersection(edge, x, y):
  p,q = edge[0], edge[1]                            #edge is pair ((1,2), (3,4))
  px,py,qx,qy = p[0], p[1], q[0], q[1]              #edge is from p to q
  px,py,qx,qy = RealVal(px), RealVal(py), RealVal(qx), RealVal(qy)
  if px == qx:                                      #vertical edge
    return (px,y)
  return (0,0) #TBD

def isOdd(n):
  return n%2 != 0
# from fractions import *

def toFloat(r): 
  f = r.as_fraction()
  return f.numerator/f.denominator









#--------------------------- Graveyard ---------------------------


""" THIS was under OKObs but LOOKS LIKE A MISTAKE - combo of geofencing and obs
  return\
    fnAnd([ fnOr([fnAnd([ x + v_x + 1/2*a_x >= -10,
                          y + v_y + 1/2*a_y >= 0,
                          x + v_x + 1/2*a_x <= 10,
                          fnNot(3 <= y + v_y + 1/2*a_y)]),
                  fnAnd([ x + v_x + 1/2*a_x >= -10,
                          y + v_y + 1/2*a_y <= 10,
                          x + v_x + 1/2*a_x <= 10,
                          fnNot(y + v_y + 1/2*a_y <= 8)]),
                  fnAnd([ y + v_y + 1/2*a_y >= 0,
                          y + v_y + 1/2*a_y <= 10,
                          x + v_x + 1/2*a_x <= 10,
                          fnNot(x + v_x + 1/2*a_x <= 4)]),
                  fnAnd([ x + v_x + 1/2*a_x >= -10,
                          y + v_y + 1/2*a_y >= 0,
                          y + v_y + 1/2*a_y <= 10,
                          fnNot(-4 <= x + v_x + 1/2*a_x)])
                ]),
            2*y + 4*v_y + 3*a_y <= 30,
            -1*x + -2*v_x + -3/2*a_x <= 15,
            -1*y + -2*v_y + -3/2*a_y <= 5,
            2*x + 4*v_x + 3*a_x <= 30
          ]) """

# attmpt to do with with Global vars set in a diff module - failed
# def oa():
#   if DOING_OBSTACLES:  #[Or(x < LObs, x > RObs), Or(y < BObs, y > TObs)]
#     # ObstacleAvoid = [Not(And(LObs <= x, x <= RObs, BObs <= y, y <= TObs))]
#     return [Or(x < LObs, x > RObs, y < BObs, y > TObs)]
#   else: return[]
# def bounds(L,R,B,T): 
#   # global L,R,B,T
#   return [L <= x, x <= R, B <= y, y <= T]

  # fnOr([at_tgt,
        # fnAnd([-1*x + -1*v_x + prey_x <= 5,
        #         y + v_y + -1*prey_y <= 5,
        #         x + v_x + -1*prey_x <= 5,
        #         -1*y + -1*v_y + prey_y <= 5]),
        # fnAnd([2/3*x + 4/3*v_x + -2/3*prey_x <= 40/3,
        #         2/3*y + 4/3*v_y + -2/3*prey_y <= 40/3,
        #         -1*y + -2*v_y + prey_y <= 20,
        #         -1*x + -2*v_x + prey_x <= 20]),
        # fnAnd([2/5*x + 6/5*v_x + -2/5*prey_x <= 18,
        #         -1*y + -3*v_y + prey_y <= 45,
        #         2/5*y + 6/5*v_y + -2/5*prey_y <= 18,
        #         -1*x + -3*v_x + prey_x <= 45])
        # fnAnd([-1*x + -3*v_x + prey_x <= 45,
        #         -1*y + -3*v_y + prey_y <= 45,
        #         2/5*y + 6/5*v_y + -2/5*prey_y <= 18,
        #         2/5*x + 6/5*v_x + -2/5*prey_x <= 18])

          # fnOr([fnAnd([x == prey_x, y == prey_y]),
          #       fnAnd([ 
          #               fnOr([2/5*x + 6/5*v_x + -2/5*prey_x <= 18,
          #                     x + v_x + -1*prey_x <= 5]),
          #               fnOr([-1*y + -2*v_y + prey_y <= 20,
          #                   2/5*y + 6/5*v_y + -2/5*prey_y <= 18]),
          #               fnOr([fnNot(2/3*x + 4/3*v_x + -2/3*prey_x <= 40/3),
          #                     -1*y + -2*v_y + prey_y <= 20,
          #                     2/5*x + 6/5*v_x + -2/5*prey_x <= 18]),
          #               fnOr([-1*x + -1*v_x + prey_x <= 5,
          #                   2/3*x + 4/3*v_x + -2/3*prey_x <= 40/3]),
          #               fnOr([-1*y + -1*v_y + prey_y <= 5,
          #                   2/5*x + 6/5*v_x + -2/5*prey_x <= 18,
          #                   2/3*x + 4/3*v_x + -2/3*prey_x <= 40/3]),
          #               fnOr([y + v_y + -1*prey_y <= 5,
          #                   2/5*x + 6/5*v_x + -2/5*prey_x <= 18,
          #                   2/3*x + 4/3*v_x + -2/3*prey_x <= 40/3]),
          #               fnOr([2/3*y + 4/3*v_y + -2/3*prey_y <= 40/3,
          #                   2/5*x + 6/5*v_x + -2/5*prey_x <= 18,
          #                   fnNot(2/3*x + 4/3*v_x + -2/3*prey_x <= 40/3)]),
          #               fnOr([2/5*x + 6/5*v_x + -2/5*prey_x <= 18,
          #                   -1*x + -2*v_x + prey_x <= 20]),
          #               fnOr([-1*x + -3*v_x + prey_x <= 45,
          #                   -1*x + -1*v_x + prey_x <= 5]),
          #               fnOr([y + v_y + -1*prey_y <= 5,
          #                   -1*x + -3*v_x + prey_x <= 45,
          #                   -1*x + -2*v_x + prey_x <= 20]),
          #               fnOr([-1*x + -3*v_x + prey_x <= 45,
          #                   -1*x + -2*v_x + prey_x <= 20,
          #                   -1*y + -1*v_y + prey_y <= 5]),
          #               fnOr([x + v_x + -1*prey_x <= 5,
          #                   -1*x + -2*v_x + prey_x <= 20]),
          #               fnOr([-1*x + -2*v_x + prey_x <= 20,
          #                   -1*y + -1*v_y + prey_y <= 5,
          #                   2/5*y + 6/5*v_y + -2/5*prey_y <= 18]),
          #               fnOr([y + v_y + -1*prey_y <= 5,
          #                   -1*y + -3*v_y + prey_y <= 45,
          #                   -1*x + -2*v_x + prey_x <= 20]),
          #               fnOr([2/3*y + 4/3*v_y + -2/3*prey_y <= 40/3,
          #                   x + v_x + -1*prey_x <= 5,
          #                   -1*x + -3*v_x + prey_x <= 45]),
          #               fnOr([-1*x + -3*v_x + prey_x <= 45,
          #                   2/3*x + 4/3*v_x + -2/3*prey_x <= 40/3]),
          #               fnOr([-1*y + -2*v_y + prey_y <= 20,
          #                   x + v_x + -1*prey_x <= 5,
          #                   -1*x + -3*v_x + prey_x <= 45]),
          #               fnOr([2/3*y + 4/3*v_y + -2/3*prey_y <= 40/3,
          #                   2/5*y + 6/5*v_y + -2/5*prey_y <= 18,
          #                   x + v_x + -1*prey_x <= 5]),
          #               fnOr([y + v_y + -1*prey_y <= 5,
          #                   -1*y + -3*v_y + prey_y <= 45,
          #                   2/3*x + 4/3*v_x + -2/3*prey_x <= 40/3]),
          #               fnOr([-1*y + -1*v_y + prey_y <= 5,
          #                   2/5*y + 6/5*v_y + -2/5*prey_y <= 18,
          #                   2/3*x + 4/3*v_x + -2/3*prey_x <= 40/3]),
          #               fnOr([y + v_y + -1*prey_y <= 5,
          #                   2/5*y + 6/5*v_y + -2/5*prey_y <= 18]),
          #               fnOr([-1*y + -3*v_y + prey_y <= 45,
          #                   -1*y + -2*v_y + prey_y <= 20,
          #                   x + v_x + -1*prey_x <= 5]),
          #               fnOr([-1*y + -3*v_y + prey_y <= 45,
          #                   -1*y + -2*v_y + prey_y <= 20,
          #                   -1*x + -1*v_x + prey_x <= 5]),
          #               fnOr([2/3*y + 4/3*v_y + -2/3*prey_y <= 40/3,
          #                   -1*y + -3*v_y + prey_y <= 45]),
          #               fnOr([y + v_y + -1*prey_y <= 5,
          #                   -1*y + -2*v_y + prey_y <= 20]),
          #               fnOr([-1*y + -1*v_y + prey_y <= 5,
          #                   -1*y + -3*v_y + prey_y <= 45]),
          #               fnOr([2/3*y + 4/3*v_y + -2/3*prey_y <= 40/3,
          #                   2/5*y + 6/5*v_y + -2/5*prey_y <= 18,
          #                   -1*x + -1*v_x + prey_x <= 5]),
          #               fnOr([-1*y + -1*v_y + prey_y <= 5,
          #                   2/3*y + 4/3*v_y + -2/3*prey_y <= 40/3])
          #             ])  #closes the And
          #     ])          #closes the top Or



  # two_steps = fnAnd([-1*y + -2*v_y + -3/2*a_y <= 4,
  #                 x + 2*v_x + 2*a_x == 9,
  #                 x + 2*v_x + 3/2*a_x <= 14,
  #                 y + 2*v_y + 2*a_y == 1,
  #                 y + 2*v_y + 3/2*a_y <= 6,
  #                 -1*x + -2*v_x + -3/2*a_x <= -4]),
# three_steps = And(y + 3*v_y + 9/2*a_y == prey_y,
  #                   x + 3*v_x + 9/2*a_x == prey_x)


#  Or(And(y + v_y + 1/2*a_y == prey_y,
#        x + v_x + 1/2*a_x == prey_x),
#    And(-1*x + -2*v_x + -3/2*a_x + prey_x <= 5,
#        x + 2*v_x + 3/2*a_x + -1*prey_x <= 5,
#        y + 2*v_y + 3/2*a_y + -1*prey_y <= 5,
#        -1*y + -2*v_y + -3/2*a_y + prey_y <= 5),
#    And(2/3*y + 2*v_y + 5/3*a_y + -2/3*prey_y <= 40/3,
#        2/3*x + 2*v_x + 5/3*a_x + -2/3*prey_x <= 40/3,
#        -1*y + -3*v_y + -5/2*a_y + prey_y <= 20,
#        -1*x + -3*v_x + -5/2*a_x + prey_x <= 20))

#----------------------- Old ---------------
# workspace_size =  10
# DOING_OBSTACLES True
# LObs RObs BObs TObs -4 4 3 8
# acceleration bounds A_MAX = 10
# GEOFENCING True
# def quickOK(agent_action, curr_st):
#    a_x,a_y = agent_action
#    x, y, v_x, v_y = curr_st

#    xX = x+delta_sPy(v_x,a_x)
#    v_xX = v_x+delta_vPy(a_x)
#    yX = y+delta_sPy(v_y,a_y)
#    v_yX = v_y+delta_vPy(a_y)
#    res =\
#           fnAnd(ObstacleAvoidX(xX,yX) + geofencingX(xX,yX))
#    return res

# def quickSolnExists(curr_st):
#    x, y, v_x, v_y = curr_st

#    xX = x
#    v_xX = v_x
#    yX = y
#    v_yX = v_y

#    res =\
#           fnAnd([-1*xX + -2*v_xX <= 30,
#                   2/3*xX + 4/3*v_xX <= 20,
#                   -1*xX + -1*v_xX <= 15,
#                   -1*yX + -1*v_yX <= 5,
#                   2*xX + 2*v_xX <= 30,
#                   2*yX + 2*v_yX <= 30,
#                   fnOr([  fnNot(1 <= -1*xX + -1*v_xX),
#                           fnNot(1 <= 2/3*xX + 1/3*v_xX),
#                           fnNot(-3 <= -2/3*yX + -1/3*v_yX),
#                           fnNot(8 <= yX + v_yX)]),
#                   fnOr([  fnNot(1 <= -2/3*xX + -1/3*v_xX),
#                           fnNot(-3 <= -2/3*yX + -1/3*v_yX),
#                           fnNot(8 <= yX + v_yX),
#                           fnNot(1 <= xX + v_xX)]),
#                   fnOr([fnNot(y <= 8), fnNot(3 <= y), fnNot(x <= 4), fnNot(-4 <= x)])
#                 ]
#                )
#    return res
