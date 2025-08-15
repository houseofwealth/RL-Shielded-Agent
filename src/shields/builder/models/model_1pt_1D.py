#1D version of model_1pt for testing
# TBD: 
# Implies(And(t != 0, x!=0), deadlineX == deadline) leads to divergence. Why? 
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
# y, yX, v_y, v_yX, a_y  = Reals('y yX v_y v_yX a_y')
# z, zX, v_z, v_zX, a_z  = Reals('z zX v_z v_zX a_z')

# prey_x, prey_y, prey_z, prey_vx, prey_vy, prey_vz = Reals('prey_x prey_y prey_z prey_vx prey_vy prey_vz')

t,tX =                      Consts('t tX', Time)  # k>=0, t=now >= 0.
K =                         Const('K', Time) 
deadline, deadlineX       = Consts('deadline deadlineX', Time)


if _3D: zstate = [z, v_z] 
else:   zstate = []
if _3D: zXstate = [zX, v_zX] 
else:   zXstate = []

globals     = []
state       = [x, v_x, t, deadline]      #not a list of the vars but a list of the corresp Z3 structs!
stateX      = [xX,v_xX,tX,deadlineX] + zXstate
subst       = list(zip(state,stateX)) 
inv_subst   = list(zip(stateX,state))

# Env, Sys Control variables
e,u = Ints('e u')

# boundedReachabilityProp = And(x==R, y==T)
# boundedReachabilityProp = And(x==prey_x, y==prey_y)  <-- to hit exactly
# Z3 has no abs fn: boundedReachabilityProp = And(abs(x - prey_x) < TARGET_RADIUS, abs(y, prey_y) <= TARGET_RADIUS)
boundedReachabilityProp = And(-TARGET_RADIUS <= x, x <= TARGET_RADIUS)
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

at_tgt = x==0 #And(x == prey_x, y == prey_y)

if DOING_BOUNDED:
  #from 2.1.2 when tgt reached, can remove the deadline
  BR1 = Implies(at_tgt, deadlineX == NONE)     
  #from 2.1.3 that says t <= min{Deadlines}
  BR2 = Implies(deadline != NONE, t <= deadline)
  #tfrom 2.2 replacement of <>_k phi in formula by deadlineX == t+K
  BR3 = Implies(t==0, deadlineX == t+STEPS_BOUND) <-- apparantly this hsould be just "deadline"
  BR = [BR2,BR3]  #BR2 moved into the action def
else:
  BR = []
BRD = (Implies(And(t!=0,x!=0), deadlineX == deadline))
# Arcs/Transitions 
#only one action corresponding to a choice made by the RL agent
stutter = {'name'       : "stutter",
          'actionPred' : And(xX == x, 
                             v_xX == v_x, 
                             deadlineX == -1,
                            #  BR1, BR3,
                             tX == t),
          # why is env var and pred on an action and not a node?
          'envVar'      : [],
          'controlVar'  : [], #a_x
          'controlPred' : at_tgt, #And(A_MIN<=a_x, a_x<=A_MAX),
          'blPred'      : boundedReachabilityProp, #And(x==R-1, y==B+1)
          # 'assumption' : True,
          # 'assumpVars' : [a_x,a_y],
          'precNode'    : singleton,
          'postNode'    : singleton
          }

action = {'name'       : "accelerate",
          'actionPred' : And(xX == x+delta_s(v_x,a_x), 
                             v_xX == v_x+delta_v(a_x), 
                             # doesn't converge: BRD,
                             tX == t+1),
          # why is env var and pred on an action and not a node?
          'envVar'      : [],
          'controlVar'  : [a_x],
          'controlPred' : And(A_MIN<=a_x, a_x<=A_MAX, Not(at_tgt)),
          'blPred'      : boundedReachabilityProp, #And(x==R-1, y==B+1)
          # 'assumption' : True,
          # 'assumpVars' : [a_x,a_y],
          'precNode'    : singleton,
          'postNode'    : singleton
          }
transitions = [action, stutter]

model       = {'name'        : '1 point predator, 1-D\n',
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

# safetyProps = [L <= x, x <= R, B <= y, y <= T] + ObstacleAvoid     #the invariant to be maintained
safetyProps = Geofencing + ObstacleAvoid + BR    #the invariant to be maintained


