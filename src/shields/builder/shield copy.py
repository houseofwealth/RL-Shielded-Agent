# shield synthesis for non probabilistic MDPs, for multinode models where nodes may have more than one outarc
# NOTE
# the way that state props and step props are handled is quite different. Although both are initially /\'d into the action def, (a) for a step prop, the action def is actually updated, for a state prop, rely on the algorithm strengthening the guard and (b) if the step prop is an implicatino it works as long as the antecedent is a subset of some action's precond, for then you update an action def p /\ q to (p /\ q) /\ (p -> s'), which nicely simplifies to p /\ q /\ s'. If you don't have p -> s' but some arbitary t -> s', then you get a wierd action def p /\ q /\ (t -> s')
# to use, just do python shield.py

# TBD:
# . refactor code as per new mr.py
# . serialize out the constructed shield
# . have a flag to choose between control sees env or control for all env
# TODO: implement forward inference from InitProps to get an initial invariant per node

# from z3        import *
import z3
# print(__package__)
# print(__name__)
# from .utilities import *
# from .simplify  import *
# Note: Be sure that this directory (Z3_shield) is on PYTHONPATH!
# from meher.shields.Z3_shield.utilities import *
# from meher.shields.Z3_shield.simplify  import *
from utilities import *
from simplify  import *
# from random    import *
# from sys       import *
# import pickle

import pdb

#set_param('parallel.enable', True) # slows _buildSafetyShield down!?
# uncomment to prettyprint large expressions
#set_option(max_args=10000, max_lines=10000, max_depth=10000, max_visited=10000)

# Working Examples
# from meher.shields.Z3_shield.model_1pt import *
from model_1pt_1D import *

DBG0 = True         #simple tracing info of critical steps
DBG1 = True        #more dbg info

# TYPES
# state,state1:List[Var]
# subst:List[Pair(Expr,Expr)]
# stateInv:Expression,
# Node:   Dictionary{name:String, vars:VarList, invariant:Expr}
# Action: Dictionary{name:String, actionPred:Expr,
#                    envVar:Expr, envPred:Expr,
#                    controlVar:Expr, controlPred:Expr}
# initProps:List(stateProperty)
# transitions:List(Action),
# safetyProps:List(ActionProperty)

#tSimp = Tactic('simplify')
t0 = Tactic('qe_rec')

# extract from an action predicate the guards: conjuncts dependent only on stateVars
def getGuard(act, stateVars):
    #print("gg:", act, stateVars)
    result = []
    for c in expr2List(act['actionPred']):
        #print("c:", c)
        if isStatePred(c,stateVars):
            result.append(c)
    #print("result", result)
    guard = And(*result)
    guard = simplify(guard, local_ctx=True)
    #print("guard", guard)
    return guard

def getGuard1(act):
    return getGuard(act, act['precNode']['vars'])

# assume: uConstraint has the form And(lb <= u, u <= ub)
def getBounds(u, uConstraint):
    result = []
    print("u:", u, "|", uConstraint)
    for ineq in uConstraint.children():
        if u == ineq.children()[0]:
            result.append(ineq.children()[1].as_long())
            print(result)
    if result[0] > result[1]:
        return [result[1], result[0]]
    else:
        return result

""" def mkCtrlCondForTransToValidTgtSt(act):    #Paper has U -> wcp + \E u. U
    wcp = mkCondForTransToValidTgtSt(act)
    if "controlVar" in act:
        wcp = makeExists(act['controlVar'], And(act['controlPred'], wcp))   #SN: \E u. U /\ wcp
        # wcp = And(act['controlPred'], wcp)  #SN: \E u. U /\ wcp           #doesnt work
    if DBG1: print("wcp (ensures transition to valid tgt state):\n", str(wcp))
    return wcp 
"""

def mkCondForTransToValidTgtSt(stateInv, act): #this is the L_n(sX) term in the paper where sX is post(act) 
    postNode = act['postNode']
    if True: #act['precNode'] == postNode:             #ie. self loop action
        # print ('postNode[invariant]', postNode['invariant'])
        postNodeInv = substitute(stateInv, postNode['subst'])
        #temp = \A s'. A(s,s') -> L(s')
        newActPred = makeForAll(postNode['postVars'], Implies(act['actionPred'], postNodeInv))
    else:
        postNodeInv = postNode['invariant']
        #postNode's vars includes both its pre and post vars if the effect of the action is to 
        #change the post node's state
        wcp = makeForAll(postNode['vars'], Implies(act['actionPred'], postNodeInv))
    if 'envVar' in act and not isEmpty(act['envVar']):
        wcp = makeForAll(act['envVar'], Implies(act['envPred'], wcp))
    # print('wcp w/ env pred (if any) in front\n', wcp)
    return wcp

def containsVars(expr, vs):
    all_vars = get_vars(expr)
    return not set(all_vars).isdisjoint(vs)

def getConjNotContaining(p, vs):
    result = []
    for c in expr2List(p):
        #print("c:", c)
        if not containsVars(c,vs):
            result.append(c)
    #print("result", result)
    guard = And(*result)
    guard = simplify(guard, local_ctx=True)
    #print("guard", guard)
    return guard

def updateGuardDisj(guardDisjunction, act, node):
  #***not sure about this, need to revisit***, see comment in the paper in this section
  # the guard of the action is actually a comb of a true guard and an assumption about the env. need to remove out those assumption terms when determining how to strengthen the inv. eg a>10 /\ x=2, the true guard is just x=2
  if 'assumpVars' in act and not isEmpty(act['assumpVars']):
    effective_guard = getGuard(act, node['vars'] + act['assumpVars'])
    actual_updated_guard = getConjNotContaining(effective_guard, act['assumpVars'])
    #print('updated guard (excl. assumption):\n' + str(actual_updated_guard))
  else:
    effective_guard = getGuard(act, node['vars'])
    # updated_guard = getGuard1(new_act_pred, node['vars']) #was effective_guard
    #print('updated guard:\n' + str(effective_guard))
  
  effective_guard = Exists(act['controlVar'], effective_guard)
  #now use the (possibly) updated guard
  # guardDisjunction = simplify(Or(guardDisjunction, updated_guard), local_ctx=True)
  guardDisjunction = simplify(Or(guardDisjunction, effective_guard), local_ctx=True)
  #print("updated guardDisjunction:\n" + str(guardDisjunction))
  return guardDisjunction

def updateGuardDisj(guardDisjunction, act):
    # effective_guard = Exists(act['controlVar'], delta)
    # existing_guard = getGuard(act, node['vars'])
    foo = makeExists(act['controlVar'], act['controlPred']) 
    guardDisjunction = Or(guardDisjunction, foo)
    print("guardDisjunction:", guardDisjunction)
    projectGuardG = t0(guardDisjunction)[0]
    print("guardDisjunction after QE", projectGuardG)
    guardDisjunction = simplify(projectGuardG.as_expr())
    print("simplified guard disjunction:\n", str(guardDisjunction))
    return guardDisjunction
  
                    
# model refinement solver - for multinode models where nodes may have more than one outarc
# TODO:  initially, simplify each transition wrt the state invariant
def buildSafetyShield():
    print("Building shield for", model['name'])
    print('**Warning - not checking for action predicates')    #<-- is this still true?
    nodes    = model['nodes']
    transitions = model['transitions']

    # initial state Properties: what should this do? SN: initialState is unused
    initialState = True
    for initProp in initProps:
        f0 = simplify(And(initialState,initProp), local_ctx=True)   #print(f0)
        initialState = cdSimplifyE(f0).as_expr()

    # refine a node via state and action properties
    if DBG0: print('initializing node labels and arc labels with given safety properties')
    stateInv = singleton['invariant']
    for phi in safetyProps:
        # Need to do sth like [any node in ..] if DBG1 and isNodePred(phi, node): print("\nInitializing node labels with state property:\n", phi)
        if len(nodes) > 1: print('***ERROR: Assuming all nodes have same stateInv')
        for node in nodes:
            if isNodePred(phi, node):  # for state goals
                #SN: this is conjoining and simplifying phi wrt any pre-existing inv on node
                inv1 = simplify(And(stateInv, phi), local_ctx=True)
                newStateInv = simplify(cdSimplifyE(inv1).as_expr(), local_ctx=True)
                #print("  derived node invariant:", newStateInv)
                delta = residue(stateInv, newStateInv)  # print("delta:", delta)
                # node['invariant'] = newStateInv
                stateInv = newStateInv
                if DBG1:
                    if (len(delta) > 0):
                        print("strengthened node label on node", node['name'],"to\n", str(stateInv))
                    else:
                        print("no change on node", node['name'])

        # if DBG1 and not isNodePred(phi, node): print("\nInitializing arc labels with transition property:\n", phi)
        for act in transitions: # for action goals
            #g = getGuard(act, act['precNode']['vars'])
            if isArcPred(phi, act):
                #print("to transition:", act['name'])
                inv0 = And(act['actionPred'], phi)
                inv1 = simplify(inv0, local_ctx=True)             
                newActPred = simplify(cdSimplifyE(inv1).as_expr(), local_ctx=True)
                delta = residue(act['actionPred'], newActPred)
                act['actionPred'] = newActPred
                if DBG1:
                    if (len(delta) > 0):
                        print("strengthened arc label on arc", act["name"], "to:\n", newActPred)
                    else:
                        print("  no change to arc", act["name"])
        # else we have a path predicate; TODO: call pn
    print('..done')

# fixpoint iteration loop
    moreWork = Bool('moreWork')
    moreWork = True
    iterCnt = 0

    # loop invariant: 
    while (moreWork and iterCnt <= 50):
        moreWork = False
        if DBG0: print("\n----------------\niteration", iterCnt)
        # print("Current Model")
        # printModel(model)
        iterCnt = iterCnt + 1

        # refine each node invariant wrt its out-arcs and post-state prop
        for node in nodes:  
            if DBG1: print("\n-- Refining the state invariant for node", node['name'])
            # stateInv = node['invariant']
            guardDisjunction = False
            node_has_an_outgoing_arc = False
            for act in transitions:
                '''this 'ere is checking that start node of the arc is the current node'''
                if act['precNode']['name'] == node['name']:
                    node_has_an_outgoing_arc = True
                    if DBG1: print('--- arc:', act['name'])
                    # was guard = getGuard(act, node['vars'])
                    if not 'envVar' in node or isEmpty(node['envVar']):
                        guard = getGuard(act, node['vars'])
                    else:
                        guard = getGuard(act, node['vars'] + node['envVar'])
                    # INCORRECT wcp = mkCtrlCondForTransToValidTgtSt(act)        #this is E u. U /\ \A e. E(e) -> L_n(s'/vals)
                    wcp = mkCondForTransToValidTgtSt(stateInv, act)
                    print('wcp = \n', wcp)
                    wcp_simp = simplify(wcp)
                    #print('wcp_simp = \n', wcp_simp)
                    # eliminate quantifiers and then simplify
                    wcp_qe = t0(wcp_simp)[0].as_expr()  # returns Goal: list of conjuncts
                    #print("t0(wcTrans):", qewcTransG)
                    #this 2nd sim may not be needed?
                    # wcp_qe = simplify(wcp_qe.as_expr())
                    if DBG1: print('wcp_simp after QE (wcp_qe) =\n' + str(wcp_qe))
                    # may not be making any difference: wcp_e = cdSimplifyE(wcp_e).as_expr()
                    newActGuard = simplify(And(act['controlPred'], wcp_qe))
                    #SN: don't rememb what this line doing..A: its doing preState(a) <- preState(a) /\ wcp ie U_a <- U_a /\ wcp
                    # nib1 = simplify(And(*cdSimplifyE(And(guard, stateInv))), local_ctx=True)
                    #print("nib1:", nib1)

                        #ie if there's anything not implied by stateInv
                        #Q: this appears to be stregthening 
                        # stateInv + G_a rather than just G_a 
                        # is G_a := G_a /\ wcp as it is in the paper
                        # newActPred = simplify(And(*cdSimplifyE(And(wcp_x, act['actionPred']))), local_ctx=True)
                    wcp_resid_over_stateInv = residue(stateInv, newActGuard)
                    # if (len(wcp_resid_over_stateInv) > 0):
                    wcp_resid_over_stateInv = simplify(wcp_resid_over_stateInv.as_expr(), local_ctx=True)
                    controlPredDelta = residue(And(act['controlPred'], stateInv), wcp_resid_over_stateInv)
                    # if not(wcp_resid_over_stateInv == True):
                    if len(controlPredDelta) > 0:
                        act['controlPred'] = wcp_resid_over_stateInv
                        if DBG1: print("new control Pred:\n" + str(act['controlPred']))
                        moreWork = True
                    # guardDisjunction = updateGuardDisj(guardDisjunction, act, node)
                    guardDisjunction = updateGuardDisj(guardDisjunction, act)
            if node_has_an_outgoing_arc:
                if DBG1: print('\n- Completed arcs from node ' + node['name'] + '. Now checking if invariant needs updating (control pred may also get updated again)')
                # this is L_m -> someGuardTrue impl as L_m(s) := L_m(s) /\ someGuardTrue but optimzed so only update if changed
                print('stateInv\n', stateInv)
                newInvDelta = residue(stateInv, guardDisjunction)
                print("newInvDelta:\n", newInvDelta)
                #print("verf check:", verfEquiv(newInvDelta0,newInvDelta))
                # if(newInvDelta == True):  
                if(len(newInvDelta) == 0):  
                    if DBG1: print("no change to invariant on", node['name'])
                else:
                    newInv = cdSimplifyE(And(stateInv, *newInvDelta)).as_expr()
                    # doesnt look right, why would old inv be stronger than new one?:    verifyImplies(stateInv, newInv)      
                    verifyImplies(newInv, stateInv)    
                    stateInv = newInv   
                    # node['invariant'] = newInv
                    if DBG1: print("new invariant\n", stateInv)
                    moreWork = True
                    #now update actpred since inv changed
                    diff_ctrlPred_vs_newStateInv = residue(stateInv, act['controlPred']).as_expr()
                    print("new ctrl Pred (diff b/w updated state inv and current ctrl pred):\n", diff_ctrlPred_vs_newStateInv)
                    act['controlPred'] = diff_ctrlPred_vs_newStateInv
            else: 
               if DBG1: print('No refinement. Node is terminal')

    if moreWork == False:
        print("\n----------------\nFinal Model - fixpoint at iteration", iterCnt)
    else:
        print("\n****************\n**Terminated! after iteration", iterCnt)
    printModel(model)
    print("..safety shield construction complete")
# --------------------------- end shield def ----------------------------------

#     folder_name = '/storage/sn/meher2/meher/meher/experiments/z3_shield/'   
#     fname = folder_name + "model_1pt"
#     if DBG0: print('Serializing model to', fname)
#     ser_model = serialize(model, fname)
#     unser_model = unSerialize(ser_model)
#     if DBG1: printModel(unser_model)

# def serialize(model,fname):
#     with open(fname, "wb") as f:
#       pickle.dump(model, f)
#       unser_model = pickle.load(f)

def buildBLShield():
    BOUND = 3
    print('---------------------------------------------------\n')
    print("Computing bounded reachability for", model['name'], "to a bound of", BOUND)
    # moreWork = Bool('moreWork')
    transitions = model['transitions']
    # stateInv = singleton['invariant']
    variant = boundedReachabilityProp
    t0 = Tactic('qe_rec')

    if not(variant == True): 
        variantX = substitute(variant, subst)
    else:
        variantX = variant #was singleton['variant']

    # fixpoint iteration loop
    moreWork = True
    iterCnt = 0

    while iterCnt <= BOUND: 
        moreWork = False
        iterCnt = iterCnt + 1
        if DBG0: print("\n----------------\niteration", iterCnt)

        """ refine an arc's guard wrt its post-node ariant """
        # stateInvX = substitute(stateInv, subst)
        for actIndex in range(len(transitions)):
            act = transitions[actIndex]
            if DBG1: print("\nRefining guard on transition:", act['name'])
            print("variantX:\n", variantX)
            wcTrans = ForAll(stateX, Implies(act['actionPred'], variantX))

            if 'envVar' in act and not isEmpty(act['envVar']) and len(act['envVar']) > 0:
                wcTrans= ForAll(act['envVar'], Implies(act['envPred'], wcTrans))
            print("wcp:\n", wcTrans)
            wcTrans_simp = simplify(wcTrans)
            if DBG1: print('wcp_simp = \n', wcTrans_simp)
            # eliminate quantifiers and then simplify
            qewcTransG = t0(wcTrans_simp)[0]  # returns Goal: list of conjuncts
            newActGuard = qewcTransG.as_expr()
            # newActGuard = simplify(Or(at_tgt, qewcTransG.as_expr())) WRONG!!!!!!!!
            if DBG1: print("guard (newActGuard):\n", newActGuard)
            #SN: this residue is diff newActGuard - invariant
            newActGuard = simplify(residue(variant, newActGuard).as_expr())
            print("newActGuard - variant:\n", newActGuard)

            if iterCnt <= BOUND:  
                # transitions[actIndex]['controlPred'] = newActGuard
                act['blPred'] = newActGuard
                # blprop = newActGuard
            elif DBG1: print("no change to blPred")

        """ refine a node invariant wrt its out-arcs to establish someGuard, and poss update guard""" 
        #print("\nRefining the state invariant")
        guardDisjunction = False
        for actIndex in range(len(transitions)):
            act = transitions[actIndex]
            # was localGuard = act['controlPred']
            localGuard = act['blPred'] 
            #print("guard for transition", act['name'], ":", localGuard)
            if len(act['controlVar']) > 0:
                localGuard = Exists(act['controlVar'], And(act['controlPred'], localGuard))
            guardDisjunction = Or(guardDisjunction, localGuard)
        
        print("\nguardDisjunction:\n", guardDisjunction)
        projectGuardG = t0(guardDisjunction)[0]
        #print("projectGuardG", projectGuardG)
        guardDisjunction = simplify(projectGuardG.as_expr())
        print("guardDisjunction after QE and simpl:\n", guardDisjunction)
        variant = guardDisjunction
        variantX = substitute(variant, subst)
        print("variantX:\n", variantX)
        moreWork = True

        """now simplify each transition's guard wrt the new state variant"""
        for actIndex in range(len(transitions)):
            act = transitions[actIndex]
            # was actGuard = act['controlPred']
            act['blPred'] = residue(variant, act['blPred']).as_expr()
            print("new blPred after updating state v:\n", act['blPred'])

    print("\n----------------\nTerminated after iteration", iterCnt)
    # printModel(model) #state, stateInv, transitions)
    print('  blPred:\n', action['blPred'])
    print('variant:\n', variant)
    
# invoke model refinement on the model
# mr(state, stateX, subst, singleton['variant'], initProps, transitions, safetyProps)
# buildBLShield()
MAX_COUNT = 2
count = 0
def goalReached():
    global count
    if count <= MAX_COUNT: 
        count = count+1
        return False
    else: return True

def getNextAction(was_bad):
  # return [(a_x,randrange(10)), (a_y,randrange(10))] 
  return [float(randrange(-5,5)), float(randrange(-5,5))]
#   return np.array([float(randrange(10)), float(randrange(10))])
# -----------------------------------------------------------------------------

solver = Solver()

def toZ3Type(v):                          #wraps the vals in their Z3 type so Z3 doesnt complain
  if isinstance(v,bool)   : return BoolVal(v)
  if isinstance(v,int)    : return IntVal(v)
  if isinstance(v,float)  : return RealVal(v)
  if isinstance(v,str)    : return StringVal(v)
  raise BaseException('toZ3Type: unknown type for value ', v)

# def getZ3Var(index):
#   return action['controlVar'][index]

def mkSubstForCtrl(replacement_terms):
#   return list(zip(action['envVar'], list(map(toZ3Type, agent_action)))) #[(v,val) for (v,val) in zip(singleton['envVar'],agent_action_values)]
  return list(zip(action['controlVar'], replacement_terms)) 

def mkSubstForState(replacement_terms):
  return list(zip(state, replacement_terms))  #state is actually a list of Z3 structs repr. state vars

def postVarToPreVar(var_n_val):                             #eg (xX,6)
  var_as_expr = Real(var_n_val[0].name())                   #b/c xX is a FunctionRef!
  return (substitute(var_as_expr, inv_subst),var_n_val[1])  #eg (x,6)


#This just tests if the proposed action leads to safe immediate state, no guarantees about beyond..
#For model_1pt, faster to use the XXXOkPy tests in model_1pt
#***This code may not be correct? OK tests for the fitness of the action before taking the step
#eg ok([6,7],[0,10]) corresp to [(a_x,6), (a_y,7)][(x,0), (y,10), etc]
def ok(agent_action, curr_st):  
    new_st = updateState(agent_action, curr_st)           #--> [8.5, 3, 1.1, 2.3, etc]
    new_val_subst = mkSubstForState(new_st)               #--> [(x,8.5), (y,3), (vx,?), etc]
    for prop in safetyProps:
        instantiated_safetyProp = substitute(prop, new_val_subst) #eg L <= 8.5
        solver.reset()
        solver.add(instantiated_safetyProp)
        res = solver.check() 
        if res!=z3.sat:
        #     print('**action', agent_action, ' in state', curr_st, ' will cause violation of safety prop:',instantiated_safetyProp)
            return False
    return True 

def updateState(agent_action, curr_st):
  value_subst = \
    mkSubstForCtrl(agent_action) + mkSubstForState(curr_st)  #[(a_x,6), (a_y,0)] + [(x,10), etc]
#   print('value_subst=', value_subst)
  inst_action = substitute(action['actionPred'], value_subst) # --> 10 + 3 + 3 >= 0 
  # if DBG1: print('instantiated_action=\n',inst_action)
  solver.reset()
  solver.add(inst_action)
  solver.check()                                        #here check just being used to simplfy exprs
  m = solver.model()                                    #[a_xX=5, a_yX=6, xX=6, yX=12, etc]
  post_st = [m[var] for var in stateX]                  #[5, 6, 6, etc]
  # print('post_st = ', post_st)
  return post_st                                        #return (post_st,post_node) when we have multiple nodes
  
MAX_STEPS = 3
def runShield():
  print('\n-----------Running shield')
  # curr_st = [(x,5), (y,1), (v_x,0), (v_y,0)]            #python vars referencing z3 s
  curr_st = [5.0, 0.0, 0.0, 0.0]
  curr_st = list(map(toZ3Type, curr_st))
  print('initial state = (x,y,v_x,v_y)', curr_st) 

  # prey_action = [1.0, 0.0]                              #prey moves with constant speed
#   while not goalReached():
  for remaining_steps in range(MAX_STEPS,1,-1):
    agent_action = getNextAction(False)
    agent_action = list(map(toZ3Type, agent_action))
    print('\ngot next RL action ',agent_action)
    while not (ok(agent_action, curr_st, safetyProps)):
    # while not (ok2(np.array(agent_action), np.array(curr_st),
    #                np.array(prey_action), np.array(curr_st), 
    #                remaining_steps)):
      print('current agent_action leads to bad state. Requesting new action')
      agent_action = getNextAction(True)
      agent_action = list(map(toZ3Type, agent_action))
    # print('skipped updating state') 
    new_st = updateState(agent_action, curr_st)            #--> [(x,6), (y,12), etc]
    curr_st = new_st
    # remaining_steps = remaining_steps - 1


if __name__ == '__main__':
    # if DOING_BOUNDED:   buildBLShield()
    # else:               buildSafetyShield()
    buildSafetyShield()
#   runShield()  








'''
-------------------------------- Graveyard --------------------------------------------------
        #This was trying to strengthen the old variant  with the diff between old inv and the guardDij but inlike the ivnariant, the variant isn't monotonilcally stronger

        # stateInv = cdSimplifyE(stateInv).as_expr()
        #TBD this may be a no op in which case remove it
        # print("guardDisjunction/stateInv after CD simpligfy:\n", guardDisjunction)
        # stateInvDeltaX = substitute(stateInv, subst)
        # I think this is correct b/c surely the variant is reqd to establish the guard
        # variantDelta = residue(variant, guardDisjunction) # returns Goal list
        # print("guard disjunction - variant:", variantDelta)
        # if(len(variantDelta) > 0):  # have a refined state variant
        #     variant = cdSimplifyE(simplify(And(variant, *variantDelta))).as_expr()
        #     print("new State variant:\n", variant)
        #     #not used until the next iteration but need to do this here
        #     variantDeltaX = substitute(And(*variantDelta), subst)
        #     print("variantDeltaX:\n", variantDeltaX)
        #     # stateInvDeltaX = substitute(Or(at_tgt, And(*invariantDelta)), subst)
        #     # print("weakened stateInvDeltaX:\n", stateInvDeltaX)
        #     moreWork = True
        # # elif DBG1: print('state variant unchanged')
        # else:
        #   if  DBG1: print('state variant unchanged')

#*TBD : this should be testing the invariant, not the guard
def solutionExists(curr_st):
  guard = getGuard(action, 
                   action['precNode']['vars'] + action['controlVar'])   
  value_subst = \
    mkSubstForState(curr_st)  #[(a_x,6), (a_y,0)] + [(x,10), etc]
  instantiated_guard = substitute(guard, value_subst) # --> 10 + 3 + 3 >= 0 
  # print('instantiated_guard=\n',instantiated_guard); sys.stdout.flush() 
  solver.reset()
  solver.add(instantiated_guard)
  res = solver.check() 
  return res==z3.sat  
 
def boundedReachabilitySolutionExists(curr_st):
  value_subst = \
    mkSubstForState(curr_st)  #[(a_x,6), (a_y,0)] + [(x,10), etc]
  instantiated_variant = substitute(singleton['blprop'], value_subst) # --> 10 + 3 + 3 >= 0 
  # print('instantiated_guard=\n',instantiated_guard); sys.stdout.flush() 
  solver.reset()
  solver.add(instantiated_variant)
  res = solver.check() 
  return res==z3.sat  


-------------------------- graveyard ------------------------------------------------
'''
