from z3 import *

def printState0(i, expr):  # expr:Expression
    if is_and(expr):
        fs = expr.children()
        print("current state",i)
        for i in range(len(fs)):
            print(i, " ", fs[i])
    else:
        print("current state",i)
        print(i, " ", expr)

def printState(expr):  # expr:Expression
    if is_and(expr):
        fs = expr.children()
        for i in range(len(fs)):
            print("  ", i, " ", fs[i])
    else:
        print(expr)

def printList(es):  # es:List(Expression)
    for i in range(len(es)):
        print("  ", es[i])

def printTransition(act, exprs):  # act:String, expr:List(Expression)
    print("Transition:", act)
    #print(And(*exprs))
    i = 0
    for expr in exprs:
        if (expr == True): continue
        print("  ", i, " ", expr)
        i = i+1

# state:Expr, transitions:List(Distionary(Action))
def printModel0(state, stateInv, transitions):
    #print("\n-------------------------\nCurrent Model")
    print("State:", state)
    print("State Invariants")
    printState(stateInv)
    #print(stateInv)
    #print("Transitions")
    for tr in transitions:
        printTransition(tr['name'], expr2List(tr['controlPred']) + expr2List(tr['envPred']) + expr2List(tr['actionPred']))
    print()

def printModel(model): # state, stateInv, transitions):
    #print("\n-------------------------\nCurrent Model")
    for node in model['nodes']:
        print("Node:", node['name'])
        print("  Vars:", node['vars'])
        # this should print without holophrasting
        print("  Node Invariant:", node['invariant'])
        #print("  Node Invariant:", node['invariant'].sexpr())  # prints whole expr but not in Z3 syntax
    for act in model['transitions']:
        printTransition(act['name'],getControlPredL(act)
                                    + getEnvPredL(act)      #was expr2List(act['envPred'])
                                    + expr2List(act['actionPred']))
    print()

def isEmpty(xs): return len(xs) == 0

def getEnvPredL(act):
    if 'envPred' in act: return expr2List(act['envPred']) 
    else: return []

def getControlPredL(act):
    if 'controlPred' in act: return expr2List(act['controlPred']) 
    else: return []

def writeFunctionHeader(codeFile, fname, params):
    codeFile.write("\ndef " + fname + "ControlFn(")
    needSep = False
    for param in params:
        if needSep == True:
            codeFile.write(",")
        needSep = True
        codeFile.write(str(param))
    codeFile.write("):\n    ")

# doesn't work - how to form an equality formula?
def model2Expr(m):
    #print("model2Expr")
    result = True
    for d in m.decls():
        #print(d.name(), "==", m[d])
        result = And(Eq(d,m[d]), result)
        #print(result)
    #print(simplify(result))
    return simplify(result)

# code from ~Documents/Code/z3-master/build/python/z3/z3util.py
def vset(seq, idfun=None, as_list=True):
    def _uniq_normal(seq):
        d_ = {}
        for s in seq:
            if s not in d_:
                d_[s] = None
                yield s
 
    def _uniq_idfun(seq,idfun):
        d_ = {}
        for s in seq:
            h_ = idfun(s)
            if h_ not in d_:
                d_[h_] = None
                yield s
 
    if idfun is None:
        res = _uniq_normal(seq)
    else: 
        res = _uniq_idfun(seq,idfun)
 
    return list(res) if as_list else res 

def is_expr_val(v):
    return is_const(v) and v.decl().kind()!=Z3_OP_UNINTERPRETED

#   >>> get_vars(Implies(And(x+y==0,x*2==10),Or(a,Implies(a,b==False))))
#  [x, y, a, b]
def get_vars(f, rs = None):
    if rs is None:
        rs = []
    if is_const(f):
        if is_expr_val(f):
            return rs
        else:  #variable
            return vset(rs + [f],str)
    else:
        for f_ in f.children():
            rs = get_vars(f_, rs)
        return vset(rs, str)

def allVarsIn(vars1, vars2):
    print("avi:", vars1, vars2)
    for v in vars1:
        print("avi:", v, vars2, (v in vars2))
        if not(v in vars2):
            return False
    return True

# returns those exprs whose variable are all in vars
def filterConjunct(exprs, vars):
    result = []
    for conj in exprs:
        print("fC:", conj, allVarsIn(get_vars(conj), vars))
        if allVarsIn(get_vars(conj), vars):
            result.append(conj)
    return result

# vars, vars1: List(Expr) of same length
def makeSubstitution(vars, vars1):
    subst = nil
    for index in range(len(vars)):
        subst = cons((vars[index], vars1[index]), subst)
    return subst
    
# handle the case that the quantification is empty (z3 doesn't)
def makeForAll(vars, matrix):
    if len(vars) > 0:
        return ForAll(vars, matrix)
    else:
        return matrix

def makeExists(vars, matrix):
    if len(vars) > 0:
        return Exists(vars, matrix)
    else:
        return matrix

def is_action(expr, state1):
    vs = get_vars(expr)
    #print("expr vars:", vs, "state1 vars", state1)
    for var in state1:
        if var in vs:
            return True
    return False

def isStatePred(expr, state):  #print("isp?", expr)
    vs = get_vars(expr)        #print("expr vars:", vs)
    for var in vs:             #print("checking", var, "not in", state)
        # if not (var in state): #print("isp", False)
        #     return False
        #SN: hack for z3 bug in ==, has now been fixed
        found = False
        for other in state:
            # print("checking", var, "not eq", other)
            # used to be if var != other: return False but Z3 crashed. Bug has since been fixed
            if var.sort() == other.sort() and var == other:
                found = True
                break
        if not found: return False
    # print("isp", True)
    return True

def isNodePred(expr, node):
    # print("inp?", expr)
    vs = get_vars(expr)
    # print("expr vars:", vs)
    nodeVars = node['vars'] + node['globals']
    # print('nodeVars', nodeVars)
    for var in vs:
        #SN: hack for z3 bug in ==
        found = False
        for other in nodeVars:
            # print("checking", var, "not eq", other)
            if var.sort() == other.sort() and var == other:
                found = True
                break
        if not found: return False
        # if not (var in nodeVars):
        #     #print("isp", False)
        #     return False
    # print("iNP", True)
    return True

def isArcPred0(expr, transition):
    if isNodePred(expr, transition['precNode']) or isNodePred(expr, transition['postNode']):
        return False
    vs = get_vars(expr)
    precvars = transition['precNode']['vars']
    postvars = transition['postNode']['vars']
    for var in vs:
        if not(var in precvars or var in postvars):
            return False
    return True

def isArcPred2(expr, transition):
    if isNodePred(expr, transition['precNode']):
        return False
    else:
        return True

# in a multinode model
# True if expr mentions vars in both pre and post node
def isArcPred(expr, transition):
    resultPre  = isNodePred(expr, transition['precNode'])
    resultPost = isNodePred(expr, transition['postNode'])
    result = (not resultPre) and (not resultPost)
    return result

# convert g:Goal to a List(Expr)
def goal2List(g):
    if is_and(g.as_expr()):
        return g.as_expr().children()
    return [g.as_expr()]

# convert g:Expr to a List(Conjuncts/Expr)
# Beware: the output is ill-typed!!
def expr2List(ex):
    if is_and(ex):
        return ex.children()
    #if is_or(ex):  return ex.children()
    return [ex]

# decide forall(x) x in vs => x in vars
def allIn(vs, vars):
    for x in vs:
        if x not in vars:
            return False
    return True

# convert equality to a rewrite
def eq2rewrite(eq, qevars):
    #print("eq:", eq)
    #print("qevars:", qevars)
    if is_eq(eq):
        cs = eq.children()
        vs0 = get_vars(cs[0])
        if allIn(vs0, qevars):
            return (cs[0], cs[1])
        else:
            vs1 = get_vars(cs[1])
            if allIn(vs1, qevars):
                return (cs[1], cs[0])
    return

def selectionSort(input_list):
   for idx in range(len(input_list)):
      min_idx = idx
      for j in range( idx +1, len(input_list)):
         if input_list[min_idx] > input_list[j]:
             min_idx = j
# Swap the minimum value with the compared value
   input_list[idx], input_list[min_idx] = input_list[min_idx], input_list[idx]
