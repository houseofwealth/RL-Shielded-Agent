from z3 import *
from .utilities import *

# global counters
trueCount = int(0)
falseCount = int(0)
andCount = int(0)
orCount = int(0)
negCount = int(0)
atomCount = int(0)

def printCounts():
    print("trueCount =", trueCount)
    #print("falseCount =", falseCount)
    print("andCount =", andCount)
    print("orCount =", orCount)
    print("negCount =", negCount)
    print("atomCount =", atomCount)

# prove ante:Expr => conseq:Expr
# Return: unsat if implication holds, otw sat
def verifyImplies(ante, conseq):  
    s = Solver()
    s.add(ante)
    s.add(Not(conseq))
    #print("verifying", ante, "=>", conseq)
    result = s.check()
    if result == sat:
        print("***verification failed. Counterexample:", s.model())
    return result

# prove ante:Expr <=> conseq:Expr
# Return: True/False
def verfEquiv(ante, conseq):  
    result1 = verifyImplies(ante, conseq)
    #print("VE result1", result1)
    if result1 == unsat:
        result2 = verifyImplies(ante, conseq)
        return result2 == unsat
    else:
        return False

# is ante1 /\ ante2 /\ hyp unsat?
def noModels(ante1, ante2, hyp):  # ante1,ante2,hyp: Expression
    # print("SN noModels:\n--", ante1, "\n--", ante2, "\n--", hyp)
    chk = Solver()
    if is_and(ante1):
        #for h in ante1:
        for h in ante1.children():
            if (hyp.get_id() != h.get_id()):
                #print("ante1 h:", h)
                chk.add(h)
    else:
        if ante1: chk.add(ante1)
    if is_and(ante2):
        #for h in ante2:
        for h in ante2.children():
            if (hyp.get_id() != h.get_id()):
                #print("ante2 h:", h)
                chk.add(h)
    else:
        # print('adding ante2 to solver')
        # chk.add(ante2)
        for i in ante2:
          chk.add(i)
    # print("noModels check of ante1 /\ ante2:", chk.check())
    # breakpoint()
    chk.add(hyp)
    # print('added hyp')
    if unsat == chk.check(): # return (unsat == chk.check())
        return True
    return False

# remove atom f from the pool:Expression
def remove(pool, f):
    newPool = Goal()
    for g in pool:
        if (f.get_id() != g.get_id()):
            newPool.add(g)
    return newPool        

# To simplify Not(P), simplify P to r, then return Not(r)
# Assume: is_not(expr)
# context: List(Expr), expr:Expr
# returns e:Goal s.t. context => (expr == e)
def simplifyNegation(context, expr):
    negand =  expr.children()[0]
    # print("SN context:", context)
    # print("SN negand:", negand)
    simpNegand = cdSimplifyE_aux(context, negand).as_expr()
    #print("simpNegand:", simpNegand)
    result = simplify(Not(simpNegand), local_ctx=True) 
    #print("simplified not:", result)
    # double-check the result
    #equiv = verfEquiv(And(*context,expr), And(*context,result))
    #print("SN equiv check:", equiv)
    return result    

# Reformulate disjunction Or(P,Q) as Not(And(Not(P),Not(Q)))
# call simplifyConjunction on And(Not(P),Not(Q)) to r, then return Not(r)
# Assume: is_or(expr)
# context: List(Expr), expr:Expr
# returns e:Goal s.t. context => (expr == e)
def simplifyDisjunction(context, expr):
    disjuncts =  expr.children()   # List of expressions
    # print("SD context:", context)
    # print("SD disjuncts:", disjuncts)
    negDisjuncts = []
    for dis in disjuncts:
        negDisjuncts.append(simplify(Not(dis), local_ctx=True))
    #print("negated disjuncts", negDisjuncts)
    simpNegDisjunction = simplifyConjunction(context, And(*negDisjuncts))
    #print("simpNegDisjunction:", simpNegDisjunction)
    if is_and(simpNegDisjunction):
        simpDisjuncts = []
        for nd in simpNegDisjunction.children():
            simpDisjuncts.append(simplify(Not(nd), local_ctx=True))
        result = simplify(Or(*simpDisjuncts), local_ctx=True) 
    else:
        result = simplify(Not(simpNegDisjunction), local_ctx=True)
    #print("simplified disjunction:", result)
    return result    
        
# Assume: is_and(expr)
# for each conjunct c, simplify c wrt the other conjuncts.
# context: List(Expr), expr:Expr
# returns e:Goal s.t. context => (expr == e)
def simplifyConjunction(context, expr):
    #expr = varElimination(expr)
    conjuncts =  expr.children()   # unsimplified conjuncts
    # print("SC context:", context)
    # print("conjuncts:", conjuncts)
    simplifiedConjuncts = []   # local context
    while len(conjuncts) > 0:   # for h in conjuncts:
        c = conjuncts.pop(0)   # extract/delete first elt from the list
        #print("context", context)
        #print("target conjunct", c)
        #print("new context:", context + simplifiedConjuncts + conjuncts)
        cSimplified = cdSimplifyE_aux(context + simplifiedConjuncts + conjuncts, c).as_expr()
        #print("cSimplified:", cSimplified)
        simplifiedConjuncts.append(cSimplified)
    #print("SCL result:", simplifiedConjuncts)
    result = simplify(And(*simplifiedConjuncts), local_ctx=True) 
    #print("simplified conjunction:", result)
    return result

# if there is an equation var=val in a conjunction, or in LHS of implication,
# then eliminate it and replace all occurrences of var by val
# This treats x == x1 as a left-to-right rewrite
# TODO: use quantification over the var plus QE to achieve this
def varElimination (expr):
    if is_and(expr):
        #print("conjuncts:", expr.children())
        eqs = True
        for h in expr.children():
            #print(h)
            if is_eq(h) and is_const(h.children()[0]):  # TODO: handle other cases too
                var = h.children()[0]
                val = h.children()[1]
                #print("h:", var, "==", val)
                expr = simplify(substitute(expr, [(var,val)]))
                eqs = And(h, eqs)
        expr = simplify(And(eqs,expr))
        #print("varElim result:", expr)
    return expr


# context:List(Expr), expr:Expr
# returns e:Goal s.t. context => (expr == e)
def cdSimplifyE_aux(context, expr):   
    #global trueCount, falseCount
    #global andCount, orCount, negCount, atomCount
    # print("SN simp Aux context before:", context)
    context = expr2List(simplify(And(*context)))
    # print("SN simp aux context after:", context)
    #print("cdse_aux assume:")
    #printList(context) 
    #print("cdse_aux expr:", expr) 
    result = Goal()

    if noModels([], context, Not(expr)):
        #print("is True")
        #trueCount = trueCount + 1
        #print("trueCount =",trueCount)
        result.add(True)
        return result
    
# empirically not needed!?  see notes
#    if noModels([], context, expr):        #
#        print(expr, "is False")
#        falseCount = falseCount + 1
#        #print("falseCount =",falseCount)
#        result.add(False)
#        return result

    #print("cdse_aux expr2:", expr) 
    # double-check the result
    #equiv = verfEquiv(And(*context,expr), And(*context,expr2))
    #print("cd-simp1 equiv check:", equiv)

    if is_not(expr):
        simplifiedNot = simplifyNegation(context, expr)
        #negCount = negCount + 1
        #print("negCount =",negCount)
        result.add(simplifiedNot)

    elif is_and(expr):
        #andCount = andCount + 1
        #print("andCount =",andCount)
        simplifiedAnd = simplifyConjunction(context, expr)
        result.add(simplifiedAnd)

    elif is_or(expr):  # disjunction
        #orCount = orCount + 1
        #print("orCount =",orCount, expr)
        simplifiedOr = simplifyDisjunction(context, expr)
        result.add(simplifiedOr)

# atomic case;  do other cases arise?
    else:     
        #print("adding atom", expr)
        #atomCount = atomCount + 1
        #print("atomCount =",atomCount)
        result.add(expr)

    #print("cdse result:", result.as_expr())
    return result

# expr:Expr -> return:Goal
def cdSimplifyE(expr):   
    #print("cdse:", expr) 
    #expr = varElimination(expr)  too eager
    #print("cdse varElim:", expr)
    result = cdSimplifyE_aux([], expr)
    #print("cdse result:", result)
    return result
        
# return a Goal that is the conjunction of f in expr not implied by ante
def residue(ante, expr):   # ante,expr:Expression
    #print("find residue of", expr)
    #print("assuming",ante)
    if is_and(ante):
        result = cdSimplifyE_aux(ante.children(), expr)
    else:
        result = cdSimplifyE_aux([ante], expr)
    #print("result:", result)
    return result

