$ python shield.py
/cygdrive/c/Users/snedunu/Documents/Projects/FM/meher-experiment-with-different-Z3-shields-sn/meher/shields/Z3_shield/simplify.py:70: SyntaxWarning: invalid escape sequence '\ '
  print("noModels check of ante1 /\ ante2:", chk.check())
Building shield for Continuous Mountain Car
initializing node labels and arc labels with given safety properties..
SN noModels:
-- []
-- [True]
-- Not(Not(9/20 <= pos))
noModels chcked ante1
adding ante2 to solver
adding True
added
noModels check of ante1 /\ ante2: sat
added hyp
SN context: [True]
SN negand: 9/20 <= pos
SN noModels:
-- []
-- [True]
-- Not(9/20 <= pos)
noModels chcked ante1
adding ante2 to solver
adding True
added
noModels check of ante1 /\ ante2: sat
added hyp
SN noModels:
-- []
-- [True]
-- Not(Not(9/20 <= pos))
noModels chcked ante1
adding ante2 to solver
adding True
added
noModels check of ante1 /\ ante2: sat
added hyp
SN context: [True]
SN negand: 9/20 <= pos
SN noModels:
-- []
-- [True]
-- Not(9/20 <= pos)
noModels chcked ante1
adding ante2 to solver
adding True
added
noModels check of ante1 /\ ante2: sat
added hyp
- strengthened node label on node singleton to
 Not(9/20 <= pos)
..done

----------------
iteration 0

-- Refining the state invariant for node singleton
--- arc: action
wcp =
 ForAll([posX, velX],
       Implies(And(velX ==
                   vel + force*3/2000 -
                   1/400*(1 - (3*pos*3*pos)/2),
                   posX == pos + vel),
               Not(9/20 <= posX)))
wcp_simp =
 Not(9/20 <= pos + vel)
wcp_simp after QE (wcp_qe) =
Not(9/20 <= pos + vel)
strengthened ctrl pred
And(force >= -1, force <= 1, Not(9/20 <= pos + vel))
SN noModels:
-- []
-- [Not(9/20 <= pos)]
-- Not(And(force >= -1, force <= 1, Not(9/20 <= pos + vel)))
noModels chcked ante1
adding ante2 to solver
adding Not(9/20 <= pos)
added
noModels check of ante1 /\ ante2: sat
added hyp
SC context: [Not(9/20 <= pos)]
conjuncts: [force >= -1, force <= 1, Not(9/20 <= pos + vel)]
SN noModels:
-- []
-- [Not(9/20 <= pos), force <= 1, Not(9/20 <= pos + vel)]
-- Not(force >= -1)
noModels chcked ante1
adding ante2 to solver
adding Not(9/20 <= pos)
added
adding force <= 1
added
adding Not(9/20 <= pos + vel)
added
noModels check of ante1 /\ ante2: sat
added hyp
SN noModels:
-- []
-- [Not(9/20 <= pos), force >= -1, Not(9/20 <= pos + vel)]
-- Not(force <= 1)
noModels chcked ante1
adding ante2 to solver
adding Not(9/20 <= pos)
added
adding force >= -1
added
adding Not(9/20 <= pos + vel)
added
noModels check of ante1 /\ ante2: sat
added hyp
SN noModels:
-- []
-- [Not(9/20 <= pos), force >= -1, force <= 1]
-- Not(Not(9/20 <= pos + vel))
noModels chcked ante1
adding ante2 to solver
adding Not(9/20 <= pos)
added
adding force >= -1
added
adding force <= 1
added
noModels check of ante1 /\ ante2: sat
added hyp
SN context: [Not(9/20 <= pos), force >= -1, force <= 1]
SN negand: 9/20 <= pos + vel
SN noModels:
-- []
-- [Not(9/20 <= pos), force >= -1, force <= 1]
-- Not(9/20 <= pos + vel)
noModels chcked ante1
adding ante2 to solver
adding Not(9/20 <= pos)
added
adding force >= -1
added
adding force <= 1
added
noModels check of ante1 /\ ante2: sat
added hyp
SN noModels:
-- []
-- [force >= -1, force <= 1, Not(9/20 <= pos)]
-- Not(And(force >= -1, force <= 1, Not(9/20 <= pos + vel)))
noModels chcked ante1
adding ante2 to solver
adding force >= -1
added
adding force <= 1
added
adding Not(9/20 <= pos)
added
noModels check of ante1 /\ ante2: sat
added hyp
SC context: [force >= -1, force <= 1, Not(9/20 <= pos)]
conjuncts: [force >= -1, force <= 1, Not(9/20 <= pos + vel)]
SN noModels:
-- []
-- [force >= -1, force <= 1, Not(9/20 <= pos), Not(9/20 <= pos + vel)]
-- Not(force >= -1)
noModels chcked ante1
adding ante2 to solver
adding force >= -1
added
adding force <= 1
added
adding Not(9/20 <= pos)
added
adding Not(9/20 <= pos + vel)
added
noModels check of ante1 /\ ante2: sat
added hyp
SN noModels:
-- []
-- [force >= -1, force <= 1, Not(9/20 <= pos), Not(9/20 <= pos + vel)]
-- Not(force <= 1)
noModels chcked ante1
adding ante2 to solver
adding force >= -1
added
adding force <= 1
added
adding Not(9/20 <= pos)
added
adding Not(9/20 <= pos + vel)
added
noModels check of ante1 /\ ante2: sat
added hyp
SN noModels:
-- []
-- [force >= -1, force <= 1, Not(9/20 <= pos)]
-- Not(Not(9/20 <= pos + vel))
noModels chcked ante1
adding ante2 to solver
adding force >= -1
added
adding force <= 1
added
adding Not(9/20 <= pos)
added
noModels check of ante1 /\ ante2: sat
added hyp
SN context: [force >= -1, force <= 1, Not(9/20 <= pos)]
SN negand: 9/20 <= pos + vel
SN noModels:
-- []
-- [force >= -1, force <= 1, Not(9/20 <= pos)]
-- Not(9/20 <= pos + vel)
noModels chcked ante1
adding ante2 to solver
adding force >= -1
added
adding force <= 1
added
adding Not(9/20 <= pos)
added
noModels check of ante1 /\ ante2: sat
added hyp
new ctrl Pred (residue of ctrlPred over Inv) for action :
And(force >= -1, force <= 1, Not(9/20 <= pos + vel))
guardDisjunction: Or(False,
   Exists(force,
          And(force >= -1,
              force <= 1,
              Not(9/20 <= pos + vel))))
guardDisjunction after QE [Or(False, Not(9/20 <= pos + vel))]
simplified guardDisjunction:
 Not(9/20 <= pos + vel)

- Completed arcs from node singleton. Now checking if invariant needs updating (control preds may also get updated again)
stateInv
 Not(9/20 <= pos)
SN noModels:
-- []
-- [Not(9/20 <= pos)]
-- Not(Not(9/20 <= pos + vel))
noModels chcked ante1
adding ante2 to solver
adding Not(9/20 <= pos)
added
noModels check of ante1 /\ ante2: sat
added hyp
SN context: [Not(9/20 <= pos)]
SN negand: 9/20 <= pos + vel
SN noModels:
-- []
-- [Not(9/20 <= pos)]
-- Not(9/20 <= pos + vel)
noModels chcked ante1
adding ante2 to solver
adding Not(9/20 <= pos)
added
noModels check of ante1 /\ ante2: sat
added hyp
newInvDelta:
 [Not(9/20 <= pos + vel)]
SN noModels:
-- []
-- [True]
-- Not(And(Not(9/20 <= pos), Not(9/20 <= pos + vel)))
noModels chcked ante1
adding ante2 to solver
adding True
added
noModels check of ante1 /\ ante2: sat
added hyp
SC context: [True]
conjuncts: [Not(9/20 <= pos), Not(9/20 <= pos + vel)]
SN noModels:
-- []
-- [Not(9/20 <= pos + vel)]
-- Not(Not(9/20 <= pos))
noModels chcked ante1
adding ante2 to solver
adding Not(9/20 <= pos + vel)
added
noModels check of ante1 /\ ante2: sat
added hyp
SN context: [Not(9/20 <= pos + vel)]
SN negand: 9/20 <= pos
SN noModels:
-- []
-- [Not(9/20 <= pos + vel)]
-- Not(9/20 <= pos)
noModels chcked ante1
adding ante2 to solver
adding Not(9/20 <= pos + vel)
added
noModels check of ante1 /\ ante2: sat
added hyp
SN noModels:
-- []
-- [Not(9/20 <= pos)]
-- Not(Not(9/20 <= pos + vel))
noModels chcked ante1
adding ante2 to solver
adding Not(9/20 <= pos)
added
noModels check of ante1 /\ ante2: sat
added hyp
SN context: [Not(9/20 <= pos)]
SN negand: 9/20 <= pos + vel
SN noModels:
-- []
-- [Not(9/20 <= pos)]
-- Not(9/20 <= pos + vel)
noModels chcked ante1
adding ante2 to solver
adding Not(9/20 <= pos)
added
noModels check of ante1 /\ ante2: sat
added hyp
new invariant
 And(Not(9/20 <= pos), Not(9/20 <= pos + vel))
SN noModels:
-- []
-- [Not(9/20 <= pos), Not(9/20 <= pos + vel)]
-- Not(And(force >= -1, force <= 1, Not(9/20 <= pos + vel)))
noModels chcked ante1
adding ante2 to solver
adding Not(9/20 <= pos)
added
adding Not(9/20 <= pos + vel)
added
noModels check of ante1 /\ ante2: sat
added hyp
SC context: [Not(9/20 <= pos), Not(9/20 <= pos + vel)]
conjuncts: [force >= -1, force <= 1, Not(9/20 <= pos + vel)]
SN noModels:
-- []
-- [Not(9/20 <= pos), Not(9/20 <= pos + vel), force <= 1]
-- Not(force >= -1)
noModels chcked ante1
adding ante2 to solver
adding Not(9/20 <= pos)
added
adding Not(9/20 <= pos + vel)
added
adding force <= 1
added
noModels check of ante1 /\ ante2: sat
added hyp
SN noModels:
-- []
-- [Not(9/20 <= pos), Not(9/20 <= pos + vel), force >= -1]
-- Not(force <= 1)
noModels chcked ante1
adding ante2 to solver
adding Not(9/20 <= pos)
added
adding Not(9/20 <= pos + vel)
added
adding force >= -1
added
noModels check of ante1 /\ ante2: sat
added hyp
SN noModels:
-- []
-- [Not(9/20 <= pos), Not(9/20 <= pos + vel), force >= -1, force <= 1]
-- Not(Not(9/20 <= pos + vel))
noModels chcked ante1
adding ante2 to solver
adding Not(9/20 <= pos)
added
adding Not(9/20 <= pos + vel)
added
adding force >= -1
added
adding force <= 1
added
noModels check of ante1 /\ ante2: sat
added hyp
new ctrl Pred (diff b/w updated state inv and current ctrl pred):
 And(force >= -1, force <= 1)

----------------
iteration 1

-- Refining the state invariant for node singleton
--- arc: action
wcp =
 ForAll([posX, velX],
       Implies(And(velX ==
                   vel + force*3/2000 -
                   1/400*(1 - (3*pos*3*pos)/2),
                   posX == pos + vel),
               And(Not(9/20 <= posX),
                   Not(9/20 <= posX + velX))))
wcp_simp =
 And(Not(9/20 <= pos + vel),
    Not(181/400 <=
        pos + 2*vel + 3/2000*force + 9/800*pos*pos))
wcp_simp after QE (wcp_qe) =
And(Not(9/20 <= pos + vel),
    Not(181/400 <=
        pos + 2*vel + 3/2000*force + 9/800*pos*pos))
strengthened ctrl pred
And(force >= -1,
    force <= 1,
    Not(9/20 <= pos + vel),
    Not(181/400 <=
        pos + 2*vel + 3/2000*force + 9/800*pos*pos))
SN noModels:
-- []
-- [Not(9/20 <= pos), Not(9/20 <= pos + vel)]
-- Not(And(force >= -1,
        force <= 1,
        Not(9/20 <= pos + vel),
        Not(181/400 <=
            pos + 2*vel + 3/2000*force + 9/800*pos*pos)))
noModels chcked ante1
adding ante2 to solver
adding Not(9/20 <= pos)
added
adding Not(9/20 <= pos + vel)
added
noModels check of ante1 /\ ante2: sat
added hyp
SC context: [Not(9/20 <= pos), Not(9/20 <= pos + vel)]
conjuncts: [force >= -1, force <= 1, Not(9/20 <= pos + vel), Not(181/400 <= pos + 2*vel + 3/2000*force + 9/800*pos*pos)]
SN noModels:
-- []
-- [Not(9/20 <= pos), Not(9/20 <= pos + vel), force <= 1, Not(181/400 <= pos + 2*vel + 3/2000*force + 9/800*pos*pos)]
-- Not(force >= -1)
noModels chcked ante1
adding ante2 to solver
adding Not(9/20 <= pos)
added
adding Not(9/20 <= pos + vel)
added
adding force <= 1
added
adding Not(181/400 <= pos + 2*vel + 3/2000*force + 9/800*pos*pos)
added
noModels check of ante1 /\ ante2: sat
added hyp
SN noModels:
-- []
-- [Not(9/20 <= pos), Not(9/20 <= pos + vel), force >= -1, Not(181/400 <= pos + 2*vel + 3/2000*force + 9/800*pos*pos)]
-- Not(force <= 1)
noModels chcked ante1
adding ante2 to solver
adding Not(9/20 <= pos)
added
adding Not(9/20 <= pos + vel)
added
adding force >= -1
added
adding Not(181/400 <= pos + 2*vel + 3/2000*force + 9/800*pos*pos)
added
noModels check of ante1 /\ ante2: sat
added hyp
SN noModels:
-- []
-- [Not(9/20 <= pos), Not(9/20 <= pos + vel), force >= -1, force <= 1, Not(181/400 <= pos + 2*vel + 3/2000*force + 9/800*pos*pos)]
-- Not(Not(9/20 <= pos + vel))
noModels chcked ante1
adding ante2 to solver
adding Not(9/20 <= pos)
added
adding Not(9/20 <= pos + vel)
added
adding force >= -1
added
adding force <= 1
added
adding Not(181/400 <= pos + 2*vel + 3/2000*force + 9/800*pos*pos)
added
noModels check of ante1 /\ ante2: sat
added hyp
SN noModels:
-- []
-- [Not(9/20 <= pos), Not(9/20 <= pos + vel), force >= -1, force <= 1]
-- Not(Not(181/400 <=
        pos + 2*vel + 3/2000*force + 9/800*pos*pos))
noModels chcked ante1
adding ante2 to solver
adding Not(9/20 <= pos)
added
adding Not(9/20 <= pos + vel)
added
adding force >= -1
added
adding force <= 1
added
noModels check of ante1 /\ ante2: sat
added hyp
SN context: [Not(9/20 <= pos), Not(9/20 <= pos + vel), force >= -1, force <= 1]
SN negand: 181/400 <= pos + 2*vel + 3/2000*force + 9/800*pos*pos
SN noModels:
-- []
-- [Not(9/20 <= pos), Not(9/20 <= pos + vel), force >= -1, force <= 1]
-- Not(181/400 <= pos + 2*vel + 3/2000*force + 9/800*pos*pos)
noModels chcked ante1
adding ante2 to solver
adding Not(9/20 <= pos)
added
adding Not(9/20 <= pos + vel)
added
adding force >= -1
added
adding force <= 1
added
noModels check of ante1 /\ ante2: sat
added hyp
SN noModels:
-- []
-- [force >= -1, force <= 1, Not(9/20 <= pos), Not(9/20 <= pos + vel)]
-- Not(And(force >= -1,
        force <= 1,
        Not(181/400 <=
            pos + 2*vel + 3/2000*force + 9/800*pos*pos)))
noModels chcked ante1
adding ante2 to solver
adding force >= -1
added
adding force <= 1
added
adding Not(9/20 <= pos)
added
adding Not(9/20 <= pos + vel)
added
noModels check of ante1 /\ ante2: sat
added hyp
SC context: [force >= -1, force <= 1, Not(9/20 <= pos), Not(9/20 <= pos + vel)]
conjuncts: [force >= -1, force <= 1, Not(181/400 <= pos + 2*vel + 3/2000*force + 9/800*pos*pos)]
SN noModels:
-- []
-- [force >= -1, force <= 1, Not(9/20 <= pos), Not(9/20 <= pos + vel), Not(181/400 <= pos + 2*vel + 3/2000*force + 9/800*pos*pos)]
-- Not(force >= -1)
noModels chcked ante1
adding ante2 to solver
adding force >= -1
added
adding force <= 1
added
adding Not(9/20 <= pos)
added
adding Not(9/20 <= pos + vel)
added
adding Not(181/400 <= pos + 2*vel + 3/2000*force + 9/800*pos*pos)
added
noModels check of ante1 /\ ante2: sat
added hyp
SN noModels:
-- []
-- [force >= -1, force <= 1, Not(9/20 <= pos), Not(9/20 <= pos + vel), Not(181/400 <= pos + 2*vel + 3/2000*force + 9/800*pos*pos)]
-- Not(force <= 1)
noModels chcked ante1
adding ante2 to solver
adding force >= -1
added
adding force <= 1
added
adding Not(9/20 <= pos)
added
adding Not(9/20 <= pos + vel)
added
adding Not(181/400 <= pos + 2*vel + 3/2000*force + 9/800*pos*pos)
added
noModels check of ante1 /\ ante2: sat
added hyp
SN noModels:
-- []
-- [force >= -1, force <= 1, Not(9/20 <= pos), Not(9/20 <= pos + vel)]
-- Not(Not(181/400 <=
        pos + 2*vel + 3/2000*force + 9/800*pos*pos))
noModels chcked ante1
adding ante2 to solver
adding force >= -1
added
adding force <= 1
added
adding Not(9/20 <= pos)
added
adding Not(9/20 <= pos + vel)
added
noModels check of ante1 /\ ante2: sat
added hyp
SN context: [force >= -1, force <= 1, Not(9/20 <= pos), Not(9/20 <= pos + vel)]
SN negand: 181/400 <= pos + 2*vel + 3/2000*force + 9/800*pos*pos
SN noModels:
-- []
-- [force >= -1, force <= 1, Not(9/20 <= pos), Not(9/20 <= pos + vel)]
-- Not(181/400 <= pos + 2*vel + 3/2000*force + 9/800*pos*pos)
noModels chcked ante1
adding ante2 to solver
adding force >= -1
added
adding force <= 1
added
adding Not(9/20 <= pos)
added
adding Not(9/20 <= pos + vel)
added
noModels check of ante1 /\ ante2: sat
added hyp
new ctrl Pred (residue of ctrlPred over Inv) for action :
And(force >= -1,
    force <= 1,
    Not(181/400 <=
        pos + 2*vel + 3/2000*force + 9/800*pos*pos))
guardDisjunction: Or(False,
   Exists(force,
          And(force >= -1,
              force <= 1,
              Not(181/400 <=
                  pos + 2*vel + 3/2000*force + 9/800*pos*pos))))
guardDisjunction after QE [Or(False,
    Not(908/3 <= 2000/3*pos + 15/2*pos*pos + 4000/3*vel))]
simplified guardDisjunction:
 Not(908/3 <= 2000/3*pos + 15/2*pos*pos + 4000/3*vel)

- Completed arcs from node singleton. Now checking if invariant needs updating (control preds may also get updated again)
stateInv
 And(Not(9/20 <= pos), Not(9/20 <= pos + vel))
SN noModels:
-- []
-- [Not(9/20 <= pos), Not(9/20 <= pos + vel)]
-- Not(Not(908/3 <= 2000/3*pos + 15/2*pos*pos + 4000/3*vel))
noModels chcked ante1
adding ante2 to solver
adding Not(9/20 <= pos)
added
adding Not(9/20 <= pos + vel)
added
noModels check of ante1 /\ ante2: sat
added hyp
SN context: [Not(9/20 <= pos), Not(9/20 <= pos + vel)]
SN negand: 908/3 <= 2000/3*pos + 15/2*pos*pos + 4000/3*vel
SN noModels:
-- []
-- [Not(9/20 <= pos), Not(9/20 <= pos + vel)]
-- Not(908/3 <= 2000/3*pos + 15/2*pos*pos + 4000/3*vel)
noModels chcked ante1
adding ante2 to solver
adding Not(9/20 <= pos)
added
adding Not(9/20 <= pos + vel)
added
noModels check of ante1 /\ ante2: sat
added hyp
newInvDelta:
 [Not(908/3 <= 2000/3*pos + 15/2*pos*pos + 4000/3*vel)]
SN noModels:
-- []
-- [True]
-- Not(And(And(Not(9/20 <= pos), Not(9/20 <= pos + vel)),
        Not(908/3 <= 2000/3*pos + 15/2*pos*pos + 4000/3*vel)))
noModels chcked ante1
adding ante2 to solver
adding True
added
noModels check of ante1 /\ ante2: sat
added hyp
SC context: [True]
conjuncts: [And(Not(9/20 <= pos), Not(9/20 <= pos + vel)), Not(908/3 <= 2000/3*pos + 15/2*pos*pos + 4000/3*vel)]
SN noModels:
-- []
-- [Not(908/3 <= 2000/3*pos + 15/2*pos*pos + 4000/3*vel)]
-- Not(And(Not(9/20 <= pos), Not(9/20 <= pos + vel)))
noModels chcked ante1
adding ante2 to solver
adding Not(908/3 <= 2000/3*pos + 15/2*pos*pos + 4000/3*vel)
added
noModels check of ante1 /\ ante2: sat
added hyp
SC context: [Not(908/3 <= 2000/3*pos + 15/2*pos*pos + 4000/3*vel)]
conjuncts: [Not(9/20 <= pos), Not(9/20 <= pos + vel)]
SN noModels:
-- []
-- [Not(908/3 <= 2000/3*pos + 15/2*pos*pos + 4000/3*vel), Not(9/20 <= pos + vel)]
-- Not(Not(9/20 <= pos))
noModels chcked ante1
adding ante2 to solver
adding Not(908/3 <= 2000/3*pos + 15/2*pos*pos + 4000/3*vel)
added
adding Not(9/20 <= pos + vel)
added
noModels check of ante1 /\ ante2: sat
added hyp
SN context: [Not(908/3 <= 2000/3*pos + 15/2*pos*pos + 4000/3*vel), Not(9/20 <= pos + vel)]
SN negand: 9/20 <= pos
SN noModels:
-- []
-- [Not(908/3 <= 2000/3*pos + 15/2*pos*pos + 4000/3*vel), Not(9/20 <= pos + vel)]
-- Not(9/20 <= pos)
noModels chcked ante1
adding ante2 to solver
adding Not(908/3 <= 2000/3*pos + 15/2*pos*pos + 4000/3*vel)
added
adding Not(9/20 <= pos + vel)
added
noModels check of ante1 /\ ante2: sat
added hyp
SN noModels:
-- []
-- [Not(908/3 <= 2000/3*pos + 15/2*pos*pos + 4000/3*vel), Not(9/20 <= pos)]
-- Not(Not(9/20 <= pos + vel))
noModels chcked ante1
adding ante2 to solver
adding Not(908/3 <= 2000/3*pos + 15/2*pos*pos + 4000/3*vel)
added
adding Not(9/20 <= pos)
added
noModels check of ante1 /\ ante2: sat
added hyp
SN context: [Not(908/3 <= 2000/3*pos + 15/2*pos*pos + 4000/3*vel), Not(9/20 <= pos)]
SN negand: 9/20 <= pos + vel
SN noModels:
-- []
-- [Not(908/3 <= 2000/3*pos + 15/2*pos*pos + 4000/3*vel), Not(9/20 <= pos)]
-- Not(9/20 <= pos + vel)
noModels chcked ante1
adding ante2 to solver
adding Not(908/3 <= 2000/3*pos + 15/2*pos*pos + 4000/3*vel)
added
adding Not(9/20 <= pos)
added
noModels check of ante1 /\ ante2: sat
added hyp
SN noModels:
-- []
-- [Not(9/20 <= pos), Not(9/20 <= pos + vel)]
-- Not(Not(908/3 <= 2000/3*pos + 15/2*pos*pos + 4000/3*vel))
noModels chcked ante1
adding ante2 to solver
adding Not(9/20 <= pos)
added
adding Not(9/20 <= pos + vel)
added
noModels check of ante1 /\ ante2: sat
added hyp
SN context: [Not(9/20 <= pos), Not(9/20 <= pos + vel)]
SN negand: 908/3 <= 2000/3*pos + 15/2*pos*pos + 4000/3*vel
SN noModels:
-- []
-- [Not(9/20 <= pos), Not(9/20 <= pos + vel)]
-- Not(908/3 <= 2000/3*pos + 15/2*pos*pos + 4000/3*vel)
noModels chcked ante1
adding ante2 to solver
adding Not(9/20 <= pos)
added
adding Not(9/20 <= pos + vel)
added
noModels check of ante1 /\ ante2: sat
added hyp
new invariant
 And(Not(9/20 <= pos),
    Not(9/20 <= pos + vel),
    Not(908/3 <= 2000/3*pos + 15/2*pos*pos + 4000/3*vel))
SN noModels:
-- []
-- [Not(9/20 <= pos), Not(9/20 <= pos + vel), Not(908/3 <= 2000/3*pos + 15/2*pos*pos + 4000/3*vel)]
-- Not(And(force >= -1,
        force <= 1,
        Not(181/400 <=
            pos + 2*vel + 3/2000*force + 9/800*pos*pos)))
noModels chcked ante1
adding ante2 to solver
adding Not(9/20 <= pos)
added
adding Not(9/20 <= pos + vel)
added
adding Not(908/3 <= 2000/3*pos + 15/2*pos*pos + 4000/3*vel)
added
noModels check of ante1 /\ ante2: sat
added hyp
SC context: [Not(9/20 <= pos), Not(9/20 <= pos + vel), Not(908/3 <= 2000/3*pos + 15/2*pos*pos + 4000/3*vel)]
conjuncts: [force >= -1, force <= 1, Not(181/400 <= pos + 2*vel + 3/2000*force + 9/800*pos*pos)]
SN noModels:
-- []
-- [Not(9/20 <= pos), Not(9/20 <= pos + vel), Not(908/3 <= 2000/3*pos + 15/2*pos*pos + 4000/3*vel), force <= 1, Not(181/400 <= pos + 2*vel + 3/2000*force + 9/800*pos*pos)]
-- Not(force >= -1)
noModels chcked ante1
adding ante2 to solver
adding Not(9/20 <= pos)
added
adding Not(9/20 <= pos + vel)
added
adding Not(908/3 <= 2000/3*pos + 15/2*pos*pos + 4000/3*vel)
added
adding force <= 1
added
adding Not(181/400 <= pos + 2*vel + 3/2000*force + 9/800*pos*pos)
added
noModels check of ante1 /\ ante2: sat
added hyp
SN noModels:
-- []
-- [Not(9/20 <= pos), Not(9/20 <= pos + vel), Not(908/3 <= 2000/3*pos + 15/2*pos*pos + 4000/3*vel), force >= -1, Not(181/400 <= pos + 2*vel + 3/2000*force + 9/800*pos*pos)]
-- Not(force <= 1)
noModels chcked ante1
adding ante2 to solver
adding Not(9/20 <= pos)
added
adding Not(9/20 <= pos + vel)
added
adding Not(908/3 <= 2000/3*pos + 15/2*pos*pos + 4000/3*vel)
added
adding force >= -1
added
adding Not(181/400 <= pos + 2*vel + 3/2000*force + 9/800*pos*pos)
added
noModels check of ante1 /\ ante2: sat
added hyp
SN noModels:
-- []
-- [Not(9/20 <= pos), Not(9/20 <= pos + vel), Not(908/3 <= 2000/3*pos + 15/2*pos*pos + 4000/3*vel), force >= -1, force <= 1]
-- Not(Not(181/400 <=
        pos + 2*vel + 3/2000*force + 9/800*pos*pos))
noModels chcked ante1
adding ante2 to solver
adding Not(9/20 <= pos)
added
adding Not(9/20 <= pos + vel)
added
adding Not(908/3 <= 2000/3*pos + 15/2*pos*pos + 4000/3*vel)
added
adding force >= -1
added
adding force <= 1
added
noModels check of ante1 /\ ante2: sat
added hyp
SN context: [Not(9/20 <= pos), Not(9/20 <= pos + vel), Not(908/3 <= 2000/3*pos + 15/2*pos*pos + 4000/3*vel), force >= -1, force <= 1]
SN negand: 181/400 <= pos + 2*vel + 3/2000*force + 9/800*pos*pos
SN noModels:
-- []
-- [Not(9/20 <= pos), Not(9/20 <= pos + vel), Not(908/3 <= 2000/3*pos + 15/2*pos*pos + 4000/3*vel), force >= -1, force <= 1]
-- Not(181/400 <= pos + 2*vel + 3/2000*force + 9/800*pos*pos)
noModels chcked ante1
adding ante2 to solver
adding Not(9/20 <= pos)
added
adding Not(9/20 <= pos + vel)
added
adding Not(908/3 <= 2000/3*pos + 15/2*pos*pos + 4000/3*vel)
added
adding force >= -1
added
adding force <= 1
added
noModels check of ante1 /\ ante2: sat
added hyp
new ctrl Pred (diff b/w updated state inv and current ctrl pred):
 And(force >= -1,
    force <= 1,
    Not(181/400 <=
        pos + 2*vel + 3/2000*force + 9/800*pos*pos))

----------------
iteration 2

-- Refining the state invariant for node singleton
--- arc: action
wcp =
 ForAll([posX, velX],
       Implies(And(velX ==
                   vel + force*3/2000 -
                   1/400*(1 - (3*pos*3*pos)/2),
                   posX == pos + vel),
               And(Not(9/20 <= posX),
                   Not(9/20 <= posX + velX),
                   Not(908/3 <=
                       2000/3*posX +
                       15/2*posX*posX +
                       4000/3*velX))))
wcp_simp =
 And(Not(9/20 <= pos + vel),
    Not(181/400 <=
        pos + 2*vel + 3/2000*force + 9/800*pos*pos),
    Not(306 <=
        2000/3*pos +
        2000*vel +
        15/2*(pos + vel)*(pos + vel) +
        2*force +
        15*pos*pos))
wcp_simp after QE (wcp_qe) =
And(Not(9/20 <= pos + vel),
    Not(181/400 <=
        pos + 2*vel + 3/2000*force + 9/800*pos*pos),
    Not(306 <=
        2000/3*pos +
        2000*vel +
        15/2*(pos + vel)*(pos + vel) +
        2*force +
        15*pos*pos))
strengthened ctrl pred
And(force >= -1,
    force <= 1,
    Not(181/400 <=
        pos + 2*vel + 3/2000*force + 9/800*pos*pos),
    Not(9/20 <= pos + vel),
    Not(306 <=
        2000/3*pos +
        2000*vel +
        15/2*(pos + vel)*(pos + vel) +
        2*force +
        15*pos*pos))
SN noModels:
-- []
-- [Not(9/20 <= pos), Not(9/20 <= pos + vel), Not(908/3 <= 2000/3*pos + 15/2*pos*pos + 4000/3*vel)]
-- Not(And(force >= -1,
        force <= 1,
        Not(181/400 <=
            pos + 2*vel + 3/2000*force + 9/800*pos*pos),
        Not(9/20 <= pos + vel),
        Not(306 <=
            2000/3*pos +
            2000*vel +
            15/2*(pos + vel)*(pos + vel) +
            2*force +
            15*pos*pos)))
noModels chcked ante1
adding ante2 to solver
adding Not(9/20 <= pos)
added
adding Not(9/20 <= pos + vel)
added
adding Not(908/3 <= 2000/3*pos + 15/2*pos*pos + 4000/3*vel)
added
noModels check of ante1 /\ ante2: sat
added hyp
SC context: [Not(9/20 <= pos), Not(9/20 <= pos + vel), Not(908/3 <= 2000/3*pos + 15/2*pos*pos + 4000/3*vel)]
conjuncts: [force >= -1, force <= 1, Not(181/400 <= pos + 2*vel + 3/2000*force + 9/800*pos*pos), Not(9/20 <= pos + vel), Not(306 <=
    2000/3*pos +
    2000*vel +
    15/2*(pos + vel)*(pos + vel) +
    2*force +
    15*pos*pos)]
SN noModels:
-- []
-- [Not(9/20 <= pos), Not(9/20 <= pos + vel), Not(908/3 <= 2000/3*pos + 15/2*pos*pos + 4000/3*vel), force <= 1, Not(181/400 <= pos + 2*vel + 3/2000*force + 9/800*pos*pos), Not(306 <=
    2000/3*pos +
    2000*vel +
    15/2*(pos + vel)*(pos + vel) +
    2*force +
    15*pos*pos)]
-- Not(force >= -1)
noModels chcked ante1
adding ante2 to solver
adding Not(9/20 <= pos)
added
adding Not(9/20 <= pos + vel)
added
adding Not(908/3 <= 2000/3*pos + 15/2*pos*pos + 4000/3*vel)
added
adding force <= 1
added
adding Not(181/400 <= pos + 2*vel + 3/2000*force + 9/800*pos*pos)
added
adding Not(306 <=
    2000/3*pos +
    2000*vel +
    15/2*(pos + vel)*(pos + vel) +
    2*force +
    15*pos*pos)
added
noModels check of ante1 /\ ante2: sat
added hyp
SN noModels:
-- []
-- [Not(9/20 <= pos), Not(9/20 <= pos + vel), Not(908/3 <= 2000/3*pos + 15/2*pos*pos + 4000/3*vel), force >= -1, Not(181/400 <= pos + 2*vel + 3/2000*force + 9/800*pos*pos), Not(306 <=
    2000/3*pos +
    2000*vel +
    15/2*(pos + vel)*(pos + vel) +
    2*force +
    15*pos*pos)]
-- Not(force <= 1)
noModels chcked ante1
adding ante2 to solver
adding Not(9/20 <= pos)
added
adding Not(9/20 <= pos + vel)
added
adding Not(908/3 <= 2000/3*pos + 15/2*pos*pos + 4000/3*vel)
added
adding force >= -1
added
adding Not(181/400 <= pos + 2*vel + 3/2000*force + 9/800*pos*pos)
added
adding Not(306 <=
    2000/3*pos +
    2000*vel +
    15/2*(pos + vel)*(pos + vel) +
    2*force +
    15*pos*pos)
added
noModels check of ante1 /\ ante2: sat
added hyp
SN noModels:
-- []
-- [Not(9/20 <= pos), Not(9/20 <= pos + vel), Not(908/3 <= 2000/3*pos + 15/2*pos*pos + 4000/3*vel), force >= -1, force <= 1, Not(306 <=
    2000/3*pos +
    2000*vel +
    15/2*(pos + vel)*(pos + vel) +
    2*force +
    15*pos*pos)]
-- Not(Not(181/400 <=
        pos + 2*vel + 3/2000*force + 9/800*pos*pos))
noModels chcked ante1
adding ante2 to solver
adding Not(9/20 <= pos)
added
adding Not(9/20 <= pos + vel)
added
adding Not(908/3 <= 2000/3*pos + 15/2*pos*pos + 4000/3*vel)
added
adding force >= -1
added
adding force <= 1
added
adding Not(306 <=
    2000/3*pos +
    2000*vel +
    15/2*(pos + vel)*(pos + vel) +
    2*force +
    15*pos*pos)
added
noModels check of ante1 /\ ante2: sat
added hyp
SN context: [Not(9/20 <= pos), Not(9/20 <= pos + vel), Not(908/3 <= 2000/3*pos + 15/2*pos*pos + 4000/3*vel), force >= -1, force <= 1, Not(306 <=
    2000/3*pos +
    2000*vel +
    15/2*(pos + vel)*(pos + vel) +
    2*force +
    15*pos*pos)]
SN negand: 181/400 <= pos + 2*vel + 3/2000*force + 9/800*pos*pos
SN noModels:
-- []
-- [Not(9/20 <= pos), Not(9/20 <= pos + vel), Not(908/3 <= 2000/3*pos + 15/2*pos*pos + 4000/3*vel), force >= -1, force <= 1, Not(306 <=
    2000/3*pos +
    2000*vel +
    15/2*(pos + vel)*(pos + vel) +
    2*force +
    15*pos*pos)]
-- Not(181/400 <= pos + 2*vel + 3/2000*force + 9/800*pos*pos)
noModels chcked ante1
adding ante2 to solver
adding Not(9/20 <= pos)
added
adding Not(9/20 <= pos + vel)
added
adding Not(908/3 <= 2000/3*pos + 15/2*pos*pos + 4000/3*vel)
added
adding force >= -1
added
adding force <= 1
added
adding Not(306 <=
    2000/3*pos +
    2000*vel +
    15/2*(pos + vel)*(pos + vel) +
    2*force +
    15*pos*pos)
added
noModels check of ante1 /\ ante2: sat
added hyp
SN noModels:
-- []
-- [Not(9/20 <= pos), Not(9/20 <= pos + vel), Not(908/3 <= 2000/3*pos + 15/2*pos*pos + 4000/3*vel), force >= -1, force <= 1, Not(181/400 <= pos + 2*vel + 3/2000*force + 9/800*pos*pos), Not(306 <=
    2000/3*pos +
    2000*vel +
    15/2*(pos + vel)*(pos + vel) +
    2*force +
    15*pos*pos)]
-- Not(Not(9/20 <= pos + vel))
noModels chcked ante1
adding ante2 to solver
adding Not(9/20 <= pos)
added
adding Not(9/20 <= pos + vel)
added
adding Not(908/3 <= 2000/3*pos + 15/2*pos*pos + 4000/3*vel)
added
adding force >= -1
added
adding force <= 1
added
adding Not(181/400 <= pos + 2*vel + 3/2000*force + 9/800*pos*pos)
added
adding Not(306 <=
    2000/3*pos +
    2000*vel +
    15/2*(pos + vel)*(pos + vel) +
    2*force +
    15*pos*pos)
added
noModels check of ante1 /\ ante2: sat
added hyp
SN noModels:
-- []
-- [Not(9/20 <= pos), Not(9/20 <= pos + vel), Not(908/3 <= 2000/3*pos + 15/2*pos*pos + 4000/3*vel), force >= -1, force <= 1, Not(181/400 <= pos + 2*vel + 3/2000*force + 9/800*pos*pos)]
-- Not(Not(306 <=
        2000/3*pos +
        2000*vel +
        15/2*(pos + vel)*(pos + vel) +
        2*force +
        15*pos*pos))
noModels chcked ante1
adding ante2 to solver
adding Not(9/20 <= pos)
added
adding Not(9/20 <= pos + vel)
added
adding Not(908/3 <= 2000/3*pos + 15/2*pos*pos + 4000/3*vel)
added
adding force >= -1
added
adding force <= 1
added
adding Not(181/400 <= pos + 2*vel + 3/2000*force + 9/800*pos*pos)
added
noModels check of ante1 /\ ante2: sat
added hyp
SN context: [Not(9/20 <= pos), Not(9/20 <= pos + vel), Not(908/3 <= 2000/3*pos + 15/2*pos*pos + 4000/3*vel), force >= -1, force <= 1, Not(181/400 <= pos + 2*vel + 3/2000*force + 9/800*pos*pos)]
SN negand: 306 <=
2000/3*pos +
2000*vel +
15/2*(pos + vel)*(pos + vel) +
2*force +
15*pos*pos
SN noModels:
-- []
-- [Not(9/20 <= pos), Not(9/20 <= pos + vel), Not(908/3 <= 2000/3*pos + 15/2*pos*pos + 4000/3*vel), force >= -1, force <= 1, Not(181/400 <= pos + 2*vel + 3/2000*force + 9/800*pos*pos)]
-- Not(306 <=
    2000/3*pos +
    2000*vel +
    15/2*(pos + vel)*(pos + vel) +
    2*force +
    15*pos*pos)
noModels chcked ante1
adding ante2 to solver
adding Not(9/20 <= pos)
added
adding Not(9/20 <= pos + vel)
added
adding Not(908/3 <= 2000/3*pos + 15/2*pos*pos + 4000/3*vel)
added
adding force >= -1
added
adding force <= 1
added
adding Not(181/400 <= pos + 2*vel + 3/2000*force + 9/800*pos*pos)
added
noModels check of ante1 /\ ante2: sat
added hyp
SN noModels:
-- []
-- [force >= -1, force <= 1, Not(181/400 <= pos + 2*vel + 3/2000*force + 9/800*pos*pos), Not(9/20 <= pos), Not(9/20 <= pos + vel), Not(908/3 <= 2000/3*pos + 15/2*pos*pos + 4000/3*vel)]
-- Not(And(force >= -1,
        force <= 1,
        Not(181/400 <=
            pos + 2*vel + 3/2000*force + 9/800*pos*pos),
        Not(306 <=
            2000/3*pos +
            2000*vel +
            15/2*(pos + vel)*(pos + vel) +
            2*force +
            15*pos*pos)))
noModels chcked ante1
adding ante2 to solver
adding force >= -1
added
adding force <= 1
added
adding Not(181/400 <= pos + 2*vel + 3/2000*force + 9/800*pos*pos)
added
adding Not(9/20 <= pos)
added
adding Not(9/20 <= pos + vel)
added
adding Not(908/3 <= 2000/3*pos + 15/2*pos*pos + 4000/3*vel)
added
noModels check of ante1 /\ ante2: sat
added hyp
SC context: [force >= -1, force <= 1, Not(181/400 <= pos + 2*vel + 3/2000*force + 9/800*pos*pos), Not(9/20 <= pos), Not(9/20 <= pos + vel), Not(908/3 <= 2000/3*pos + 15/2*pos*pos + 4000/3*vel)]
conjuncts: [force >= -1, force <= 1, Not(181/400 <= pos + 2*vel + 3/2000*force + 9/800*pos*pos), Not(306 <=
    2000/3*pos +
    2000*vel +
    15/2*(pos + vel)*(pos + vel) +
    2*force +
    15*pos*pos)]
SN noModels:
-- []
-- [force >= -1, force <= 1, Not(181/400 <= pos + 2*vel + 3/2000*force + 9/800*pos*pos), Not(9/20 <= pos), Not(9/20 <= pos + vel), Not(908/3 <= 2000/3*pos + 15/2*pos*pos + 4000/3*vel), Not(306 <=
    2000/3*pos +
    2000*vel +
    15/2*(pos + vel)*(pos + vel) +
    2*force +
    15*pos*pos)]
-- Not(force >= -1)
noModels chcked ante1
adding ante2 to solver
adding force >= -1
added
adding force <= 1
added
adding Not(181/400 <= pos + 2*vel + 3/2000*force + 9/800*pos*pos)
added
adding Not(9/20 <= pos)
added
adding Not(9/20 <= pos + vel)
added
adding Not(908/3 <= 2000/3*pos + 15/2*pos*pos + 4000/3*vel)
added
adding Not(306 <=
    2000/3*pos +
    2000*vel +
    15/2*(pos + vel)*(pos + vel) +
    2*force +
    15*pos*pos)
added
noModels check of ante1 /\ ante2: sat
added hyp
SN noModels:
-- []
-- [force >= -1, force <= 1, Not(181/400 <= pos + 2*vel + 3/2000*force + 9/800*pos*pos), Not(9/20 <= pos), Not(9/20 <= pos + vel), Not(908/3 <= 2000/3*pos + 15/2*pos*pos + 4000/3*vel), Not(306 <=
    2000/3*pos +
    2000*vel +
    15/2*(pos + vel)*(pos + vel) +
    2*force +
    15*pos*pos)]
-- Not(force <= 1)
noModels chcked ante1
adding ante2 to solver
adding force >= -1
added
adding force <= 1
added
adding Not(181/400 <= pos + 2*vel + 3/2000*force + 9/800*pos*pos)
added
adding Not(9/20 <= pos)
added
adding Not(9/20 <= pos + vel)
added
adding Not(908/3 <= 2000/3*pos + 15/2*pos*pos + 4000/3*vel)
added
adding Not(306 <=
    2000/3*pos +
    2000*vel +
    15/2*(pos + vel)*(pos + vel) +
    2*force +
    15*pos*pos)
added
noModels check of ante1 /\ ante2: sat
added hyp
SN noModels:
-- []
-- [force >= -1, force <= 1, Not(181/400 <= pos + 2*vel + 3/2000*force + 9/800*pos*pos), Not(9/20 <= pos), Not(9/20 <= pos + vel), Not(908/3 <= 2000/3*pos + 15/2*pos*pos + 4000/3*vel), Not(306 <=
    2000/3*pos +
    2000*vel +
    15/2*(pos + vel)*(pos + vel) +
    2*force +
    15*pos*pos)]
-- Not(Not(181/400 <=
        pos + 2*vel + 3/2000*force + 9/800*pos*pos))
noModels chcked ante1
adding ante2 to solver
adding force >= -1
added
adding force <= 1
added
adding Not(181/400 <= pos + 2*vel + 3/2000*force + 9/800*pos*pos)
added
adding Not(9/20 <= pos)
added
adding Not(9/20 <= pos + vel)
added
adding Not(908/3 <= 2000/3*pos + 15/2*pos*pos + 4000/3*vel)
added
adding Not(306 <=
    2000/3*pos +
    2000*vel +
    15/2*(pos + vel)*(pos + vel) +
    2*force +
    15*pos*pos)
added
noModels check of ante1 /\ ante2: sat
added hyp
SN noModels:
-- []
-- [force >= -1, force <= 1, Not(181/400 <= pos + 2*vel + 3/2000*force + 9/800*pos*pos), Not(9/20 <= pos), Not(9/20 <= pos + vel), Not(908/3 <= 2000/3*pos + 15/2*pos*pos + 4000/3*vel)]
-- Not(Not(306 <=
        2000/3*pos +
        2000*vel +
        15/2*(pos + vel)*(pos + vel) +
        2*force +
        15*pos*pos))
noModels chcked ante1
adding ante2 to solver
adding force >= -1
added
adding force <= 1
added
adding Not(181/400 <= pos + 2*vel + 3/2000*force + 9/800*pos*pos)
added
adding Not(9/20 <= pos)
added
adding Not(9/20 <= pos + vel)
added
adding Not(908/3 <= 2000/3*pos + 15/2*pos*pos + 4000/3*vel)
added
noModels check of ante1 /\ ante2: sat
added hyp
SN context: [force >= -1, force <= 1, Not(181/400 <= pos + 2*vel + 3/2000*force + 9/800*pos*pos), Not(9/20 <= pos), Not(9/20 <= pos + vel), Not(908/3 <= 2000/3*pos + 15/2*pos*pos + 4000/3*vel)]
SN negand: 306 <=
2000/3*pos +
2000*vel +
15/2*(pos + vel)*(pos + vel) +
2*force +
15*pos*pos
SN noModels:
-- []
-- [force >= -1, force <= 1, Not(181/400 <= pos + 2*vel + 3/2000*force + 9/800*pos*pos), Not(9/20 <= pos), Not(9/20 <= pos + vel), Not(908/3 <= 2000/3*pos + 15/2*pos*pos + 4000/3*vel)]
-- Not(306 <=
    2000/3*pos +
    2000*vel +
    15/2*(pos + vel)*(pos + vel) +
    2*force +
    15*pos*pos)
noModels chcked ante1
adding ante2 to solver
adding force >= -1
added
adding force <= 1
added
adding Not(181/400 <= pos + 2*vel + 3/2000*force + 9/800*pos*pos)
added
adding Not(9/20 <= pos)
added
adding Not(9/20 <= pos + vel)
added
adding Not(908/3 <= 2000/3*pos + 15/2*pos*pos + 4000/3*vel)
added
noModels check of ante1 /\ ante2: sat
added hyp
new ctrl Pred (residue of ctrlPred over Inv) for action :
And(force >= -1,
    force <= 1,
    Not(181/400 <=
        pos + 2*vel + 3/2000*force + 9/800*pos*pos),
    Not(306 <=
        2000/3*pos +
        2000*vel +
        15/2*(pos + vel)*(pos + vel) +
        2*force +
        15*pos*pos))
guardDisjunction: Or(False,
   Exists(force,
          And(force >= -1,
              force <= 1,
              Not(181/400 <=
                  pos + 2*vel + 3/2000*force + 9/800*pos*pos),
              Not(306 <=
                  2000/3*pos +
                  2000*vel +
                  15/2*(pos + vel)*(pos + vel) +
                  2*force +
                  15*pos*pos))))
guardDisjunction after QE [Or(False,
    And(Not(154 <=
            1000/3*pos +
            15/4*(pos + vel)*(pos + vel) +
            15/2*pos*pos +
            1000*vel),
        Not(908/3 <= 2000/3*pos + 15/2*pos*pos + 4000/3*vel)))]
simplified guardDisjunction:
 And(Not(154 <=
        1000/3*pos +
        15/4*(pos + vel)*(pos + vel) +
        15/2*pos*pos +
        1000*vel),
    Not(908/3 <= 2000/3*pos + 15/2*pos*pos + 4000/3*vel))

- Completed arcs from node singleton. Now checking if invariant needs updating (control preds may also get updated again)
stateInv
 And(Not(9/20 <= pos),
    Not(9/20 <= pos + vel),
    Not(908/3 <= 2000/3*pos + 15/2*pos*pos + 4000/3*vel))
SN noModels:
-- []
-- [Not(9/20 <= pos), Not(9/20 <= pos + vel), Not(908/3 <= 2000/3*pos + 15/2*pos*pos + 4000/3*vel)]
-- Not(And(Not(154 <=
            1000/3*pos +
            15/4*(pos + vel)*(pos + vel) +
            15/2*pos*pos +
            1000*vel),
        Not(908/3 <= 2000/3*pos + 15/2*pos*pos + 4000/3*vel)))
noModels chcked ante1
adding ante2 to solver
adding Not(9/20 <= pos)
added
adding Not(9/20 <= pos + vel)
added
adding Not(908/3 <= 2000/3*pos + 15/2*pos*pos + 4000/3*vel)
added
noModels check of ante1 /\ ante2: sat
added hyp
SC context: [Not(9/20 <= pos), Not(9/20 <= pos + vel), Not(908/3 <= 2000/3*pos + 15/2*pos*pos + 4000/3*vel)]
conjuncts: [Not(154 <=
    1000/3*pos +
    15/4*(pos + vel)*(pos + vel) +
    15/2*pos*pos +
    1000*vel), Not(908/3 <= 2000/3*pos + 15/2*pos*pos + 4000/3*vel)]
SN noModels:
-- []
-- [Not(9/20 <= pos), Not(9/20 <= pos + vel), Not(908/3 <= 2000/3*pos + 15/2*pos*pos + 4000/3*vel)]
-- Not(Not(154 <=
        1000/3*pos +
        15/4*(pos + vel)*(pos + vel) +
        15/2*pos*pos +
        1000*vel))
noModels chcked ante1
adding ante2 to solver
adding Not(9/20 <= pos)
added
adding Not(9/20 <= pos + vel)
added
adding Not(908/3 <= 2000/3*pos + 15/2*pos*pos + 4000/3*vel)
added
noModels check of ante1 /\ ante2: sat
added hyp
SN context: [Not(9/20 <= pos), Not(9/20 <= pos + vel), Not(908/3 <= 2000/3*pos + 15/2*pos*pos + 4000/3*vel)]
SN negand: 154 <=
1000/3*pos +
15/4*(pos + vel)*(pos + vel) +
15/2*pos*pos +
1000*vel
SN noModels:
-- []
-- [Not(9/20 <= pos), Not(9/20 <= pos + vel), Not(908/3 <= 2000/3*pos + 15/2*pos*pos + 4000/3*vel)]
-- Not(154 <=
    1000/3*pos +
    15/4*(pos + vel)*(pos + vel) +
    15/2*pos*pos +
    1000*vel)
noModels chcked ante1
adding ante2 to solver
adding Not(9/20 <= pos)
added
adding Not(9/20 <= pos + vel)
added
adding Not(908/3 <= 2000/3*pos + 15/2*pos*pos + 4000/3*vel)
added
noModels check of ante1 /\ ante2: sat
added hyp
SN noModels:
-- []
-- [Not(9/20 <= pos), Not(9/20 <= pos + vel), Not(908/3 <= 2000/3*pos + 15/2*pos*pos + 4000/3*vel), Not(154 <=
    1000/3*pos +
    15/4*(pos + vel)*(pos + vel) +
    15/2*pos*pos +
    1000*vel)]
-- Not(Not(908/3 <= 2000/3*pos + 15/2*pos*pos + 4000/3*vel))
noModels chcked ante1
adding ante2 to solver
adding Not(9/20 <= pos)
added
adding Not(9/20 <= pos + vel)
added
adding Not(908/3 <= 2000/3*pos + 15/2*pos*pos + 4000/3*vel)
added
adding Not(154 <=
    1000/3*pos +
    15/4*(pos + vel)*(pos + vel) +
    15/2*pos*pos +
    1000*vel)
added
noModels check of ante1 /\ ante2: sat
added hyp
newInvDelta:
 [Not(154 <=
     1000/3*pos +
     15/4*(pos + vel)*(pos + vel) +
     15/2*pos*pos +
     1000*vel)]
SN noModels:
-- []
-- [True]
-- Not(And(And(Not(9/20 <= pos),
            Not(9/20 <= pos + vel),
            Not(908/3 <=
                2000/3*pos + 15/2*pos*pos + 4000/3*vel)),
        Not(154 <=
            1000/3*pos +
            15/4*(pos + vel)*(pos + vel) +
            15/2*pos*pos +
            1000*vel)))
noModels chcked ante1
adding ante2 to solver
adding True
added
noModels check of ante1 /\ ante2: sat
added hyp
SC context: [True]
conjuncts: [And(Not(9/20 <= pos),
    Not(9/20 <= pos + vel),
    Not(908/3 <= 2000/3*pos + 15/2*pos*pos + 4000/3*vel)), Not(154 <=
    1000/3*pos +
    15/4*(pos + vel)*(pos + vel) +
    15/2*pos*pos +
    1000*vel)]
SN noModels:
-- []
-- [Not(154 <=
    1000/3*pos +
    15/4*(pos + vel)*(pos + vel) +
    15/2*pos*pos +
    1000*vel)]
-- Not(And(Not(9/20 <= pos),
        Not(9/20 <= pos + vel),
        Not(908/3 <= 2000/3*pos + 15/2*pos*pos + 4000/3*vel)))
noModels chcked ante1
adding ante2 to solver
adding Not(154 <=
    1000/3*pos +
    15/4*(pos + vel)*(pos + vel) +
    15/2*pos*pos +
    1000*vel)
added
noModels check of ante1 /\ ante2: sat
added hyp
SC context: [Not(154 <=
    1000/3*pos +
    15/4*(pos + vel)*(pos + vel) +
    15/2*pos*pos +
    1000*vel)]
conjuncts: [Not(9/20 <= pos), Not(9/20 <= pos + vel), Not(908/3 <= 2000/3*pos + 15/2*pos*pos + 4000/3*vel)]
SN noModels:
-- []
-- [Not(154 <=
    1000/3*pos +
    15/4*(pos + vel)*(pos + vel) +
    15/2*pos*pos +
    1000*vel), Not(9/20 <= pos + vel), Not(908/3 <= 2000/3*pos + 15/2*pos*pos + 4000/3*vel)]
-- Not(Not(9/20 <= pos))
noModels chcked ante1
adding ante2 to solver
adding Not(154 <=
    1000/3*pos +
    15/4*(pos + vel)*(pos + vel) +
    15/2*pos*pos +
    1000*vel)
added
adding Not(9/20 <= pos + vel)
added
adding Not(908/3 <= 2000/3*pos + 15/2*pos*pos + 4000/3*vel)
added
noModels check of ante1 /\ ante2: sat
added hyp
SN context: [Not(154 <=
    1000/3*pos +
    15/4*(pos + vel)*(pos + vel) +
    15/2*pos*pos +
    1000*vel), Not(9/20 <= pos + vel), Not(908/3 <= 2000/3*pos + 15/2*pos*pos + 4000/3*vel)]
SN negand: 9/20 <= pos
SN noModels:
-- []
-- [Not(154 <=
    1000/3*pos +
    15/4*(pos + vel)*(pos + vel) +
    15/2*pos*pos +
    1000*vel), Not(9/20 <= pos + vel), Not(908/3 <= 2000/3*pos + 15/2*pos*pos + 4000/3*vel)]
-- Not(9/20 <= pos)
noModels chcked ante1
adding ante2 to solver
adding Not(154 <=
    1000/3*pos +
    15/4*(pos + vel)*(pos + vel) +
    15/2*pos*pos +
    1000*vel)
added
adding Not(9/20 <= pos + vel)
added
adding Not(908/3 <= 2000/3*pos + 15/2*pos*pos + 4000/3*vel)
added
noModels check of ante1 /\ ante2: sat
added hyp
SN noModels:
-- []
-- [Not(154 <=
    1000/3*pos +
    15/4*(pos + vel)*(pos + vel) +
    15/2*pos*pos +
    1000*vel), Not(9/20 <= pos), Not(908/3 <= 2000/3*pos + 15/2*pos*pos + 4000/3*vel)]
-- Not(Not(9/20 <= pos + vel))
noModels chcked ante1
adding ante2 to solver
adding Not(154 <=
    1000/3*pos +
    15/4*(pos + vel)*(pos + vel) +
    15/2*pos*pos +
    1000*vel)
added
adding Not(9/20 <= pos)
added
adding Not(908/3 <= 2000/3*pos + 15/2*pos*pos + 4000/3*vel)
added
noModels check of ante1 /\ ante2: sat
added hyp
SN context: [Not(154 <=
    1000/3*pos +
    15/4*(pos + vel)*(pos + vel) +
    15/2*pos*pos +
    1000*vel), Not(9/20 <= pos), Not(908/3 <= 2000/3*pos + 15/2*pos*pos + 4000/3*vel)]
SN negand: 9/20 <= pos + vel
SN noModels:
-- []
-- [Not(154 <=
    1000/3*pos +
    15/4*(pos + vel)*(pos + vel) +
    15/2*pos*pos +
    1000*vel), Not(9/20 <= pos), Not(908/3 <= 2000/3*pos + 15/2*pos*pos + 4000/3*vel)]
-- Not(9/20 <= pos + vel)
noModels chcked ante1
adding ante2 to solver
adding Not(154 <=
    1000/3*pos +
    15/4*(pos + vel)*(pos + vel) +
    15/2*pos*pos +
    1000*vel)
added
adding Not(9/20 <= pos)
added
adding Not(908/3 <= 2000/3*pos + 15/2*pos*pos + 4000/3*vel)
added
noModels check of ante1 /\ ante2: sat
added hyp
SN noModels:
-- []
-- [Not(154 <=
    1000/3*pos +
    15/4*(pos + vel)*(pos + vel) +
    15/2*pos*pos +
    1000*vel), Not(9/20 <= pos), Not(9/20 <= pos + vel)]
-- Not(Not(908/3 <= 2000/3*pos + 15/2*pos*pos + 4000/3*vel))
noModels chcked ante1
adding ante2 to solver
adding Not(154 <=
    1000/3*pos +
    15/4*(pos + vel)*(pos + vel) +
    15/2*pos*pos +
    1000*vel)
added
adding Not(9/20 <= pos)
added
adding Not(9/20 <= pos + vel)
added
noModels check of ante1 /\ ante2: sat
added hyp
SN context: [Not(154 <=
    1000/3*pos +
    15/4*(pos + vel)*(pos + vel) +
    15/2*pos*pos +
    1000*vel), Not(9/20 <= pos), Not(9/20 <= pos + vel)]
SN negand: 908/3 <= 2000/3*pos + 15/2*pos*pos + 4000/3*vel
SN noModels:
-- []
-- [Not(154 <=
    1000/3*pos +
    15/4*(pos + vel)*(pos + vel) +
    15/2*pos*pos +
    1000*vel), Not(9/20 <= pos), Not(9/20 <= pos + vel)]
-- Not(908/3 <= 2000/3*pos + 15/2*pos*pos + 4000/3*vel)
noModels chcked ante1
adding ante2 to solver
adding Not(154 <=
    1000/3*pos +
    15/4*(pos + vel)*(pos + vel) +
    15/2*pos*pos +
    1000*vel)
added
adding Not(9/20 <= pos)
added
adding Not(9/20 <= pos + vel)
added
noModels check of ante1 /\ ante2: sat
added hyp
SN noModels:
-- []
-- [Not(9/20 <= pos), Not(9/20 <= pos + vel), Not(908/3 <= 2000/3*pos + 15/2*pos*pos + 4000/3*vel)]
-- Not(Not(154 <=
        1000/3*pos +
        15/4*(pos + vel)*(pos + vel) +
        15/2*pos*pos +
        1000*vel))
noModels chcked ante1
adding ante2 to solver
adding Not(9/20 <= pos)
added
adding Not(9/20 <= pos + vel)
added
adding Not(908/3 <= 2000/3*pos + 15/2*pos*pos + 4000/3*vel)
added
noModels check of ante1 /\ ante2: sat
added hyp
SN context: [Not(9/20 <= pos), Not(9/20 <= pos + vel), Not(908/3 <= 2000/3*pos + 15/2*pos*pos + 4000/3*vel)]
SN negand: 154 <=
1000/3*pos +
15/4*(pos + vel)*(pos + vel) +
15/2*pos*pos +
1000*vel
SN noModels:
-- []
-- [Not(9/20 <= pos), Not(9/20 <= pos + vel), Not(908/3 <= 2000/3*pos + 15/2*pos*pos + 4000/3*vel)]
-- Not(154 <=
    1000/3*pos +
    15/4*(pos + vel)*(pos + vel) +
    15/2*pos*pos +
    1000*vel)
noModels chcked ante1
adding ante2 to solver
adding Not(9/20 <= pos)
added
adding Not(9/20 <= pos + vel)
added
adding Not(908/3 <= 2000/3*pos + 15/2*pos*pos + 4000/3*vel)
added
noModels check of ante1 /\ ante2: sat
added hyp
new invariant
 And(Not(9/20 <= pos),
    Not(9/20 <= pos + vel),
    Not(908/3 <= 2000/3*pos + 15/2*pos*pos + 4000/3*vel),
    Not(154 <=
        1000/3*pos +
        15/4*(pos + vel)*(pos + vel) +
        15/2*pos*pos +
        1000*vel))
SN noModels:
-- []
-- [Not(9/20 <= pos), Not(9/20 <= pos + vel), Not(908/3 <= 2000/3*pos + 15/2*pos*pos + 4000/3*vel), Not(154 <=
    1000/3*pos +
    15/4*(pos + vel)*(pos + vel) +
    15/2*pos*pos +
    1000*vel)]
-- Not(And(force >= -1,
        force <= 1,
        Not(181/400 <=
            pos + 2*vel + 3/2000*force + 9/800*pos*pos),
        Not(306 <=
            2000/3*pos +
            2000*vel +
            15/2*(pos + vel)*(pos + vel) +
            2*force +
            15*pos*pos)))
noModels chcked ante1
adding ante2 to solver
adding Not(9/20 <= pos)
added
adding Not(9/20 <= pos + vel)
added
adding Not(908/3 <= 2000/3*pos + 15/2*pos*pos + 4000/3*vel)
added
adding Not(154 <=
    1000/3*pos +
    15/4*(pos + vel)*(pos + vel) +
    15/2*pos*pos +
    1000*vel)
added
noModels check of ante1 /\ ante2: sat
added hyp
SC context: [Not(9/20 <= pos), Not(9/20 <= pos + vel), Not(908/3 <= 2000/3*pos + 15/2*pos*pos + 4000/3*vel), Not(154 <=
    1000/3*pos +
    15/4*(pos + vel)*(pos + vel) +
    15/2*pos*pos +
    1000*vel)]
conjuncts: [force >= -1, force <= 1, Not(181/400 <= pos + 2*vel + 3/2000*force + 9/800*pos*pos), Not(306 <=
    2000/3*pos +
    2000*vel +
    15/2*(pos + vel)*(pos + vel) +
    2*force +
    15*pos*pos)]
SN noModels:
-- []
-- [Not(9/20 <= pos), Not(9/20 <= pos + vel), Not(908/3 <= 2000/3*pos + 15/2*pos*pos + 4000/3*vel), Not(154 <=
    1000/3*pos +
    15/4*(pos + vel)*(pos + vel) +
    15/2*pos*pos +
    1000*vel), force <= 1, Not(181/400 <= pos + 2*vel + 3/2000*force + 9/800*pos*pos), Not(306 <=
    2000/3*pos +
    2000*vel +
    15/2*(pos + vel)*(pos + vel) +
    2*force +
    15*pos*pos)]
-- Not(force >= -1)
noModels chcked ante1
adding ante2 to solver
adding Not(9/20 <= pos)
added
adding Not(9/20 <= pos + vel)
added
adding Not(908/3 <= 2000/3*pos + 15/2*pos*pos + 4000/3*vel)
added
adding Not(154 <=
    1000/3*pos +
    15/4*(pos + vel)*(pos + vel) +
    15/2*pos*pos +
    1000*vel)
added
adding force <= 1
added
adding Not(181/400 <= pos + 2*vel + 3/2000*force + 9/800*pos*pos)
added
adding Not(306 <=
    2000/3*pos +
    2000*vel +
    15/2*(pos + vel)*(pos + vel) +
    2*force +
    15*pos*pos)
added
noModels check of ante1 /\ ante2: sat
added hyp
SN noModels:
-- []
-- [Not(9/20 <= pos), Not(9/20 <= pos + vel), Not(908/3 <= 2000/3*pos + 15/2*pos*pos + 4000/3*vel), Not(154 <=
    1000/3*pos +
    15/4*(pos + vel)*(pos + vel) +
    15/2*pos*pos +
    1000*vel), force >= -1, Not(181/400 <= pos + 2*vel + 3/2000*force + 9/800*pos*pos), Not(306 <=
    2000/3*pos +
    2000*vel +
    15/2*(pos + vel)*(pos + vel) +
    2*force +
    15*pos*pos)]
-- Not(force <= 1)
noModels chcked ante1
adding ante2 to solver
adding Not(9/20 <= pos)
added
adding Not(9/20 <= pos + vel)
added
adding Not(908/3 <= 2000/3*pos + 15/2*pos*pos + 4000/3*vel)
added
adding Not(154 <=
    1000/3*pos +
    15/4*(pos + vel)*(pos + vel) +
    15/2*pos*pos +
    1000*vel)
added
adding force >= -1
added
adding Not(181/400 <= pos + 2*vel + 3/2000*force + 9/800*pos*pos)
added
adding Not(306 <=
    2000/3*pos +
    2000*vel +
    15/2*(pos + vel)*(pos + vel) +
    2*force +
    15*pos*pos)
added
noModels check of ante1 /\ ante2: sat
added hyp
SN noModels:
-- []
-- [Not(9/20 <= pos), Not(9/20 <= pos + vel), Not(908/3 <= 2000/3*pos + 15/2*pos*pos + 4000/3*vel), Not(154 <=
    1000/3*pos +
    15/4*(pos + vel)*(pos + vel) +
    15/2*pos*pos +
    1000*vel), force >= -1, force <= 1, Not(306 <=
    2000/3*pos +
    2000*vel +
    15/2*(pos + vel)*(pos + vel) +
    2*force +
    15*pos*pos)]
-- Not(Not(181/400 <=
        pos + 2*vel + 3/2000*force + 9/800*pos*pos))
noModels chcked ante1
adding ante2 to solver
adding Not(9/20 <= pos)
added
adding Not(9/20 <= pos + vel)
added
adding Not(908/3 <= 2000/3*pos + 15/2*pos*pos + 4000/3*vel)
added
adding Not(154 <=
    1000/3*pos +
    15/4*(pos + vel)*(pos + vel) +
    15/2*pos*pos +
    1000*vel)
added
adding force >= -1
added
adding force <= 1
added
adding Not(306 <=
    2000/3*pos +
    2000*vel +
    15/2*(pos + vel)*(pos + vel) +
    2*force +
    15*pos*pos)
added
noModels check of ante1 /\ ante2: sat
added hyp
SN context: [Not(9/20 <= pos), Not(9/20 <= pos + vel), Not(908/3 <= 2000/3*pos + 15/2*pos*pos + 4000/3*vel), Not(154 <=
    1000/3*pos +
    15/4*(pos + vel)*(pos + vel) +
    15/2*pos*pos +
    1000*vel), force >= -1, force <= 1, Not(306 <=
    2000/3*pos +
    2000*vel +
    15/2*(pos + vel)*(pos + vel) +
    2*force +
    15*pos*pos)]
SN negand: 181/400 <= pos + 2*vel + 3/2000*force + 9/800*pos*pos
SN noModels:
-- []
-- [Not(9/20 <= pos), Not(9/20 <= pos + vel), Not(908/3 <= 2000/3*pos + 15/2*pos*pos + 4000/3*vel), Not(154 <=
    1000/3*pos +
    15/4*(pos + vel)*(pos + vel) +
    15/2*pos*pos +
    1000*vel), force >= -1, force <= 1, Not(306 <=
    2000/3*pos +
    2000*vel +
    15/2*(pos + vel)*(pos + vel) +
    2*force +
    15*pos*pos)]
-- Not(181/400 <= pos + 2*vel + 3/2000*force + 9/800*pos*pos)
noModels chcked ante1
adding ante2 to solver
adding Not(9/20 <= pos)
added
adding Not(9/20 <= pos + vel)
added
adding Not(908/3 <= 2000/3*pos + 15/2*pos*pos + 4000/3*vel)
added
adding Not(154 <=
    1000/3*pos +
    15/4*(pos + vel)*(pos + vel) +
    15/2*pos*pos +
    1000*vel)
added
adding force >= -1
added
adding force <= 1
added
adding Not(306 <=
    2000/3*pos +
    2000*vel +
    15/2*(pos + vel)*(pos + vel) +
    2*force +
    15*pos*pos)
added
noModels check of ante1 /\ ante2: sat
added hyp
SN noModels:
-- []
-- [Not(9/20 <= pos), Not(9/20 <= pos + vel), Not(908/3 <= 2000/3*pos + 15/2*pos*pos + 4000/3*vel), Not(154 <=
    1000/3*pos +
    15/4*(pos + vel)*(pos + vel) +
    15/2*pos*pos +
    1000*vel), force >= -1, force <= 1, Not(181/400 <= pos + 2*vel + 3/2000*force + 9/800*pos*pos)]
-- Not(Not(306 <=
        2000/3*pos +
        2000*vel +
        15/2*(pos + vel)*(pos + vel) +
        2*force +
        15*pos*pos))
noModels chcked ante1
adding ante2 to solver
adding Not(9/20 <= pos)
added
adding Not(9/20 <= pos + vel)
added
adding Not(908/3 <= 2000/3*pos + 15/2*pos*pos + 4000/3*vel)
added
adding Not(154 <=
    1000/3*pos +
    15/4*(pos + vel)*(pos + vel) +
    15/2*pos*pos +
    1000*vel)
added
adding force >= -1
added
adding force <= 1
added
adding Not(181/400 <= pos + 2*vel + 3/2000*force + 9/800*pos*pos)
added
noModels check of ante1 /\ ante2: sat
added hyp
SN context: [Not(9/20 <= pos), Not(9/20 <= pos + vel), Not(908/3 <= 2000/3*pos + 15/2*pos*pos + 4000/3*vel), Not(154 <=
    1000/3*pos +
    15/4*(pos + vel)*(pos + vel) +
    15/2*pos*pos +
    1000*vel), force >= -1, force <= 1, Not(181/400 <= pos + 2*vel + 3/2000*force + 9/800*pos*pos)]
SN negand: 306 <=
2000/3*pos +
2000*vel +
15/2*(pos + vel)*(pos + vel) +
2*force +
15*pos*pos
SN noModels:
-- []
-- [Not(9/20 <= pos), Not(9/20 <= pos + vel), Not(908/3 <= 2000/3*pos + 15/2*pos*pos + 4000/3*vel), Not(154 <=
    1000/3*pos +
    15/4*(pos + vel)*(pos + vel) +
    15/2*pos*pos +
    1000*vel), force >= -1, force <= 1, Not(181/400 <= pos + 2*vel + 3/2000*force + 9/800*pos*pos)]
-- Not(306 <=
    2000/3*pos +
    2000*vel +
    15/2*(pos + vel)*(pos + vel) +
    2*force +
    15*pos*pos)
noModels chcked ante1
adding ante2 to solver
adding Not(9/20 <= pos)
added
adding Not(9/20 <= pos + vel)
added
adding Not(908/3 <= 2000/3*pos + 15/2*pos*pos + 4000/3*vel)
added
adding Not(154 <=
    1000/3*pos +
    15/4*(pos + vel)*(pos + vel) +
    15/2*pos*pos +
    1000*vel)
added
adding force >= -1
added
adding force <= 1
added
adding Not(181/400 <= pos + 2*vel + 3/2000*force + 9/800*pos*pos)
added
noModels check of ante1 /\ ante2: sat
added hyp
new ctrl Pred (diff b/w updated state inv and current ctrl pred):
 And(force >= -1,
    force <= 1,
    Not(181/400 <=
        pos + 2*vel + 3/2000*force + 9/800*pos*pos),
    Not(306 <=
        2000/3*pos +
        2000*vel +
        15/2*(pos + vel)*(pos + vel) +
        2*force +
        15*pos*pos))

----------------
iteration 3

-- Refining the state invariant for node singleton
--- arc: action
wcp =
 ForAll([posX, velX],
       Implies(And(velX ==
                   vel + force*3/2000 -
                   1/400*(1 - (3*pos*3*pos)/2),
                   posX == pos + vel),
               And(Not(9/20 <= posX),
                   Not(9/20 <= posX + velX),
                   Not(908/3 <=
                       2000/3*posX +
                       15/2*posX*posX +
                       4000/3*velX),
                   Not(154 <=
                       1000/3*posX +
                       15/4*(posX + velX)*(posX + velX) +
                       15/2*posX*posX +
                       1000*velX))))
wcp_simp =
 And(Not(9/20 <= pos + vel),
    Not(181/400 <=
        pos + 2*vel + 3/2000*force + 9/800*pos*pos),
    Not(306 <=
        2000/3*pos +
        2000*vel +
        15/2*(pos + vel)*(pos + vel) +
        2*force +
        15*pos*pos),
    Not(313/2 <=
        1000/3*pos +
        4000/3*vel +
        15/4*
        (-1/400 + pos + 2*vel + 3/2000*force + 9/800*pos*pos)*
        (-1/400 + pos + 2*vel + 3/2000*force + 9/800*pos*pos) +
        15/2*(pos + vel)*(pos + vel) +
        3/2*force +
        45/4*pos*pos))
wcp_simp after QE (wcp_qe) =
And(Not(9/20 <= pos + vel),
    Not(181/400 <=
        pos + 2*vel + 3/2000*force + 9/800*pos*pos),
    Not(306 <=
        2000/3*pos +
        2000*vel +
        15/2*(pos + vel)*(pos + vel) +
        2*force +
        15*pos*pos),
    Not(313/2 <=
        1000/3*pos +
        4000/3*vel +
        15/4*
        (-1/400 + pos + 2*vel + 3/2000*force + 9/800*pos*pos)*
        (-1/400 + pos + 2*vel + 3/2000*force + 9/800*pos*pos) +
        15/2*(pos + vel)*(pos + vel) +
        3/2*force +
        45/4*pos*pos))
strengthened ctrl pred
And(force >= -1,
    force <= 1,
    Not(181/400 <=
        pos + 2*vel + 3/2000*force + 9/800*pos*pos),
    Not(306 <=
        2000/3*pos +
        2000*vel +
        15/2*(pos + vel)*(pos + vel) +
        2*force +
        15*pos*pos),
    Not(9/20 <= pos + vel),
    Not(313/2 <=
        1000/3*pos +
        4000/3*vel +
        15/4*
        (-1/400 + pos + 2*vel + 3/2000*force + 9/800*pos*pos)*
        (-1/400 + pos + 2*vel + 3/2000*force + 9/800*pos*pos) +
        15/2*(pos + vel)*(pos + vel) +
        3/2*force +
        45/4*pos*pos))
SN noModels:
-- []
-- [Not(9/20 <= pos), Not(9/20 <= pos + vel), Not(908/3 <= 2000/3*pos + 15/2*pos*pos + 4000/3*vel), Not(154 <=
    1000/3*pos +
    15/4*(pos + vel)*(pos + vel) +
    15/2*pos*pos +
    1000*vel)]
-- Not(And(force >= -1,
        force <= 1,
        Not(181/400 <=
            pos + 2*vel + 3/2000*force + 9/800*pos*pos),
        Not(306 <=
            2000/3*pos +
            2000*vel +
            15/2*(pos + vel)*(pos + vel) +
            2*force +
            15*pos*pos),
        Not(9/20 <= pos + vel),
        Not(313/2 <=
            1000/3*pos +
            4000/3*vel +
            15/4*
            (-1/400 +
             pos +
             2*vel +
             3/2000*force +
             9/800*pos*pos)*
            (-1/400 +
             pos +
             2*vel +
             3/2000*force +
             9/800*pos*pos) +
            15/2*(pos + vel)*(pos + vel) +
            3/2*force +
            45/4*pos*pos)))
noModels chcked ante1
adding ante2 to solver
adding Not(9/20 <= pos)
added
adding Not(9/20 <= pos + vel)
added
adding Not(908/3 <= 2000/3*pos + 15/2*pos*pos + 4000/3*vel)
added
adding Not(154 <=
    1000/3*pos +
    15/4*(pos + vel)*(pos + vel) +
    15/2*pos*pos +
    1000*vel)
added
noModels check of ante1 /\ ante2: sat
added hyp
SC context: [Not(9/20 <= pos), Not(9/20 <= pos + vel), Not(908/3 <= 2000/3*pos + 15/2*pos*pos + 4000/3*vel), Not(154 <=
    1000/3*pos +
    15/4*(pos + vel)*(pos + vel) +
    15/2*pos*pos +
    1000*vel)]
conjuncts: [force >= -1, force <= 1, Not(181/400 <= pos + 2*vel + 3/2000*force + 9/800*pos*pos), Not(306 <=
    2000/3*pos +
    2000*vel +
    15/2*(pos + vel)*(pos + vel) +
    2*force +
    15*pos*pos), Not(9/20 <= pos + vel), Not(313/2 <=
    1000/3*pos +
    4000/3*vel +
    15/4*
    (-1/400 + pos + 2*vel + 3/2000*force + 9/800*pos*pos)*
    (-1/400 + pos + 2*vel + 3/2000*force + 9/800*pos*pos) +
    15/2*(pos + vel)*(pos + vel) +
    3/2*force +
    45/4*pos*pos)]
SN noModels:
-- []
-- [Not(9/20 <= pos), Not(9/20 <= pos + vel), Not(908/3 <= 2000/3*pos + 15/2*pos*pos + 4000/3*vel), Not(154 <=
    1000/3*pos +
    15/4*(pos + vel)*(pos + vel) +
    15/2*pos*pos +
    1000*vel), force <= 1, Not(181/400 <= pos + 2*vel + 3/2000*force + 9/800*pos*pos), Not(306 <=
    2000/3*pos +
    2000*vel +
    15/2*(pos + vel)*(pos + vel) +
    2*force +
    15*pos*pos), Not(313/2 <=
    1000/3*pos +
    4000/3*vel +
    15/4*
    (-1/400 + pos + 2*vel + 3/2000*force + 9/800*pos*pos)*
    (-1/400 + pos + 2*vel + 3/2000*force + 9/800*pos*pos) +
    15/2*(pos + vel)*(pos + vel) +
    3/2*force +
    45/4*pos*pos)]
-- Not(force >= -1)
noModels chcked ante1
adding ante2 to solver
adding Not(9/20 <= pos)
added
adding Not(9/20 <= pos + vel)
added
adding Not(908/3 <= 2000/3*pos + 15/2*pos*pos + 4000/3*vel)
added
adding Not(154 <=
    1000/3*pos +
    15/4*(pos + vel)*(pos + vel) +
    15/2*pos*pos +
    1000*vel)
added
adding force <= 1
added
adding Not(181/400 <= pos + 2*vel + 3/2000*force + 9/800*pos*pos)
added
adding Not(306 <=
    2000/3*pos +
    2000*vel +
    15/2*(pos + vel)*(pos + vel) +
    2*force +
    15*pos*pos)
added
adding Not(313/2 <=
    1000/3*pos +
    4000/3*vel +
    15/4*
    (-1/400 + pos + 2*vel + 3/2000*force + 9/800*pos*pos)*
    (-1/400 + pos + 2*vel + 3/2000*force + 9/800*pos*pos) +
    15/2*(pos + vel)*(pos + vel) +
    3/2*force +
    45/4*pos*pos)
added
noModels check of ante1 /\ ante2: sat
added hyp
SN noModels:
-- []
-- [Not(9/20 <= pos), Not(9/20 <= pos + vel), Not(908/3 <= 2000/3*pos + 15/2*pos*pos + 4000/3*vel), Not(154 <=
    1000/3*pos +
    15/4*(pos + vel)*(pos + vel) +
    15/2*pos*pos +
    1000*vel), force >= -1, Not(181/400 <= pos + 2*vel + 3/2000*force + 9/800*pos*pos), Not(306 <=
    2000/3*pos +
    2000*vel +
    15/2*(pos + vel)*(pos + vel) +
    2*force +
    15*pos*pos), Not(313/2 <=
    1000/3*pos +
    4000/3*vel +
    15/4*
    (-1/400 + pos + 2*vel + 3/2000*force + 9/800*pos*pos)*
    (-1/400 + pos + 2*vel + 3/2000*force + 9/800*pos*pos) +
    15/2*(pos + vel)*(pos + vel) +
    3/2*force +
    45/4*pos*pos)]
-- Not(force <= 1)
noModels chcked ante1
adding ante2 to solver
adding Not(9/20 <= pos)
added
adding Not(9/20 <= pos + vel)
added
adding Not(908/3 <= 2000/3*pos + 15/2*pos*pos + 4000/3*vel)
added
adding Not(154 <=
    1000/3*pos +
    15/4*(pos + vel)*(pos + vel) +
    15/2*pos*pos +
    1000*vel)
added
adding force >= -1
added
adding Not(181/400 <= pos + 2*vel + 3/2000*force + 9/800*pos*pos)
added
adding Not(306 <=
    2000/3*pos +
    2000*vel +
    15/2*(pos + vel)*(pos + vel) +
    2*force +
    15*pos*pos)
added
adding Not(313/2 <=
    1000/3*pos +
    4000/3*vel +
    15/4*
    (-1/400 + pos + 2*vel + 3/2000*force + 9/800*pos*pos)*
    (-1/400 + pos + 2*vel + 3/2000*force + 9/800*pos*pos) +
    15/2*(pos + vel)*(pos + vel) +
    3/2*force +
    45/4*pos*pos)
added
noModels check of ante1 /\ ante2: sat
added hyp
SN noModels:
-- []
-- [Not(9/20 <= pos), Not(9/20 <= pos + vel), Not(908/3 <= 2000/3*pos + 15/2*pos*pos + 4000/3*vel), Not(154 <=
    1000/3*pos +
    15/4*(pos + vel)*(pos + vel) +
    15/2*pos*pos +
    1000*vel), force >= -1, force <= 1, Not(306 <=
    2000/3*pos +
    2000*vel +
    15/2*(pos + vel)*(pos + vel) +
    2*force +
    15*pos*pos), Not(313/2 <=
    1000/3*pos +
    4000/3*vel +
    15/4*
    (-1/400 + pos + 2*vel + 3/2000*force + 9/800*pos*pos)*
    (-1/400 + pos + 2*vel + 3/2000*force + 9/800*pos*pos) +
    15/2*(pos + vel)*(pos + vel) +
    3/2*force +
    45/4*pos*pos)]
-- Not(Not(181/400 <=
        pos + 2*vel + 3/2000*force + 9/800*pos*pos))
noModels chcked ante1
adding ante2 to solver
adding Not(9/20 <= pos)
added
adding Not(9/20 <= pos + vel)
added
adding Not(908/3 <= 2000/3*pos + 15/2*pos*pos + 4000/3*vel)
added
adding Not(154 <=
    1000/3*pos +
    15/4*(pos + vel)*(pos + vel) +
    15/2*pos*pos +
    1000*vel)
added
adding force >= -1
added
adding force <= 1
added
adding Not(306 <=
    2000/3*pos +
    2000*vel +
    15/2*(pos + vel)*(pos + vel) +
    2*force +
    15*pos*pos)
added
adding Not(313/2 <=
    1000/3*pos +
    4000/3*vel +
    15/4*
    (-1/400 + pos + 2*vel + 3/2000*force + 9/800*pos*pos)*
    (-1/400 + pos + 2*vel + 3/2000*force + 9/800*pos*pos) +
    15/2*(pos + vel)*(pos + vel) +
    3/2*force +
    45/4*pos*pos)
added
noModels check of ante1 /\ ante2: sat
added hyp
SN context: [Not(9/20 <= pos), Not(9/20 <= pos + vel), Not(908/3 <= 2000/3*pos + 15/2*pos*pos + 4000/3*vel), Not(154 <=
    1000/3*pos +
    15/4*(pos + vel)*(pos + vel) +
    15/2*pos*pos +
    1000*vel), force >= -1, force <= 1, Not(306 <=
    2000/3*pos +
    2000*vel +
    15/2*(pos + vel)*(pos + vel) +
    2*force +
    15*pos*pos), Not(313/2 <=
    1000/3*pos +
    4000/3*vel +
    15/4*
    (-1/400 + pos + 2*vel + 3/2000*force + 9/800*pos*pos)*
    (-1/400 + pos + 2*vel + 3/2000*force + 9/800*pos*pos) +
    15/2*(pos + vel)*(pos + vel) +
    3/2*force +
    45/4*pos*pos)]
SN negand: 181/400 <= pos + 2*vel + 3/2000*force + 9/800*pos*pos
SN noModels:
-- []
-- [Not(9/20 <= pos), Not(9/20 <= pos + vel), Not(908/3 <= 2000/3*pos + 15/2*pos*pos + 4000/3*vel), Not(154 <=
    1000/3*pos +
    15/4*(pos + vel)*(pos + vel) +
    15/2*pos*pos +
    1000*vel), force >= -1, force <= 1, Not(306 <=
    2000/3*pos +
    2000*vel +
    15/2*(pos + vel)*(pos + vel) +
    2*force +
    15*pos*pos), Not(313/2 <=
    1000/3*pos +
    4000/3*vel +
    15/4*
    (-1/400 + pos + 2*vel + 3/2000*force + 9/800*pos*pos)*
    (-1/400 + pos + 2*vel + 3/2000*force + 9/800*pos*pos) +
    15/2*(pos + vel)*(pos + vel) +
    3/2*force +
    45/4*pos*pos)]
-- Not(181/400 <= pos + 2*vel + 3/2000*force + 9/800*pos*pos)
noModels chcked ante1
adding ante2 to solver
adding Not(9/20 <= pos)
added
adding Not(9/20 <= pos + vel)
added
adding Not(908/3 <= 2000/3*pos + 15/2*pos*pos + 4000/3*vel)
added
adding Not(154 <=
    1000/3*pos +
    15/4*(pos + vel)*(pos + vel) +
    15/2*pos*pos +
    1000*vel)
added
adding force >= -1
added
adding force <= 1
added
adding Not(306 <=
    2000/3*pos +
    2000*vel +
    15/2*(pos + vel)*(pos + vel) +
    2*force +
    15*pos*pos)
added
adding Not(313/2 <=
    1000/3*pos +
    4000/3*vel +
    15/4*
    (-1/400 + pos + 2*vel + 3/2000*force + 9/800*pos*pos)*
    (-1/400 + pos + 2*vel + 3/2000*force + 9/800*pos*pos) +
    15/2*(pos + vel)*(pos + vel) +
    3/2*force +
    45/4*pos*pos)
added
noModels check of ante1 /\ ante2: sat
added hyp
SN noModels:
-- []
-- [Not(9/20 <= pos), Not(9/20 <= pos + vel), Not(908/3 <= 2000/3*pos + 15/2*pos*pos + 4000/3*vel), Not(154 <=
    1000/3*pos +
    15/4*(pos + vel)*(pos + vel) +
    15/2*pos*pos +
    1000*vel), force >= -1, force <= 1, Not(181/400 <= pos + 2*vel + 3/2000*force + 9/800*pos*pos), Not(313/2 <=
    1000/3*pos +
    4000/3*vel +
    15/4*
    (-1/400 + pos + 2*vel + 3/2000*force + 9/800*pos*pos)*
    (-1/400 + pos + 2*vel + 3/2000*force + 9/800*pos*pos) +
    15/2*(pos + vel)*(pos + vel) +
    3/2*force +
    45/4*pos*pos)]
-- Not(Not(306 <=
        2000/3*pos +
        2000*vel +
        15/2*(pos + vel)*(pos + vel) +
        2*force +
        15*pos*pos))
noModels chcked ante1
adding ante2 to solver
adding Not(9/20 <= pos)
added
adding Not(9/20 <= pos + vel)
added
adding Not(908/3 <= 2000/3*pos + 15/2*pos*pos + 4000/3*vel)
added
adding Not(154 <=
    1000/3*pos +
    15/4*(pos + vel)*(pos + vel) +
    15/2*pos*pos +
    1000*vel)
added
adding force >= -1
added
adding force <= 1
added
adding Not(181/400 <= pos + 2*vel + 3/2000*force + 9/800*pos*pos)
added
adding Not(313/2 <=
    1000/3*pos +
    4000/3*vel +
    15/4*
    (-1/400 + pos + 2*vel + 3/2000*force + 9/800*pos*pos)*
    (-1/400 + pos + 2*vel + 3/2000*force + 9/800*pos*pos) +
    15/2*(pos + vel)*(pos + vel) +
    3/2*force +
    45/4*pos*pos)
added
noModels check of ante1 /\ ante2: sat
added hyp
SN context: [Not(9/20 <= pos), Not(9/20 <= pos + vel), Not(908/3 <= 2000/3*pos + 15/2*pos*pos + 4000/3*vel), Not(154 <=
    1000/3*pos +
    15/4*(pos + vel)*(pos + vel) +
    15/2*pos*pos +
    1000*vel), force >= -1, force <= 1, Not(181/400 <= pos + 2*vel + 3/2000*force + 9/800*pos*pos), Not(313/2 <=
    1000/3*pos +
    4000/3*vel +
    15/4*
    (-1/400 + pos + 2*vel + 3/2000*force + 9/800*pos*pos)*
    (-1/400 + pos + 2*vel + 3/2000*force + 9/800*pos*pos) +
    15/2*(pos + vel)*(pos + vel) +
    3/2*force +
    45/4*pos*pos)]
SN negand: 306 <=
2000/3*pos +
2000*vel +
15/2*(pos + vel)*(pos + vel) +
2*force +
15*pos*pos
SN noModels:
-- []
-- [Not(9/20 <= pos), Not(9/20 <= pos + vel), Not(908/3 <= 2000/3*pos + 15/2*pos*pos + 4000/3*vel), Not(154 <=
    1000/3*pos +
    15/4*(pos + vel)*(pos + vel) +
    15/2*pos*pos +
    1000*vel), force >= -1, force <= 1, Not(181/400 <= pos + 2*vel + 3/2000*force + 9/800*pos*pos), Not(313/2 <=
    1000/3*pos +
    4000/3*vel +
    15/4*
    (-1/400 + pos + 2*vel + 3/2000*force + 9/800*pos*pos)*
    (-1/400 + pos + 2*vel + 3/2000*force + 9/800*pos*pos) +
    15/2*(pos + vel)*(pos + vel) +
    3/2*force +
    45/4*pos*pos)]
-- Not(306 <=
    2000/3*pos +
    2000*vel +
    15/2*(pos + vel)*(pos + vel) +
    2*force +
    15*pos*pos)
noModels chcked ante1
adding ante2 to solver
adding Not(9/20 <= pos)
added
adding Not(9/20 <= pos + vel)
added
adding Not(908/3 <= 2000/3*pos + 15/2*pos*pos + 4000/3*vel)
added
adding Not(154 <=
    1000/3*pos +
    15/4*(pos + vel)*(pos + vel) +
    15/2*pos*pos +
    1000*vel)
added
adding force >= -1
added
adding force <= 1
added
adding Not(181/400 <= pos + 2*vel + 3/2000*force + 9/800*pos*pos)
added
adding Not(313/2 <=
    1000/3*pos +
    4000/3*vel +
    15/4*
    (-1/400 + pos + 2*vel + 3/2000*force + 9/800*pos*pos)*
    (-1/400 + pos + 2*vel + 3/2000*force + 9/800*pos*pos) +
    15/2*(pos + vel)*(pos + vel) +
    3/2*force +
    45/4*pos*pos)
added
noModels check of ante1 /\ ante2: sat
added hyp
SN noModels:
-- []
-- [Not(9/20 <= pos), Not(9/20 <= pos + vel), Not(908/3 <= 2000/3*pos + 15/2*pos*pos + 4000/3*vel), Not(154 <=
    1000/3*pos +
    15/4*(pos + vel)*(pos + vel) +
    15/2*pos*pos +
    1000*vel), force >= -1, force <= 1, Not(181/400 <= pos + 2*vel + 3/2000*force + 9/800*pos*pos), Not(306 <=
    2000/3*pos +
    2000*vel +
    15/2*(pos + vel)*(pos + vel) +
    2*force +
    15*pos*pos), Not(313/2 <=
    1000/3*pos +
    4000/3*vel +
    15/4*
    (-1/400 + pos + 2*vel + 3/2000*force + 9/800*pos*pos)*
    (-1/400 + pos + 2*vel + 3/2000*force + 9/800*pos*pos) +
    15/2*(pos + vel)*(pos + vel) +
    3/2*force +
    45/4*pos*pos)]
-- Not(Not(9/20 <= pos + vel))
noModels chcked ante1
adding ante2 to solver
adding Not(9/20 <= pos)
added
adding Not(9/20 <= pos + vel)
added
adding Not(908/3 <= 2000/3*pos + 15/2*pos*pos + 4000/3*vel)
added
adding Not(154 <=
    1000/3*pos +
    15/4*(pos + vel)*(pos + vel) +
    15/2*pos*pos +
    1000*vel)
added
adding force >= -1
added
adding force <= 1
added
adding Not(181/400 <= pos + 2*vel + 3/2000*force + 9/800*pos*pos)
added
adding Not(306 <=
    2000/3*pos +
    2000*vel +
    15/2*(pos + vel)*(pos + vel) +
    2*force +
    15*pos*pos)
added
adding Not(313/2 <=
    1000/3*pos +
    4000/3*vel +
    15/4*
    (-1/400 + pos + 2*vel + 3/2000*force + 9/800*pos*pos)*
    (-1/400 + pos + 2*vel + 3/2000*force + 9/800*pos*pos) +
    15/2*(pos + vel)*(pos + vel) +
    3/2*force +
    45/4*pos*pos)
added
noModels check of ante1 /\ ante2: sat
added hyp
SN noModels:
-- []
-- [Not(9/20 <= pos), Not(9/20 <= pos + vel), Not(908/3 <= 2000/3*pos + 15/2*pos*pos + 4000/3*vel), Not(154 <=
    1000/3*pos +
    15/4*(pos + vel)*(pos + vel) +
    15/2*pos*pos +
    1000*vel), force >= -1, force <= 1, Not(181/400 <= pos + 2*vel + 3/2000*force + 9/800*pos*pos), Not(306 <=
    2000/3*pos +
    2000*vel +
    15/2*(pos + vel)*(pos + vel) +
    2*force +
    15*pos*pos)]
-- Not(Not(313/2 <=
        1000/3*pos +
        4000/3*vel +
        15/4*
        (-1/400 + pos + 2*vel + 3/2000*force + 9/800*pos*pos)*
        (-1/400 + pos + 2*vel + 3/2000*force + 9/800*pos*pos) +
        15/2*(pos + vel)*(pos + vel) +
        3/2*force +
        45/4*pos*pos))
noModels chcked ante1
adding ante2 to solver
adding Not(9/20 <= pos)
added
adding Not(9/20 <= pos + vel)
added
adding Not(908/3 <= 2000/3*pos + 15/2*pos*pos + 4000/3*vel)
added
adding Not(154 <=
    1000/3*pos +
    15/4*(pos + vel)*(pos + vel) +
    15/2*pos*pos +
    1000*vel)
added
adding force >= -1
added
adding force <= 1
added
adding Not(181/400 <= pos + 2*vel + 3/2000*force + 9/800*pos*pos)
added
adding Not(306 <=
    2000/3*pos +
    2000*vel +
    15/2*(pos + vel)*(pos + vel) +
    2*force +
    15*pos*pos)
added
noModels check of ante1 /\ ante2: sat
added hyp
SN context: [Not(9/20 <= pos), Not(9/20 <= pos + vel), Not(908/3 <= 2000/3*pos + 15/2*pos*pos + 4000/3*vel), Not(154 <=
    1000/3*pos +
    15/4*(pos + vel)*(pos + vel) +
    15/2*pos*pos +
    1000*vel), force >= -1, force <= 1, Not(181/400 <= pos + 2*vel + 3/2000*force + 9/800*pos*pos), Not(306 <=
    2000/3*pos +
    2000*vel +
    15/2*(pos + vel)*(pos + vel) +
    2*force +
    15*pos*pos)]
SN negand: 313/2 <=
1000/3*pos +
4000/3*vel +
15/4*
(-1/400 + pos + 2*vel + 3/2000*force + 9/800*pos*pos)*
(-1/400 + pos + 2*vel + 3/2000*force + 9/800*pos*pos) +
15/2*(pos + vel)*(pos + vel) +
3/2*force +
45/4*pos*pos
SN noModels:
-- []
-- [Not(9/20 <= pos), Not(9/20 <= pos + vel), Not(908/3 <= 2000/3*pos + 15/2*pos*pos + 4000/3*vel), Not(154 <=
    1000/3*pos +
    15/4*(pos + vel)*(pos + vel) +
    15/2*pos*pos +
    1000*vel), force >= -1, force <= 1, Not(181/400 <= pos + 2*vel + 3/2000*force + 9/800*pos*pos), Not(306 <=
    2000/3*pos +
    2000*vel +
    15/2*(pos + vel)*(pos + vel) +
    2*force +
    15*pos*pos)]
-- Not(313/2 <=
    1000/3*pos +
    4000/3*vel +
    15/4*
    (-1/400 + pos + 2*vel + 3/2000*force + 9/800*pos*pos)*
    (-1/400 + pos + 2*vel + 3/2000*force + 9/800*pos*pos) +
    15/2*(pos + vel)*(pos + vel) +
    3/2*force +
    45/4*pos*pos)
noModels chcked ante1
adding ante2 to solver
adding Not(9/20 <= pos)
added
adding Not(9/20 <= pos + vel)
added
adding Not(908/3 <= 2000/3*pos + 15/2*pos*pos + 4000/3*vel)
added
adding Not(154 <=
    1000/3*pos +
    15/4*(pos + vel)*(pos + vel) +
    15/2*pos*pos +
    1000*vel)
added
adding force >= -1
added
adding force <= 1
added
adding Not(181/400 <= pos + 2*vel + 3/2000*force + 9/800*pos*pos)
added
adding Not(306 <=
    2000/3*pos +
    2000*vel +
    15/2*(pos + vel)*(pos + vel) +
    2*force +
    15*pos*pos)
added
noModels check of ante1 /\ ante2: sat
added hyp
SN noModels:
-- []
-- [force >= -1, force <= 1, Not(181/400 <= pos + 2*vel + 3/2000*force + 9/800*pos*pos), Not(306 <=
    2000/3*pos +
    2000*vel +
    15/2*(pos + vel)*(pos + vel) +
    2*force +
    15*pos*pos), Not(9/20 <= pos), Not(9/20 <= pos + vel), Not(908/3 <= 2000/3*pos + 15/2*pos*pos + 4000/3*vel), Not(154 <=
    1000/3*pos +
    15/4*(pos + vel)*(pos + vel) +
    15/2*pos*pos +
    1000*vel)]
-- Not(And(force >= -1,
        force <= 1,
        Not(181/400 <=
            pos + 2*vel + 3/2000*force + 9/800*pos*pos),
        Not(306 <=
            2000/3*pos +
            2000*vel +
            15/2*(pos + vel)*(pos + vel) +
            2*force +
            15*pos*pos),
        Not(313/2 <=
            1000/3*pos +
            4000/3*vel +
            15/4*
            (-1/400 +
             pos +
             2*vel +
             3/2000*force +
             9/800*pos*pos)*
            (-1/400 +
             pos +
             2*vel +
             3/2000*force +
             9/800*pos*pos) +
            15/2*(pos + vel)*(pos + vel) +
            3/2*force +
            45/4*pos*pos)))
noModels chcked ante1
adding ante2 to solver
adding force >= -1
added
adding force <= 1
added
adding Not(181/400 <= pos + 2*vel + 3/2000*force + 9/800*pos*pos)
added
adding Not(306 <=
    2000/3*pos +
    2000*vel +
    15/2*(pos + vel)*(pos + vel) +
    2*force +
    15*pos*pos)
added
adding Not(9/20 <= pos)
added
adding Not(9/20 <= pos + vel)
added
adding Not(908/3 <= 2000/3*pos + 15/2*pos*pos + 4000/3*vel)
added
adding Not(154 <=
    1000/3*pos +
    15/4*(pos + vel)*(pos + vel) +
    15/2*pos*pos +
    1000*vel)
added
noModels check of ante1 /\ ante2: sat
added hyp
SC context: [force >= -1, force <= 1, Not(181/400 <= pos + 2*vel + 3/2000*force + 9/800*pos*pos), Not(306 <=
    2000/3*pos +
    2000*vel +
    15/2*(pos + vel)*(pos + vel) +
    2*force +
    15*pos*pos), Not(9/20 <= pos), Not(9/20 <= pos + vel), Not(908/3 <= 2000/3*pos + 15/2*pos*pos + 4000/3*vel), Not(154 <=
    1000/3*pos +
    15/4*(pos + vel)*(pos + vel) +
    15/2*pos*pos +
    1000*vel)]
conjuncts: [force >= -1, force <= 1, Not(181/400 <= pos + 2*vel + 3/2000*force + 9/800*pos*pos), Not(306 <=
    2000/3*pos +
    2000*vel +
    15/2*(pos + vel)*(pos + vel) +
    2*force +
    15*pos*pos), Not(313/2 <=
    1000/3*pos +
    4000/3*vel +
    15/4*
    (-1/400 + pos + 2*vel + 3/2000*force + 9/800*pos*pos)*
    (-1/400 + pos + 2*vel + 3/2000*force + 9/800*pos*pos) +
    15/2*(pos + vel)*(pos + vel) +
    3/2*force +
    45/4*pos*pos)]
SN noModels:
-- []
-- [force >= -1, force <= 1, Not(181/400 <= pos + 2*vel + 3/2000*force + 9/800*pos*pos), Not(306 <=
    2000/3*pos +
    2000*vel +
    15/2*(pos + vel)*(pos + vel) +
    2*force +
    15*pos*pos), Not(9/20 <= pos), Not(9/20 <= pos + vel), Not(908/3 <= 2000/3*pos + 15/2*pos*pos + 4000/3*vel), Not(154 <=
    1000/3*pos +
    15/4*(pos + vel)*(pos + vel) +
    15/2*pos*pos +
    1000*vel), Not(313/2 <=
    1000/3*pos +
    4000/3*vel +
    15/4*
    (-1/400 + pos + 2*vel + 3/2000*force + 9/800*pos*pos)*
    (-1/400 + pos + 2*vel + 3/2000*force + 9/800*pos*pos) +
    15/2*(pos + vel)*(pos + vel) +
    3/2*force +
    45/4*pos*pos)]
-- Not(force >= -1)
noModels chcked ante1
adding ante2 to solver
adding force >= -1
added
adding force <= 1
added
adding Not(181/400 <= pos + 2*vel + 3/2000*force + 9/800*pos*pos)
added
adding Not(306 <=
    2000/3*pos +
    2000*vel +
    15/2*(pos + vel)*(pos + vel) +
    2*force +
    15*pos*pos)
added
adding Not(9/20 <= pos)
added
adding Not(9/20 <= pos + vel)
added
adding Not(908/3 <= 2000/3*pos + 15/2*pos*pos + 4000/3*vel)
added
adding Not(154 <=
    1000/3*pos +
    15/4*(pos + vel)*(pos + vel) +
    15/2*pos*pos +
    1000*vel)
added
adding Not(313/2 <=
    1000/3*pos +
    4000/3*vel +
    15/4*
    (-1/400 + pos + 2*vel + 3/2000*force + 9/800*pos*pos)*
    (-1/400 + pos + 2*vel + 3/2000*force + 9/800*pos*pos) +
    15/2*(pos + vel)*(pos + vel) +
    3/2*force +
    45/4*pos*pos)
added
noModels check of ante1 /\ ante2: sat
added hyp
SN noModels:
-- []
-- [force >= -1, force <= 1, Not(181/400 <= pos + 2*vel + 3/2000*force + 9/800*pos*pos), Not(306 <=
    2000/3*pos +
    2000*vel +
    15/2*(pos + vel)*(pos + vel) +
    2*force +
    15*pos*pos), Not(9/20 <= pos), Not(9/20 <= pos + vel), Not(908/3 <= 2000/3*pos + 15/2*pos*pos + 4000/3*vel), Not(154 <=
    1000/3*pos +
    15/4*(pos + vel)*(pos + vel) +
    15/2*pos*pos +
    1000*vel), Not(313/2 <=
    1000/3*pos +
    4000/3*vel +
    15/4*
    (-1/400 + pos + 2*vel + 3/2000*force + 9/800*pos*pos)*
    (-1/400 + pos + 2*vel + 3/2000*force + 9/800*pos*pos) +
    15/2*(pos + vel)*(pos + vel) +
    3/2*force +
    45/4*pos*pos)]
-- Not(force <= 1)
noModels chcked ante1
adding ante2 to solver
adding force >= -1
added
adding force <= 1
added
adding Not(181/400 <= pos + 2*vel + 3/2000*force + 9/800*pos*pos)
added
adding Not(306 <=
    2000/3*pos +
    2000*vel +
    15/2*(pos + vel)*(pos + vel) +
    2*force +
    15*pos*pos)
added
adding Not(9/20 <= pos)
added
adding Not(9/20 <= pos + vel)
added
adding Not(908/3 <= 2000/3*pos + 15/2*pos*pos + 4000/3*vel)
added
adding Not(154 <=
    1000/3*pos +
    15/4*(pos + vel)*(pos + vel) +
    15/2*pos*pos +
    1000*vel)
added
adding Not(313/2 <=
    1000/3*pos +
    4000/3*vel +
    15/4*
    (-1/400 + pos + 2*vel + 3/2000*force + 9/800*pos*pos)*
    (-1/400 + pos + 2*vel + 3/2000*force + 9/800*pos*pos) +
    15/2*(pos + vel)*(pos + vel) +
    3/2*force +
    45/4*pos*pos)
added
noModels check of ante1 /\ ante2: sat
added hyp
SN noModels:
-- []
-- [force >= -1, force <= 1, Not(181/400 <= pos + 2*vel + 3/2000*force + 9/800*pos*pos), Not(306 <=
    2000/3*pos +
    2000*vel +
    15/2*(pos + vel)*(pos + vel) +
    2*force +
    15*pos*pos), Not(9/20 <= pos), Not(9/20 <= pos + vel), Not(908/3 <= 2000/3*pos + 15/2*pos*pos + 4000/3*vel), Not(154 <=
    1000/3*pos +
    15/4*(pos + vel)*(pos + vel) +
    15/2*pos*pos +
    1000*vel), Not(313/2 <=
    1000/3*pos +
    4000/3*vel +
    15/4*
    (-1/400 + pos + 2*vel + 3/2000*force + 9/800*pos*pos)*
    (-1/400 + pos + 2*vel + 3/2000*force + 9/800*pos*pos) +
    15/2*(pos + vel)*(pos + vel) +
    3/2*force +
    45/4*pos*pos)]
-- Not(Not(181/400 <=
        pos + 2*vel + 3/2000*force + 9/800*pos*pos))
noModels chcked ante1
adding ante2 to solver
adding force >= -1
added
adding force <= 1
added
adding Not(181/400 <= pos + 2*vel + 3/2000*force + 9/800*pos*pos)
added
adding Not(306 <=
    2000/3*pos +
    2000*vel +
    15/2*(pos + vel)*(pos + vel) +
    2*force +
    15*pos*pos)
added
adding Not(9/20 <= pos)
added
adding Not(9/20 <= pos + vel)
added
adding Not(908/3 <= 2000/3*pos + 15/2*pos*pos + 4000/3*vel)
added
adding Not(154 <=
    1000/3*pos +
    15/4*(pos + vel)*(pos + vel) +
    15/2*pos*pos +
    1000*vel)
added
adding Not(313/2 <=
    1000/3*pos +
    4000/3*vel +
    15/4*
    (-1/400 + pos + 2*vel + 3/2000*force + 9/800*pos*pos)*
    (-1/400 + pos + 2*vel + 3/2000*force + 9/800*pos*pos) +
    15/2*(pos + vel)*(pos + vel) +
    3/2*force +
    45/4*pos*pos)
added
noModels check of ante1 /\ ante2: sat
added hyp
SN noModels:
-- []
-- [force >= -1, force <= 1, Not(181/400 <= pos + 2*vel + 3/2000*force + 9/800*pos*pos), Not(306 <=
    2000/3*pos +
    2000*vel +
    15/2*(pos + vel)*(pos + vel) +
    2*force +
    15*pos*pos), Not(9/20 <= pos), Not(9/20 <= pos + vel), Not(908/3 <= 2000/3*pos + 15/2*pos*pos + 4000/3*vel), Not(154 <=
    1000/3*pos +
    15/4*(pos + vel)*(pos + vel) +
    15/2*pos*pos +
    1000*vel), Not(313/2 <=
    1000/3*pos +
    4000/3*vel +
    15/4*
    (-1/400 + pos + 2*vel + 3/2000*force + 9/800*pos*pos)*
    (-1/400 + pos + 2*vel + 3/2000*force + 9/800*pos*pos) +
    15/2*(pos + vel)*(pos + vel) +
    3/2*force +
    45/4*pos*pos)]
-- Not(Not(306 <=
        2000/3*pos +
        2000*vel +
        15/2*(pos + vel)*(pos + vel) +
        2*force +
        15*pos*pos))
noModels chcked ante1
adding ante2 to solver
adding force >= -1
added
adding force <= 1
added
adding Not(181/400 <= pos + 2*vel + 3/2000*force + 9/800*pos*pos)
added
adding Not(306 <=
    2000/3*pos +
    2000*vel +
    15/2*(pos + vel)*(pos + vel) +
    2*force +
    15*pos*pos)
added
adding Not(9/20 <= pos)
added
adding Not(9/20 <= pos + vel)
added
adding Not(908/3 <= 2000/3*pos + 15/2*pos*pos + 4000/3*vel)
added
adding Not(154 <=
    1000/3*pos +
    15/4*(pos + vel)*(pos + vel) +
    15/2*pos*pos +
    1000*vel)
added
adding Not(313/2 <=
    1000/3*pos +
    4000/3*vel +
    15/4*
    (-1/400 + pos + 2*vel + 3/2000*force + 9/800*pos*pos)*
    (-1/400 + pos + 2*vel + 3/2000*force + 9/800*pos*pos) +
    15/2*(pos + vel)*(pos + vel) +
    3/2*force +
    45/4*pos*pos)
added
noModels check of ante1 /\ ante2: sat
added hyp
SN noModels:
-- []
-- [force >= -1, force <= 1, Not(181/400 <= pos + 2*vel + 3/2000*force + 9/800*pos*pos), Not(306 <=
    2000/3*pos +
    2000*vel +
    15/2*(pos + vel)*(pos + vel) +
    2*force +
    15*pos*pos), Not(9/20 <= pos), Not(9/20 <= pos + vel), Not(908/3 <= 2000/3*pos + 15/2*pos*pos + 4000/3*vel), Not(154 <=
    1000/3*pos +
    15/4*(pos + vel)*(pos + vel) +
    15/2*pos*pos +
    1000*vel)]
-- Not(Not(313/2 <=
        1000/3*pos +
        4000/3*vel +
        15/4*
        (-1/400 + pos + 2*vel + 3/2000*force + 9/800*pos*pos)*
        (-1/400 + pos + 2*vel + 3/2000*force + 9/800*pos*pos) +
        15/2*(pos + vel)*(pos + vel) +
        3/2*force +
        45/4*pos*pos))
noModels chcked ante1
adding ante2 to solver
adding force >= -1
added
adding force <= 1
added
adding Not(181/400 <= pos + 2*vel + 3/2000*force + 9/800*pos*pos)
added
adding Not(306 <=
    2000/3*pos +
    2000*vel +
    15/2*(pos + vel)*(pos + vel) +
    2*force +
    15*pos*pos)
added
adding Not(9/20 <= pos)
added
adding Not(9/20 <= pos + vel)
added
adding Not(908/3 <= 2000/3*pos + 15/2*pos*pos + 4000/3*vel)
added
adding Not(154 <=
    1000/3*pos +
    15/4*(pos + vel)*(pos + vel) +
    15/2*pos*pos +
    1000*vel)
added
noModels check of ante1 /\ ante2: sat
added hyp
SN context: [force >= -1, force <= 1, Not(181/400 <= pos + 2*vel + 3/2000*force + 9/800*pos*pos), Not(306 <=
    2000/3*pos +
    2000*vel +
    15/2*(pos + vel)*(pos + vel) +
    2*force +
    15*pos*pos), Not(9/20 <= pos), Not(9/20 <= pos + vel), Not(908/3 <= 2000/3*pos + 15/2*pos*pos + 4000/3*vel), Not(154 <=
    1000/3*pos +
    15/4*(pos + vel)*(pos + vel) +
    15/2*pos*pos +
    1000*vel)]
SN negand: 313/2 <=
1000/3*pos +
4000/3*vel +
15/4*
(-1/400 + pos + 2*vel + 3/2000*force + 9/800*pos*pos)*
(-1/400 + pos + 2*vel + 3/2000*force + 9/800*pos*pos) +
15/2*(pos + vel)*(pos + vel) +
3/2*force +
45/4*pos*pos
SN noModels:
-- []
-- [force >= -1, force <= 1, Not(181/400 <= pos + 2*vel + 3/2000*force + 9/800*pos*pos), Not(306 <=
    2000/3*pos +
    2000*vel +
    15/2*(pos + vel)*(pos + vel) +
    2*force +
    15*pos*pos), Not(9/20 <= pos), Not(9/20 <= pos + vel), Not(908/3 <= 2000/3*pos + 15/2*pos*pos + 4000/3*vel), Not(154 <=
    1000/3*pos +
    15/4*(pos + vel)*(pos + vel) +
    15/2*pos*pos +
    1000*vel)]
-- Not(313/2 <=
    1000/3*pos +
    4000/3*vel +
    15/4*
    (-1/400 + pos + 2*vel + 3/2000*force + 9/800*pos*pos)*
    (-1/400 + pos + 2*vel + 3/2000*force + 9/800*pos*pos) +
    15/2*(pos + vel)*(pos + vel) +
    3/2*force +
    45/4*pos*pos)
noModels chcked ante1
adding ante2 to solver
adding force >= -1
added
adding force <= 1
added
adding Not(181/400 <= pos + 2*vel + 3/2000*force + 9/800*pos*pos)
added
adding Not(306 <=
    2000/3*pos +
    2000*vel +
    15/2*(pos + vel)*(pos + vel) +
    2*force +
    15*pos*pos)
added
adding Not(9/20 <= pos)
added
adding Not(9/20 <= pos + vel)
added
adding Not(908/3 <= 2000/3*pos + 15/2*pos*pos + 4000/3*vel)
added
adding Not(154 <=
    1000/3*pos +
    15/4*(pos + vel)*(pos + vel) +
    15/2*pos*pos +
    1000*vel)
added
noModels check of ante1 /\ ante2: sat
added hyp
new ctrl Pred (residue of ctrlPred over Inv) for action :
And(force >= -1,
    force <= 1,
    Not(181/400 <=
        pos + 2*vel + 3/2000*force + 9/800*pos*pos),
    Not(306 <=
        2000/3*pos +
        2000*vel +
        15/2*(pos + vel)*(pos + vel) +
        2*force +
        15*pos*pos),
    Not(313/2 <=
        1000/3*pos +
        4000/3*vel +
        15/4*
        (-1/400 + pos + 2*vel + 3/2000*force + 9/800*pos*pos)*
        (-1/400 + pos + 2*vel + 3/2000*force + 9/800*pos*pos) +
        15/2*(pos + vel)*(pos + vel) +
        3/2*force +
        45/4*pos*pos))
guardDisjunction: Or(False,
   Exists(force,
          And(force >= -1,
              force <= 1,
              Not(181/400 <=
                  pos + 2*vel + 3/2000*force + 9/800*pos*pos),
              Not(306 <=
                  2000/3*pos +
                  2000*vel +
                  15/2*(pos + vel)*(pos + vel) +
                  2*force +
                  15*pos*pos),
              Not(313/2 <=
                  1000/3*pos +
                  4000/3*vel +
                  15/4*
                  (-1/400 +
                   pos +
                   2*vel +
                   3/2000*force +
                   9/800*pos*pos)*
                  (-1/400 +
                   pos +
                   2*vel +
                   3/2000*force +
                   9/800*pos*pos) +
                  15/2*(pos + vel)*(pos + vel) +
                  3/2*force +
                  45/4*pos*pos))))
guardDisjunction after QE [Or(False,
    Exists(force,
           And(force >= -1,
               force <= 1,
               Not(181/400 <=
                   pos +
                   2*vel +
                   3/2000*force +
                   9/800*pos*pos),
               Not(306 <=
                   2000/3*pos +
                   2000*vel +
                   15/2*(pos + vel)*(pos + vel) +
                   2*force +
                   15*pos*pos),
               Not(313/2 <=
                   1000/3*pos +
                   4000/3*vel +
                   15/4*
                   (-1/400 +
                    pos +
                    2*vel +
                    3/2000*force +
                    9/800*pos*pos)*
                   (-1/400 +
                    pos +
                    2*vel +
                    3/2000*force +
                    9/800*pos*pos) +
                   15/2*(pos + vel)*(pos + vel) +
                   3/2*force +
                   45/4*pos*pos))))]
simplified guardDisjunction:
 Exists(force,
       And(force >= -1,
           force <= 1,
           Not(181/400 <=
               pos + 2*vel + 3/2000*force + 9/800*pos*pos),
           Not(306 <=
               2000/3*pos +
               2000*vel +
               15/2*(pos + vel)*(pos + vel) +
               2*force +
               15*pos*pos),
           Not(313/2 <=
               1000/3*pos +
               4000/3*vel +
               15/4*
               (-1/400 +
                pos +
                2*vel +
                3/2000*force +
                9/800*pos*pos)*
               (-1/400 +
                pos +
                2*vel +
                3/2000*force +
                9/800*pos*pos) +
               15/2*(pos + vel)*(pos + vel) +
               3/2*force +
               45/4*pos*pos)))

- Completed arcs from node singleton. Now checking if invariant needs updating (control preds may also get updated again)
stateInv
 And(Not(9/20 <= pos),
    Not(9/20 <= pos + vel),
    Not(908/3 <= 2000/3*pos + 15/2*pos*pos + 4000/3*vel),
    Not(154 <=
        1000/3*pos +
        15/4*(pos + vel)*(pos + vel) +
        15/2*pos*pos +
        1000*vel))
SN noModels:
-- []
-- [Not(9/20 <= pos), Not(9/20 <= pos + vel), Not(908/3 <= 2000/3*pos + 15/2*pos*pos + 4000/3*vel), Not(154 <=
    1000/3*pos +
    15/4*(pos + vel)*(pos + vel) +
    15/2*pos*pos +
    1000*vel)]
-- Not(Exists(force,
           And(force >= -1,
               force <= 1,
               Not(181/400 <=
                   pos +
                   2*vel +
                   3/2000*force +
                   9/800*pos*pos),
               Not(306 <=
                   2000/3*pos +
                   2000*vel +
                   15/2*(pos + vel)*(pos + vel) +
                   2*force +
                   15*pos*pos),
               Not(313/2 <=
                   1000/3*pos +
                   4000/3*vel +
                   15/4*
                   (-1/400 +
                    pos +
                    2*vel +
                    3/2000*force +
                    9/800*pos*pos)*
                   (-1/400 +
                    pos +
                    2*vel +
                    3/2000*force +
                    9/800*pos*pos) +
                   15/2*(pos + vel)*(pos + vel) +
                   3/2*force +
                   45/4*pos*pos))))
noModels chcked ante1
adding ante2 to solver
adding Not(9/20 <= pos)
added
adding Not(9/20 <= pos + vel)
added
adding Not(908/3 <= 2000/3*pos + 15/2*pos*pos + 4000/3*vel)
added
adding Not(154 <=
    1000/3*pos +
    15/4*(pos + vel)*(pos + vel) +
    15/2*pos*pos +
    1000*vel)
added
noModels check of ante1 /\ ante2: sat
added hyp
newInvDelta:
 [Exists(force,
        And(force >= -1,
            force <= 1,
            Not(181/400 <=
                pos + 2*vel + 3/2000*force + 9/800*pos*pos),
            Not(306 <=
                2000/3*pos +
                2000*vel +
                15/2*(pos + vel)*(pos + vel) +
                2*force +
                15*pos*pos),
            Not(313/2 <=
                1000/3*pos +
                4000/3*vel +
                15/4*
                (-1/400 +
                 pos +
                 2*vel +
                 3/2000*force +
                 9/800*pos*pos)*
                (-1/400 +
                 pos +
                 2*vel +
                 3/2000*force +
                 9/800*pos*pos) +
                15/2*(pos + vel)*(pos + vel) +
                3/2*force +
                45/4*pos*pos)))]
SN noModels:
-- []
-- [True]
-- Not(And(And(Not(9/20 <= pos),
            Not(9/20 <= pos + vel),
            Not(908/3 <=
                2000/3*pos + 15/2*pos*pos + 4000/3*vel),
            Not(154 <=
                1000/3*pos +
                15/4*(pos + vel)*(pos + vel) +
                15/2*pos*pos +
                1000*vel)),
        Exists(force,
               And(force >= -1,
                   force <= 1,
                   Not(181/400 <=
                       pos +
                       2*vel +
                       3/2000*force +
                       9/800*pos*pos),
                   Not(306 <=
                       2000/3*pos +
                       2000*vel +
                       15/2*(pos + vel)*(pos + vel) +
                       2*force +
                       15*pos*pos),
                   Not(313/2 <=
                       1000/3*pos +
                       4000/3*vel +
                       15/4*
                       (-1/400 +
                        pos +
                        2*vel +
                        3/2000*force +
                        9/800*pos*pos)*
                       (-1/400 +
                        pos +
                        2*vel +
                        3/2000*force +
                        9/800*pos*pos) +
                       15/2*(pos + vel)*(pos + vel) +
                       3/2*force +
                       45/4*pos*pos)))))
noModels chcked ante1
adding ante2 to solver
adding True
added
noModels check of ante1 /\ ante2: sat
added hyp
SC context: [True]
conjuncts: [And(Not(9/20 <= pos),
    Not(9/20 <= pos + vel),
    Not(908/3 <= 2000/3*pos + 15/2*pos*pos + 4000/3*vel),
    Not(154 <=
        1000/3*pos +
        15/4*(pos + vel)*(pos + vel) +
        15/2*pos*pos +
        1000*vel)), Exists(force,
       And(force >= -1,
           force <= 1,
           Not(181/400 <=
               pos + 2*vel + 3/2000*force + 9/800*pos*pos),
           Not(306 <=
               2000/3*pos +
               2000*vel +
               15/2*(pos + vel)*(pos + vel) +
               2*force +
               15*pos*pos),
           Not(313/2 <=
               1000/3*pos +
               4000/3*vel +
               15/4*
               (-1/400 +
                pos +
                2*vel +
                3/2000*force +
                9/800*pos*pos)*
               (-1/400 +
                pos +
                2*vel +
                3/2000*force +
                9/800*pos*pos) +
               15/2*(pos + vel)*(pos + vel) +
               3/2*force +
               45/4*pos*pos)))]
SN noModels:
-- []
-- [Exists(force,
       And(force >= -1,
           force <= 1,
           Not(181/400 <=
               pos + 2*vel + 3/2000*force + 9/800*pos*pos),
           Not(306 <=
               2000/3*pos +
               2000*vel +
               15/2*(pos + vel)*(pos + vel) +
               2*force +
               15*pos*pos),
           Not(313/2 <=
               1000/3*pos +
               4000/3*vel +
               15/4*
               (-1/400 +
                pos +
                2*vel +
                3/2000*force +
                9/800*pos*pos)*
               (-1/400 +
                pos +
                2*vel +
                3/2000*force +
                9/800*pos*pos) +
               15/2*(pos + vel)*(pos + vel) +
               3/2*force +
               45/4*pos*pos)))]
-- Not(And(Not(9/20 <= pos),
        Not(9/20 <= pos + vel),
        Not(908/3 <= 2000/3*pos + 15/2*pos*pos + 4000/3*vel),
        Not(154 <=
            1000/3*pos +
            15/4*(pos + vel)*(pos + vel) +
            15/2*pos*pos +
            1000*vel)))
noModels chcked ante1
adding ante2 to solver
adding Exists(force,
       And(force >= -1,
           force <= 1,
           Not(181/400 <=
               pos + 2*vel + 3/2000*force + 9/800*pos*pos),
           Not(306 <=
               2000/3*pos +
               2000*vel +
               15/2*(pos + vel)*(pos + vel) +
               2*force +
               15*pos*pos),
           Not(313/2 <=
               1000/3*pos +
               4000/3*vel +
               15/4*
               (-1/400 +
                pos +
                2*vel +
                3/2000*force +
                9/800*pos*pos)*
               (-1/400 +
                pos +
                2*vel +
                3/2000*force +
                9/800*pos*pos) +
               15/2*(pos + vel)*(pos + vel) +
               3/2*force +
               45/4*pos*pos)))
added
noModels check of ante1 /\ ante2: sat
added hyp
SC context: [Exists(force,
       And(force >= -1,
           force <= 1,
           Not(181/400 <=
               pos + 2*vel + 3/2000*force + 9/800*pos*pos),
           Not(306 <=
               2000/3*pos +
               2000*vel +
               15/2*(pos + vel)*(pos + vel) +
               2*force +
               15*pos*pos),
           Not(313/2 <=
               1000/3*pos +
               4000/3*vel +
               15/4*
               (-1/400 +
                pos +
                2*vel +
                3/2000*force +
                9/800*pos*pos)*
               (-1/400 +
                pos +
                2*vel +
                3/2000*force +
                9/800*pos*pos) +
               15/2*(pos + vel)*(pos + vel) +
               3/2*force +
               45/4*pos*pos)))]
conjuncts: [Not(9/20 <= pos), Not(9/20 <= pos + vel), Not(908/3 <= 2000/3*pos + 15/2*pos*pos + 4000/3*vel), Not(154 <=
    1000/3*pos +
    15/4*(pos + vel)*(pos + vel) +
    15/2*pos*pos +
    1000*vel)]
SN noModels:
-- []
-- [Exists(force,
       And(force >= -1,
           force <= 1,
           Not(181/400 <=
               pos + 2*vel + 3/2000*force + 9/800*pos*pos),
           Not(306 <=
               2000/3*pos +
               2000*vel +
               15/2*(pos + vel)*(pos + vel) +
               2*force +
               15*pos*pos),
           Not(313/2 <=
               1000/3*pos +
               4000/3*vel +
               15/4*
               (-1/400 +
                pos +
                2*vel +
                3/2000*force +
                9/800*pos*pos)*
               (-1/400 +
                pos +
                2*vel +
                3/2000*force +
                9/800*pos*pos) +
               15/2*(pos + vel)*(pos + vel) +
               3/2*force +
               45/4*pos*pos))), Not(9/20 <= pos + vel), Not(908/3 <= 2000/3*pos + 15/2*pos*pos + 4000/3*vel), Not(154 <=
    1000/3*pos +
    15/4*(pos + vel)*(pos + vel) +
    15/2*pos*pos +
    1000*vel)]
-- Not(Not(9/20 <= pos))
noModels chcked ante1
adding ante2 to solver
adding Exists(force,
       And(force >= -1,
           force <= 1,
           Not(181/400 <=
               pos + 2*vel + 3/2000*force + 9/800*pos*pos),
           Not(306 <=
               2000/3*pos +
               2000*vel +
               15/2*(pos + vel)*(pos + vel) +
               2*force +
               15*pos*pos),
           Not(313/2 <=
               1000/3*pos +
               4000/3*vel +
               15/4*
               (-1/400 +
                pos +
                2*vel +
                3/2000*force +
                9/800*pos*pos)*
               (-1/400 +
                pos +
                2*vel +
                3/2000*force +
                9/800*pos*pos) +
               15/2*(pos + vel)*(pos + vel) +
               3/2*force +
               45/4*pos*pos)))
added
adding Not(9/20 <= pos + vel)
added
adding Not(908/3 <= 2000/3*pos + 15/2*pos*pos + 4000/3*vel)
added
adding Not(154 <=
    1000/3*pos +
    15/4*(pos + vel)*(pos + vel) +
    15/2*pos*pos +
    1000*vel)
added
noModels check of ante1 /\ ante2: sat
added hyp
SN context: [Exists(force,
       And(force >= -1,
           force <= 1,
           Not(181/400 <=
               pos + 2*vel + 3/2000*force + 9/800*pos*pos),
           Not(306 <=
               2000/3*pos +
               2000*vel +
               15/2*(pos + vel)*(pos + vel) +
               2*force +
               15*pos*pos),
           Not(313/2 <=
               1000/3*pos +
               4000/3*vel +
               15/4*
               (-1/400 +
                pos +
                2*vel +
                3/2000*force +
                9/800*pos*pos)*
               (-1/400 +
                pos +
                2*vel +
                3/2000*force +
                9/800*pos*pos) +
               15/2*(pos + vel)*(pos + vel) +
               3/2*force +
               45/4*pos*pos))), Not(9/20 <= pos + vel), Not(908/3 <= 2000/3*pos + 15/2*pos*pos + 4000/3*vel), Not(154 <=
    1000/3*pos +
    15/4*(pos + vel)*(pos + vel) +
    15/2*pos*pos +
    1000*vel)]
SN negand: 9/20 <= pos
SN noModels:
-- []
-- [Exists(force,
       And(force >= -1,
           force <= 1,
           Not(181/400 <=
               pos + 2*vel + 3/2000*force + 9/800*pos*pos),
           Not(306 <=
               2000/3*pos +
               2000*vel +
               15/2*(pos + vel)*(pos + vel) +
               2*force +
               15*pos*pos),
           Not(313/2 <=
               1000/3*pos +
               4000/3*vel +
               15/4*
               (-1/400 +
                pos +
                2*vel +
                3/2000*force +
                9/800*pos*pos)*
               (-1/400 +
                pos +
                2*vel +
                3/2000*force +
                9/800*pos*pos) +
               15/2*(pos + vel)*(pos + vel) +
               3/2*force +
               45/4*pos*pos))), Not(9/20 <= pos + vel), Not(908/3 <= 2000/3*pos + 15/2*pos*pos + 4000/3*vel), Not(154 <=
    1000/3*pos +
    15/4*(pos + vel)*(pos + vel) +
    15/2*pos*pos +
    1000*vel)]
-- Not(9/20 <= pos)
noModels chcked ante1
adding ante2 to solver
adding Exists(force,
       And(force >= -1,
           force <= 1,
           Not(181/400 <=
               pos + 2*vel + 3/2000*force + 9/800*pos*pos),
           Not(306 <=
               2000/3*pos +
               2000*vel +
               15/2*(pos + vel)*(pos + vel) +
               2*force +
               15*pos*pos),
           Not(313/2 <=
               1000/3*pos +
               4000/3*vel +
               15/4*
               (-1/400 +
                pos +
                2*vel +
                3/2000*force +
                9/800*pos*pos)*
               (-1/400 +
                pos +
                2*vel +
                3/2000*force +
                9/800*pos*pos) +
               15/2*(pos + vel)*(pos + vel) +
               3/2*force +
               45/4*pos*pos)))
added
adding Not(9/20 <= pos + vel)
added
adding Not(908/3 <= 2000/3*pos + 15/2*pos*pos + 4000/3*vel)
added
adding Not(154 <=
    1000/3*pos +
    15/4*(pos + vel)*(pos + vel) +
    15/2*pos*pos +
    1000*vel)
added
noModels check of ante1 /\ ante2: sat
added hyp
SN noModels:
-- []
-- [Exists(force,
       And(force >= -1,
           force <= 1,
           Not(181/400 <=
               pos + 2*vel + 3/2000*force + 9/800*pos*pos),
           Not(306 <=
               2000/3*pos +
               2000*vel +
               15/2*(pos + vel)*(pos + vel) +
               2*force +
               15*pos*pos),
           Not(313/2 <=
               1000/3*pos +
               4000/3*vel +
               15/4*
               (-1/400 +
                pos +
                2*vel +
                3/2000*force +
                9/800*pos*pos)*
               (-1/400 +
                pos +
                2*vel +
                3/2000*force +
                9/800*pos*pos) +
               15/2*(pos + vel)*(pos + vel) +
               3/2*force +
               45/4*pos*pos))), Not(9/20 <= pos), Not(908/3 <= 2000/3*pos + 15/2*pos*pos + 4000/3*vel), Not(154 <=
    1000/3*pos +
    15/4*(pos + vel)*(pos + vel) +
    15/2*pos*pos +
    1000*vel)]
-- Not(Not(9/20 <= pos + vel))
noModels chcked ante1
adding ante2 to solver
adding Exists(force,
       And(force >= -1,
           force <= 1,
           Not(181/400 <=
               pos + 2*vel + 3/2000*force + 9/800*pos*pos),
           Not(306 <=
               2000/3*pos +
               2000*vel +
               15/2*(pos + vel)*(pos + vel) +
               2*force +
               15*pos*pos),
           Not(313/2 <=
               1000/3*pos +
               4000/3*vel +
               15/4*
               (-1/400 +
                pos +
                2*vel +
                3/2000*force +
                9/800*pos*pos)*
               (-1/400 +
                pos +
                2*vel +
                3/2000*force +
                9/800*pos*pos) +
               15/2*(pos + vel)*(pos + vel) +
               3/2*force +
               45/4*pos*pos)))
added
adding Not(9/20 <= pos)
added
adding Not(908/3 <= 2000/3*pos + 15/2*pos*pos + 4000/3*vel)
added
adding Not(154 <=
    1000/3*pos +
    15/4*(pos + vel)*(pos + vel) +
    15/2*pos*pos +
    1000*vel)
added
noModels check of ante1 /\ ante2: sat
added hyp
SN context: [Exists(force,
       And(force >= -1,
           force <= 1,
           Not(181/400 <=
               pos + 2*vel + 3/2000*force + 9/800*pos*pos),
           Not(306 <=
               2000/3*pos +
               2000*vel +
               15/2*(pos + vel)*(pos + vel) +
               2*force +
               15*pos*pos),
           Not(313/2 <=
               1000/3*pos +
               4000/3*vel +
               15/4*
               (-1/400 +
                pos +
                2*vel +
                3/2000*force +
                9/800*pos*pos)*
               (-1/400 +
                pos +
                2*vel +
                3/2000*force +
                9/800*pos*pos) +
               15/2*(pos + vel)*(pos + vel) +
               3/2*force +
               45/4*pos*pos))), Not(9/20 <= pos), Not(908/3 <= 2000/3*pos + 15/2*pos*pos + 4000/3*vel), Not(154 <=
    1000/3*pos +
    15/4*(pos + vel)*(pos + vel) +
    15/2*pos*pos +
    1000*vel)]
SN negand: 9/20 <= pos + vel
SN noModels:
-- []
-- [Exists(force,
       And(force >= -1,
           force <= 1,
           Not(181/400 <=
               pos + 2*vel + 3/2000*force + 9/800*pos*pos),
           Not(306 <=
               2000/3*pos +
               2000*vel +
               15/2*(pos + vel)*(pos + vel) +
               2*force +
               15*pos*pos),
           Not(313/2 <=
               1000/3*pos +
               4000/3*vel +
               15/4*
               (-1/400 +
                pos +
                2*vel +
                3/2000*force +
                9/800*pos*pos)*
               (-1/400 +
                pos +
                2*vel +
                3/2000*force +
                9/800*pos*pos) +
               15/2*(pos + vel)*(pos + vel) +
               3/2*force +
               45/4*pos*pos))), Not(9/20 <= pos), Not(908/3 <= 2000/3*pos + 15/2*pos*pos + 4000/3*vel), Not(154 <=
    1000/3*pos +
    15/4*(pos + vel)*(pos + vel) +
    15/2*pos*pos +
    1000*vel)]
-- Not(9/20 <= pos + vel)
noModels chcked ante1
adding ante2 to solver
adding Exists(force,
       And(force >= -1,
           force <= 1,
           Not(181/400 <=
               pos + 2*vel + 3/2000*force + 9/800*pos*pos),
           Not(306 <=
               2000/3*pos +
               2000*vel +
               15/2*(pos + vel)*(pos + vel) +
               2*force +
               15*pos*pos),
           Not(313/2 <=
               1000/3*pos +
               4000/3*vel +
               15/4*
               (-1/400 +
                pos +
                2*vel +
                3/2000*force +
                9/800*pos*pos)*
               (-1/400 +
                pos +
                2*vel +
                3/2000*force +
                9/800*pos*pos) +
               15/2*(pos + vel)*(pos + vel) +
               3/2*force +
               45/4*pos*pos)))
added
adding Not(9/20 <= pos)
added
adding Not(908/3 <= 2000/3*pos + 15/2*pos*pos + 4000/3*vel)
added
adding Not(154 <=
    1000/3*pos +
    15/4*(pos + vel)*(pos + vel) +
    15/2*pos*pos +
    1000*vel)
added
noModels check of ante1 /\ ante2: sat
added hyp
SN noModels:
-- []
-- [Exists(force,
       And(force >= -1,
           force <= 1,
           Not(181/400 <=
               pos + 2*vel + 3/2000*force + 9/800*pos*pos),
           Not(306 <=
               2000/3*pos +
               2000*vel +
               15/2*(pos + vel)*(pos + vel) +
               2*force +
               15*pos*pos),
           Not(313/2 <=
               1000/3*pos +
               4000/3*vel +
               15/4*
               (-1/400 +
                pos +
                2*vel +
                3/2000*force +
                9/800*pos*pos)*
               (-1/400 +
                pos +
                2*vel +
                3/2000*force +
                9/800*pos*pos) +
               15/2*(pos + vel)*(pos + vel) +
               3/2*force +
               45/4*pos*pos))), Not(9/20 <= pos), Not(9/20 <= pos + vel), Not(154 <=
    1000/3*pos +
    15/4*(pos + vel)*(pos + vel) +
    15/2*pos*pos +
    1000*vel)]
-- Not(Not(908/3 <= 2000/3*pos + 15/2*pos*pos + 4000/3*vel))
noModels chcked ante1
adding ante2 to solver
adding Exists(force,
       And(force >= -1,
           force <= 1,
           Not(181/400 <=
               pos + 2*vel + 3/2000*force + 9/800*pos*pos),
           Not(306 <=
               2000/3*pos +
               2000*vel +
               15/2*(pos + vel)*(pos + vel) +
               2*force +
               15*pos*pos),
           Not(313/2 <=
               1000/3*pos +
               4000/3*vel +
               15/4*
               (-1/400 +
                pos +
                2*vel +
                3/2000*force +
                9/800*pos*pos)*
               (-1/400 +
                pos +
                2*vel +
                3/2000*force +
                9/800*pos*pos) +
               15/2*(pos + vel)*(pos + vel) +
               3/2*force +
               45/4*pos*pos)))
added
adding Not(9/20 <= pos)
added
adding Not(9/20 <= pos + vel)
added
adding Not(154 <=
    1000/3*pos +
    15/4*(pos + vel)*(pos + vel) +
    15/2*pos*pos +
    1000*vel)
added
noModels check of ante1 /\ ante2: sat
added hyp
SN noModels:
-- []
-- [Exists(force,
       And(force >= -1,
           force <= 1,
           Not(181/400 <=
               pos + 2*vel + 3/2000*force + 9/800*pos*pos),
           Not(306 <=
               2000/3*pos +
               2000*vel +
               15/2*(pos + vel)*(pos + vel) +
               2*force +
               15*pos*pos),
           Not(313/2 <=
               1000/3*pos +
               4000/3*vel +
               15/4*
               (-1/400 +
                pos +
                2*vel +
                3/2000*force +
                9/800*pos*pos)*
               (-1/400 +
                pos +
                2*vel +
                3/2000*force +
                9/800*pos*pos) +
               15/2*(pos + vel)*(pos + vel) +
               3/2*force +
               45/4*pos*pos))), Not(9/20 <= pos), Not(9/20 <= pos + vel)]
-- Not(Not(154 <=
        1000/3*pos +
        15/4*(pos + vel)*(pos + vel) +
        15/2*pos*pos +
        1000*vel))
noModels chcked ante1
adding ante2 to solver
adding Exists(force,
       And(force >= -1,
           force <= 1,
           Not(181/400 <=
               pos + 2*vel + 3/2000*force + 9/800*pos*pos),
           Not(306 <=
               2000/3*pos +
               2000*vel +
               15/2*(pos + vel)*(pos + vel) +
               2*force +
               15*pos*pos),
           Not(313/2 <=
               1000/3*pos +
               4000/3*vel +
               15/4*
               (-1/400 +
                pos +
                2*vel +
                3/2000*force +
                9/800*pos*pos)*
               (-1/400 +
                pos +
                2*vel +
                3/2000*force +
                9/800*pos*pos) +
               15/2*(pos + vel)*(pos + vel) +
               3/2*force +
               45/4*pos*pos)))
added
adding Not(9/20 <= pos)
added
adding Not(9/20 <= pos + vel)
added
noModels check of ante1 /\ ante2: sat
added hyp
SN noModels:
-- []
-- [Not(9/20 <= pos), Not(9/20 <= pos + vel)]
-- Not(Exists(force,
           And(force >= -1,
               force <= 1,
               Not(181/400 <=
                   pos +
                   2*vel +
                   3/2000*force +
                   9/800*pos*pos),
               Not(306 <=
                   2000/3*pos +
                   2000*vel +
                   15/2*(pos + vel)*(pos + vel) +
                   2*force +
                   15*pos*pos),
               Not(313/2 <=
                   1000/3*pos +
                   4000/3*vel +
                   15/4*
                   (-1/400 +
                    pos +
                    2*vel +
                    3/2000*force +
                    9/800*pos*pos)*
                   (-1/400 +
                    pos +
                    2*vel +
                    3/2000*force +
                    9/800*pos*pos) +
                   15/2*(pos + vel)*(pos + vel) +
                   3/2*force +
                   45/4*pos*pos))))
noModels chcked ante1
adding ante2 to solver
adding Not(9/20 <= pos)
added
adding Not(9/20 <= pos + vel)
added
noModels check of ante1 /\ ante2: sat
added hyp
new invariant
 And(Not(9/20 <= pos),
    Not(9/20 <= pos + vel),
    Exists(force,
           And(force >= -1,
               force <= 1,
               Not(181/400 <=
                   pos +
                   2*vel +
                   3/2000*force +
                   9/800*pos*pos),
               Not(306 <=
                   2000/3*pos +
                   2000*vel +
                   15/2*(pos + vel)*(pos + vel) +
                   2*force +
                   15*pos*pos),
               Not(313/2 <=
                   1000/3*pos +
                   4000/3*vel +
                   15/4*
                   (-1/400 +
                    pos +
                    2*vel +
                    3/2000*force +
                    9/800*pos*pos)*
                   (-1/400 +
                    pos +
                    2*vel +
                    3/2000*force +
                    9/800*pos*pos) +
                   15/2*(pos + vel)*(pos + vel) +
                   3/2*force +
                   45/4*pos*pos))))
SN noModels:
-- []
-- [Not(9/20 <= pos), Not(9/20 <= pos + vel), Exists(force,
       And(force >= -1,
           force <= 1,
           Not(181/400 <=
               pos + 2*vel + 3/2000*force + 9/800*pos*pos),
           Not(306 <=
               2000/3*pos +
               2000*vel +
               15/2*(pos + vel)*(pos + vel) +
               2*force +
               15*pos*pos),
           Not(313/2 <=
               1000/3*pos +
               4000/3*vel +
               15/4*
               (-1/400 +
                pos +
                2*vel +
                3/2000*force +
                9/800*pos*pos)*
               (-1/400 +
                pos +
                2*vel +
                3/2000*force +
                9/800*pos*pos) +
               15/2*(pos + vel)*(pos + vel) +
               3/2*force +
               45/4*pos*pos)))]
-- Not(And(force >= -1,
        force <= 1,
        Not(181/400 <=
            pos + 2*vel + 3/2000*force + 9/800*pos*pos),
        Not(306 <=
            2000/3*pos +
            2000*vel +
            15/2*(pos + vel)*(pos + vel) +
            2*force +
            15*pos*pos),
        Not(313/2 <=
            1000/3*pos +
            4000/3*vel +
            15/4*
            (-1/400 +
             pos +
             2*vel +
             3/2000*force +
             9/800*pos*pos)*
            (-1/400 +
             pos +
             2*vel +
             3/2000*force +
             9/800*pos*pos) +
            15/2*(pos + vel)*(pos + vel) +
            3/2*force +
            45/4*pos*pos)))
noModels chcked ante1
adding ante2 to solver
adding Not(9/20 <= pos)
added
adding Not(9/20 <= pos + vel)
added
adding Exists(force,
       And(force >= -1,
           force <= 1,
           Not(181/400 <=
               pos + 2*vel + 3/2000*force + 9/800*pos*pos),
           Not(306 <=
               2000/3*pos +
               2000*vel +
               15/2*(pos + vel)*(pos + vel) +
               2*force +
               15*pos*pos),
           Not(313/2 <=
               1000/3*pos +
               4000/3*vel +
               15/4*
               (-1/400 +
                pos +
                2*vel +
                3/2000*force +
                9/800*pos*pos)*
               (-1/400 +
                pos +
                2*vel +
                3/2000*force +
                9/800*pos*pos) +
               15/2*(pos + vel)*(pos + vel) +
               3/2*force +
               45/4*pos*pos)))
added
noModels check of ante1 /\ ante2: sat
added hyp
SC context: [Not(9/20 <= pos), Not(9/20 <= pos + vel), Exists(force,
       And(force >= -1,
           force <= 1,
           Not(181/400 <=
               pos + 2*vel + 3/2000*force + 9/800*pos*pos),
           Not(306 <=
               2000/3*pos +
               2000*vel +
               15/2*(pos + vel)*(pos + vel) +
               2*force +
               15*pos*pos),
           Not(313/2 <=
               1000/3*pos +
               4000/3*vel +
               15/4*
               (-1/400 +
                pos +
                2*vel +
                3/2000*force +
                9/800*pos*pos)*
               (-1/400 +
                pos +
                2*vel +
                3/2000*force +
                9/800*pos*pos) +
               15/2*(pos + vel)*(pos + vel) +
               3/2*force +
               45/4*pos*pos)))]
conjuncts: [force >= -1, force <= 1, Not(181/400 <= pos + 2*vel + 3/2000*force + 9/800*pos*pos), Not(306 <=
    2000/3*pos +
    2000*vel +
    15/2*(pos + vel)*(pos + vel) +
    2*force +
    15*pos*pos), Not(313/2 <=
    1000/3*pos +
    4000/3*vel +
    15/4*
    (-1/400 + pos + 2*vel + 3/2000*force + 9/800*pos*pos)*
    (-1/400 + pos + 2*vel + 3/2000*force + 9/800*pos*pos) +
    15/2*(pos + vel)*(pos + vel) +
    3/2*force +
    45/4*pos*pos)]
SN noModels:
-- []
-- [Not(9/20 <= pos), Not(9/20 <= pos + vel), Exists(force,
       And(force >= -1,
           force <= 1,
           Not(181/400 <=
               pos + 2*vel + 3/2000*force + 9/800*pos*pos),
           Not(306 <=
               2000/3*pos +
               2000*vel +
               15/2*(pos + vel)*(pos + vel) +
               2*force +
               15*pos*pos),
           Not(313/2 <=
               1000/3*pos +
               4000/3*vel +
               15/4*
               (-1/400 +
                pos +
                2*vel +
                3/2000*force +
                9/800*pos*pos)*
               (-1/400 +
                pos +
                2*vel +
                3/2000*force +
                9/800*pos*pos) +
               15/2*(pos + vel)*(pos + vel) +
               3/2*force +
               45/4*pos*pos))), force <= 1, Not(181/400 <= pos + 2*vel + 3/2000*force + 9/800*pos*pos), Not(306 <=
    2000/3*pos +
    2000*vel +
    15/2*(pos + vel)*(pos + vel) +
    2*force +
    15*pos*pos), Not(313/2 <=
    1000/3*pos +
    4000/3*vel +
    15/4*
    (-1/400 + pos + 2*vel + 3/2000*force + 9/800*pos*pos)*
    (-1/400 + pos + 2*vel + 3/2000*force + 9/800*pos*pos) +
    15/2*(pos + vel)*(pos + vel) +
    3/2*force +
    45/4*pos*pos)]
-- Not(force >= -1)
noModels chcked ante1
adding ante2 to solver
adding Not(9/20 <= pos)
added
adding Not(9/20 <= pos + vel)
added
adding Exists(force,
       And(force >= -1,
           force <= 1,
           Not(181/400 <=
               pos + 2*vel + 3/2000*force + 9/800*pos*pos),
           Not(306 <=
               2000/3*pos +
               2000*vel +
               15/2*(pos + vel)*(pos + vel) +
               2*force +
               15*pos*pos),
           Not(313/2 <=
               1000/3*pos +
               4000/3*vel +
               15/4*
               (-1/400 +
                pos +
                2*vel +
                3/2000*force +
                9/800*pos*pos)*
               (-1/400 +
                pos +
                2*vel +
                3/2000*force +
                9/800*pos*pos) +
               15/2*(pos + vel)*(pos + vel) +
               3/2*force +
               45/4*pos*pos)))
added
adding force <= 1
added
adding Not(181/400 <= pos + 2*vel + 3/2000*force + 9/800*pos*pos)
added
adding Not(306 <=
    2000/3*pos +
    2000*vel +
    15/2*(pos + vel)*(pos + vel) +
    2*force +
    15*pos*pos)
added
adding Not(313/2 <=
    1000/3*pos +
    4000/3*vel +
    15/4*
    (-1/400 + pos + 2*vel + 3/2000*force + 9/800*pos*pos)*
    (-1/400 + pos + 2*vel + 3/2000*force + 9/800*pos*pos) +
    15/2*(pos + vel)*(pos + vel) +
    3/2*force +
    45/4*pos*pos)
added
noModels check of ante1 /\ ante2: sat
added hyp
SN noModels:
-- []
-- [Not(9/20 <= pos), Not(9/20 <= pos + vel), Exists(force,
       And(force >= -1,
           force <= 1,
           Not(181/400 <=
               pos + 2*vel + 3/2000*force + 9/800*pos*pos),
           Not(306 <=
               2000/3*pos +
               2000*vel +
               15/2*(pos + vel)*(pos + vel) +
               2*force +
               15*pos*pos),
           Not(313/2 <=
               1000/3*pos +
               4000/3*vel +
               15/4*
               (-1/400 +
                pos +
                2*vel +
                3/2000*force +
                9/800*pos*pos)*
               (-1/400 +
                pos +
                2*vel +
                3/2000*force +
                9/800*pos*pos) +
               15/2*(pos + vel)*(pos + vel) +
               3/2*force +
               45/4*pos*pos))), force >= -1, Not(181/400 <= pos + 2*vel + 3/2000*force + 9/800*pos*pos), Not(306 <=
    2000/3*pos +
    2000*vel +
    15/2*(pos + vel)*(pos + vel) +
    2*force +
    15*pos*pos), Not(313/2 <=
    1000/3*pos +
    4000/3*vel +
    15/4*
    (-1/400 + pos + 2*vel + 3/2000*force + 9/800*pos*pos)*
    (-1/400 + pos + 2*vel + 3/2000*force + 9/800*pos*pos) +
    15/2*(pos + vel)*(pos + vel) +
    3/2*force +
    45/4*pos*pos)]
-- Not(force <= 1)
noModels chcked ante1
adding ante2 to solver
adding Not(9/20 <= pos)
added
adding Not(9/20 <= pos + vel)
added
adding Exists(force,
       And(force >= -1,
           force <= 1,
           Not(181/400 <=
               pos + 2*vel + 3/2000*force + 9/800*pos*pos),
           Not(306 <=
               2000/3*pos +
               2000*vel +
               15/2*(pos + vel)*(pos + vel) +
               2*force +
               15*pos*pos),
           Not(313/2 <=
               1000/3*pos +
               4000/3*vel +
               15/4*
               (-1/400 +
                pos +
                2*vel +
                3/2000*force +
                9/800*pos*pos)*
               (-1/400 +
                pos +
                2*vel +
                3/2000*force +
                9/800*pos*pos) +
               15/2*(pos + vel)*(pos + vel) +
               3/2*force +
               45/4*pos*pos)))
added
adding force >= -1
added
adding Not(181/400 <= pos + 2*vel + 3/2000*force + 9/800*pos*pos)
added
adding Not(306 <=
    2000/3*pos +
    2000*vel +
    15/2*(pos + vel)*(pos + vel) +
    2*force +
    15*pos*pos)
added
adding Not(313/2 <=
    1000/3*pos +
    4000/3*vel +
    15/4*
    (-1/400 + pos + 2*vel + 3/2000*force + 9/800*pos*pos)*
    (-1/400 + pos + 2*vel + 3/2000*force + 9/800*pos*pos) +
    15/2*(pos + vel)*(pos + vel) +
    3/2*force +
    45/4*pos*pos)
added
noModels check of ante1 /\ ante2: sat
added hyp
SN noModels:
-- []
-- [Not(9/20 <= pos), Not(9/20 <= pos + vel), Exists(force,
       And(force >= -1,
           force <= 1,
           Not(181/400 <=
               pos + 2*vel + 3/2000*force + 9/800*pos*pos),
           Not(306 <=
               2000/3*pos +
               2000*vel +
               15/2*(pos + vel)*(pos + vel) +
               2*force +
               15*pos*pos),
           Not(313/2 <=
               1000/3*pos +
               4000/3*vel +
               15/4*
               (-1/400 +
                pos +
                2*vel +
                3/2000*force +
                9/800*pos*pos)*
               (-1/400 +
                pos +
                2*vel +
                3/2000*force +
                9/800*pos*pos) +
               15/2*(pos + vel)*(pos + vel) +
               3/2*force +
               45/4*pos*pos))), force >= -1, force <= 1, Not(306 <=
    2000/3*pos +
    2000*vel +
    15/2*(pos + vel)*(pos + vel) +
    2*force +
    15*pos*pos), Not(313/2 <=
    1000/3*pos +
    4000/3*vel +
    15/4*
    (-1/400 + pos + 2*vel + 3/2000*force + 9/800*pos*pos)*
    (-1/400 + pos + 2*vel + 3/2000*force + 9/800*pos*pos) +
    15/2*(pos + vel)*(pos + vel) +
    3/2*force +
    45/4*pos*pos)]
-- Not(Not(181/400 <=
        pos + 2*vel + 3/2000*force + 9/800*pos*pos))
noModels chcked ante1
adding ante2 to solver
adding Not(9/20 <= pos)
added
adding Not(9/20 <= pos + vel)
added
adding Exists(force,
       And(force >= -1,
           force <= 1,
           Not(181/400 <=
               pos + 2*vel + 3/2000*force + 9/800*pos*pos),
           Not(306 <=
               2000/3*pos +
               2000*vel +
               15/2*(pos + vel)*(pos + vel) +
               2*force +
               15*pos*pos),
           Not(313/2 <=
               1000/3*pos +
               4000/3*vel +
               15/4*
               (-1/400 +
                pos +
                2*vel +
                3/2000*force +
                9/800*pos*pos)*
               (-1/400 +
                pos +
                2*vel +
                3/2000*force +
                9/800*pos*pos) +
               15/2*(pos + vel)*(pos + vel) +
               3/2*force +
               45/4*pos*pos)))
added
adding force >= -1
added
adding force <= 1
added
adding Not(306 <=
    2000/3*pos +
    2000*vel +
    15/2*(pos + vel)*(pos + vel) +
    2*force +
    15*pos*pos)
added
adding Not(313/2 <=
    1000/3*pos +
    4000/3*vel +
    15/4*
    (-1/400 + pos + 2*vel + 3/2000*force + 9/800*pos*pos)*
    (-1/400 + pos + 2*vel + 3/2000*force + 9/800*pos*pos) +
    15/2*(pos + vel)*(pos + vel) +
    3/2*force +
    45/4*pos*pos)
added
noModels check of ante1 /\ ante2: sat
added hyp
SN context: [Not(9/20 <= pos), Not(9/20 <= pos + vel), Exists(force,
       And(force >= -1,
           force <= 1,
           Not(181/400 <=
               pos + 2*vel + 3/2000*force + 9/800*pos*pos),
           Not(306 <=
               2000/3*pos +
               2000*vel +
               15/2*(pos + vel)*(pos + vel) +
               2*force +
               15*pos*pos),
           Not(313/2 <=
               1000/3*pos +
               4000/3*vel +
               15/4*
               (-1/400 +
                pos +
                2*vel +
                3/2000*force +
                9/800*pos*pos)*
               (-1/400 +
                pos +
                2*vel +
                3/2000*force +
                9/800*pos*pos) +
               15/2*(pos + vel)*(pos + vel) +
               3/2*force +
               45/4*pos*pos))), force >= -1, force <= 1, Not(306 <=
    2000/3*pos +
    2000*vel +
    15/2*(pos + vel)*(pos + vel) +
    2*force +
    15*pos*pos), Not(313/2 <=
    1000/3*pos +
    4000/3*vel +
    15/4*
    (-1/400 + pos + 2*vel + 3/2000*force + 9/800*pos*pos)*
    (-1/400 + pos + 2*vel + 3/2000*force + 9/800*pos*pos) +
    15/2*(pos + vel)*(pos + vel) +
    3/2*force +
    45/4*pos*pos)]
SN negand: 181/400 <= pos + 2*vel + 3/2000*force + 9/800*pos*pos
SN noModels:
-- []
-- [Not(9/20 <= pos), Not(9/20 <= pos + vel), Exists(force,
       And(force >= -1,
           force <= 1,
           Not(181/400 <=
               pos + 2*vel + 3/2000*force + 9/800*pos*pos),
           Not(306 <=
               2000/3*pos +
               2000*vel +
               15/2*(pos + vel)*(pos + vel) +
               2*force +
               15*pos*pos),
           Not(313/2 <=
               1000/3*pos +
               4000/3*vel +
               15/4*
               (-1/400 +
                pos +
                2*vel +
                3/2000*force +
                9/800*pos*pos)*
               (-1/400 +
                pos +
                2*vel +
                3/2000*force +
                9/800*pos*pos) +
               15/2*(pos + vel)*(pos + vel) +
               3/2*force +
               45/4*pos*pos))), force >= -1, force <= 1, Not(306 <=
    2000/3*pos +
    2000*vel +
    15/2*(pos + vel)*(pos + vel) +
    2*force +
    15*pos*pos), Not(313/2 <=
    1000/3*pos +
    4000/3*vel +
    15/4*
    (-1/400 + pos + 2*vel + 3/2000*force + 9/800*pos*pos)*
    (-1/400 + pos + 2*vel + 3/2000*force + 9/800*pos*pos) +
    15/2*(pos + vel)*(pos + vel) +
    3/2*force +
    45/4*pos*pos)]
-- Not(181/400 <= pos + 2*vel + 3/2000*force + 9/800*pos*pos)
noModels chcked ante1
adding ante2 to solver
adding Not(9/20 <= pos)
added
adding Not(9/20 <= pos + vel)
added
adding Exists(force,
       And(force >= -1,
           force <= 1,
           Not(181/400 <=
               pos + 2*vel + 3/2000*force + 9/800*pos*pos),
           Not(306 <=
               2000/3*pos +
               2000*vel +
               15/2*(pos + vel)*(pos + vel) +
               2*force +
               15*pos*pos),
           Not(313/2 <=
               1000/3*pos +
               4000/3*vel +
               15/4*
               (-1/400 +
                pos +
                2*vel +
                3/2000*force +
                9/800*pos*pos)*
               (-1/400 +
                pos +
                2*vel +
                3/2000*force +
                9/800*pos*pos) +
               15/2*(pos + vel)*(pos + vel) +
               3/2*force +
               45/4*pos*pos)))
added
adding force >= -1
added
adding force <= 1
added
adding Not(306 <=
    2000/3*pos +
    2000*vel +
    15/2*(pos + vel)*(pos + vel) +
    2*force +
    15*pos*pos)
added
adding Not(313/2 <=
    1000/3*pos +
    4000/3*vel +
    15/4*
    (-1/400 + pos + 2*vel + 3/2000*force + 9/800*pos*pos)*
    (-1/400 + pos + 2*vel + 3/2000*force + 9/800*pos*pos) +
    15/2*(pos + vel)*(pos + vel) +
    3/2*force +
    45/4*pos*pos)
added
noModels check of ante1 /\ ante2: sat
added hyp
SN noModels:
-- []
-- [Not(9/20 <= pos), Not(9/20 <= pos + vel), Exists(force,
       And(force >= -1,
           force <= 1,
           Not(181/400 <=
               pos + 2*vel + 3/2000*force + 9/800*pos*pos),
           Not(306 <=
               2000/3*pos +
               2000*vel +
               15/2*(pos + vel)*(pos + vel) +
               2*force +
               15*pos*pos),
           Not(313/2 <=
               1000/3*pos +
               4000/3*vel +
               15/4*
               (-1/400 +
                pos +
                2*vel +
                3/2000*force +
                9/800*pos*pos)*
               (-1/400 +
                pos +
                2*vel +
                3/2000*force +
                9/800*pos*pos) +
               15/2*(pos + vel)*(pos + vel) +
               3/2*force +
               45/4*pos*pos))), force >= -1, force <= 1, Not(181/400 <= pos + 2*vel + 3/2000*force + 9/800*pos*pos), Not(313/2 <=
    1000/3*pos +
    4000/3*vel +
    15/4*
    (-1/400 + pos + 2*vel + 3/2000*force + 9/800*pos*pos)*
    (-1/400 + pos + 2*vel + 3/2000*force + 9/800*pos*pos) +
    15/2*(pos + vel)*(pos + vel) +
    3/2*force +
    45/4*pos*pos)]
-- Not(Not(306 <=
        2000/3*pos +
        2000*vel +
        15/2*(pos + vel)*(pos + vel) +
        2*force +
        15*pos*pos))
noModels chcked ante1
adding ante2 to solver
adding Not(9/20 <= pos)
added
adding Not(9/20 <= pos + vel)
added
adding Exists(force,
       And(force >= -1,
           force <= 1,
           Not(181/400 <=
               pos + 2*vel + 3/2000*force + 9/800*pos*pos),
           Not(306 <=
               2000/3*pos +
               2000*vel +
               15/2*(pos + vel)*(pos + vel) +
               2*force +
               15*pos*pos),
           Not(313/2 <=
               1000/3*pos +
               4000/3*vel +
               15/4*
               (-1/400 +
                pos +
                2*vel +
                3/2000*force +
                9/800*pos*pos)*
               (-1/400 +
                pos +
                2*vel +
                3/2000*force +
                9/800*pos*pos) +
               15/2*(pos + vel)*(pos + vel) +
               3/2*force +
               45/4*pos*pos)))
added
adding force >= -1
added
adding force <= 1
added
adding Not(181/400 <= pos + 2*vel + 3/2000*force + 9/800*pos*pos)
added
adding Not(313/2 <=
    1000/3*pos +
    4000/3*vel +
    15/4*
    (-1/400 + pos + 2*vel + 3/2000*force + 9/800*pos*pos)*
    (-1/400 + pos + 2*vel + 3/2000*force + 9/800*pos*pos) +
    15/2*(pos + vel)*(pos + vel) +
    3/2*force +
    45/4*pos*pos)
added
noModels check of ante1 /\ ante2: sat
added hyp
SN context: [Not(9/20 <= pos), Not(9/20 <= pos + vel), Exists(force,
       And(force >= -1,
           force <= 1,
           Not(181/400 <=
               pos + 2*vel + 3/2000*force + 9/800*pos*pos),
           Not(306 <=
               2000/3*pos +
               2000*vel +
               15/2*(pos + vel)*(pos + vel) +
               2*force +
               15*pos*pos),
           Not(313/2 <=
               1000/3*pos +
               4000/3*vel +
               15/4*
               (-1/400 +
                pos +
                2*vel +
                3/2000*force +
                9/800*pos*pos)*
               (-1/400 +
                pos +
                2*vel +
                3/2000*force +
                9/800*pos*pos) +
               15/2*(pos + vel)*(pos + vel) +
               3/2*force +
               45/4*pos*pos))), force >= -1, force <= 1, Not(181/400 <= pos + 2*vel + 3/2000*force + 9/800*pos*pos), Not(313/2 <=
    1000/3*pos +
    4000/3*vel +
    15/4*
    (-1/400 + pos + 2*vel + 3/2000*force + 9/800*pos*pos)*
    (-1/400 + pos + 2*vel + 3/2000*force + 9/800*pos*pos) +
    15/2*(pos + vel)*(pos + vel) +
    3/2*force +
    45/4*pos*pos)]
SN negand: 306 <=
2000/3*pos +
2000*vel +
15/2*(pos + vel)*(pos + vel) +
2*force +
15*pos*pos
SN noModels:
-- []
-- [Not(9/20 <= pos), Not(9/20 <= pos + vel), Exists(force,
       And(force >= -1,
           force <= 1,
           Not(181/400 <=
               pos + 2*vel + 3/2000*force + 9/800*pos*pos),
           Not(306 <=
               2000/3*pos +
               2000*vel +
               15/2*(pos + vel)*(pos + vel) +
               2*force +
               15*pos*pos),
           Not(313/2 <=
               1000/3*pos +
               4000/3*vel +
               15/4*
               (-1/400 +
                pos +
                2*vel +
                3/2000*force +
                9/800*pos*pos)*
               (-1/400 +
                pos +
                2*vel +
                3/2000*force +
                9/800*pos*pos) +
               15/2*(pos + vel)*(pos + vel) +
               3/2*force +
               45/4*pos*pos))), force >= -1, force <= 1, Not(181/400 <= pos + 2*vel + 3/2000*force + 9/800*pos*pos), Not(313/2 <=
    1000/3*pos +
    4000/3*vel +
    15/4*
    (-1/400 + pos + 2*vel + 3/2000*force + 9/800*pos*pos)*
    (-1/400 + pos + 2*vel + 3/2000*force + 9/800*pos*pos) +
    15/2*(pos + vel)*(pos + vel) +
    3/2*force +
    45/4*pos*pos)]
-- Not(306 <=
    2000/3*pos +
    2000*vel +
    15/2*(pos + vel)*(pos + vel) +
    2*force +
    15*pos*pos)
noModels chcked ante1
adding ante2 to solver
adding Not(9/20 <= pos)
added
adding Not(9/20 <= pos + vel)
added
adding Exists(force,
       And(force >= -1,
           force <= 1,
           Not(181/400 <=
               pos + 2*vel + 3/2000*force + 9/800*pos*pos),
           Not(306 <=
               2000/3*pos +
               2000*vel +
               15/2*(pos + vel)*(pos + vel) +
               2*force +
               15*pos*pos),
           Not(313/2 <=
               1000/3*pos +
               4000/3*vel +
               15/4*
               (-1/400 +
                pos +
                2*vel +
                3/2000*force +
                9/800*pos*pos)*
               (-1/400 +
                pos +
                2*vel +
                3/2000*force +
                9/800*pos*pos) +
               15/2*(pos + vel)*(pos + vel) +
               3/2*force +
               45/4*pos*pos)))
added
adding force >= -1
added
adding force <= 1
added
adding Not(181/400 <= pos + 2*vel + 3/2000*force + 9/800*pos*pos)
added
adding Not(313/2 <=
    1000/3*pos +
    4000/3*vel +
    15/4*
    (-1/400 + pos + 2*vel + 3/2000*force + 9/800*pos*pos)*
    (-1/400 + pos + 2*vel + 3/2000*force + 9/800*pos*pos) +
    15/2*(pos + vel)*(pos + vel) +
    3/2*force +
    45/4*pos*pos)
added
noModels check of ante1 /\ ante2: sat
added hyp
SN noModels:
-- []
-- [Not(9/20 <= pos), Not(9/20 <= pos + vel), Exists(force,
       And(force >= -1,
           force <= 1,
           Not(181/400 <=
               pos + 2*vel + 3/2000*force + 9/800*pos*pos),
           Not(306 <=
               2000/3*pos +
               2000*vel +
               15/2*(pos + vel)*(pos + vel) +
               2*force +
               15*pos*pos),
           Not(313/2 <=
               1000/3*pos +
               4000/3*vel +
               15/4*
               (-1/400 +
                pos +
                2*vel +
                3/2000*force +
                9/800*pos*pos)*
               (-1/400 +
                pos +
                2*vel +
                3/2000*force +
                9/800*pos*pos) +
               15/2*(pos + vel)*(pos + vel) +
               3/2*force +
               45/4*pos*pos))), force >= -1, force <= 1, Not(181/400 <= pos + 2*vel + 3/2000*force + 9/800*pos*pos), Not(306 <=
    2000/3*pos +
    2000*vel +
    15/2*(pos + vel)*(pos + vel) +
    2*force +
    15*pos*pos)]
-- Not(Not(313/2 <=
        1000/3*pos +
        4000/3*vel +
        15/4*
        (-1/400 + pos + 2*vel + 3/2000*force + 9/800*pos*pos)*
        (-1/400 + pos + 2*vel + 3/2000*force + 9/800*pos*pos) +
        15/2*(pos + vel)*(pos + vel) +
        3/2*force +
        45/4*pos*pos))
noModels chcked ante1
adding ante2 to solver
adding Not(9/20 <= pos)
added
adding Not(9/20 <= pos + vel)
added
adding Exists(force,
       And(force >= -1,
           force <= 1,
           Not(181/400 <=
               pos + 2*vel + 3/2000*force + 9/800*pos*pos),
           Not(306 <=
               2000/3*pos +
               2000*vel +
               15/2*(pos + vel)*(pos + vel) +
               2*force +
               15*pos*pos),
           Not(313/2 <=
               1000/3*pos +
               4000/3*vel +
               15/4*
               (-1/400 +
                pos +
                2*vel +
                3/2000*force +
                9/800*pos*pos)*
               (-1/400 +
                pos +
                2*vel +
                3/2000*force +
                9/800*pos*pos) +
               15/2*(pos + vel)*(pos + vel) +
               3/2*force +
               45/4*pos*pos)))
added
adding force >= -1
added
adding force <= 1
added
adding Not(181/400 <= pos + 2*vel + 3/2000*force + 9/800*pos*pos)
added
adding Not(306 <=
    2000/3*pos +
    2000*vel +
    15/2*(pos + vel)*(pos + vel) +
    2*force +
    15*pos*pos)
added
noModels check of ante1 /\ ante2: sat
added hyp
SN context: [Not(9/20 <= pos), Not(9/20 <= pos + vel), Exists(force,
       And(force >= -1,
           force <= 1,
           Not(181/400 <=
               pos + 2*vel + 3/2000*force + 9/800*pos*pos),
           Not(306 <=
               2000/3*pos +
               2000*vel +
               15/2*(pos + vel)*(pos + vel) +
               2*force +
               15*pos*pos),
           Not(313/2 <=
               1000/3*pos +
               4000/3*vel +
               15/4*
               (-1/400 +
                pos +
                2*vel +
                3/2000*force +
                9/800*pos*pos)*
               (-1/400 +
                pos +
                2*vel +
                3/2000*force +
                9/800*pos*pos) +
               15/2*(pos + vel)*(pos + vel) +
               3/2*force +
               45/4*pos*pos))), force >= -1, force <= 1, Not(181/400 <= pos + 2*vel + 3/2000*force + 9/800*pos*pos), Not(306 <=
    2000/3*pos +
    2000*vel +
    15/2*(pos + vel)*(pos + vel) +
    2*force +
    15*pos*pos)]
SN negand: 313/2 <=
1000/3*pos +
4000/3*vel +
15/4*
(-1/400 + pos + 2*vel + 3/2000*force + 9/800*pos*pos)*
(-1/400 + pos + 2*vel + 3/2000*force + 9/800*pos*pos) +
15/2*(pos + vel)*(pos + vel) +
3/2*force +
45/4*pos*pos
SN noModels:
-- []
-- [Not(9/20 <= pos), Not(9/20 <= pos + vel), Exists(force,
       And(force >= -1,
           force <= 1,
           Not(181/400 <=
               pos + 2*vel + 3/2000*force + 9/800*pos*pos),
           Not(306 <=
               2000/3*pos +
               2000*vel +
               15/2*(pos + vel)*(pos + vel) +
               2*force +
               15*pos*pos),
           Not(313/2 <=
               1000/3*pos +
               4000/3*vel +
               15/4*
               (-1/400 +
                pos +
                2*vel +
                3/2000*force +
                9/800*pos*pos)*
               (-1/400 +
                pos +
                2*vel +
                3/2000*force +
                9/800*pos*pos) +
               15/2*(pos + vel)*(pos + vel) +
               3/2*force +
               45/4*pos*pos))), force >= -1, force <= 1, Not(181/400 <= pos + 2*vel + 3/2000*force + 9/800*pos*pos), Not(306 <=
    2000/3*pos +
    2000*vel +
    15/2*(pos + vel)*(pos + vel) +
    2*force +
    15*pos*pos)]
-- Not(313/2 <=
    1000/3*pos +
    4000/3*vel +
    15/4*
    (-1/400 + pos + 2*vel + 3/2000*force + 9/800*pos*pos)*
    (-1/400 + pos + 2*vel + 3/2000*force + 9/800*pos*pos) +
    15/2*(pos + vel)*(pos + vel) +
    3/2*force +
    45/4*pos*pos)
noModels chcked ante1
adding ante2 to solver
adding Not(9/20 <= pos)
added
adding Not(9/20 <= pos + vel)
added
adding Exists(force,
       And(force >= -1,
           force <= 1,
           Not(181/400 <=
               pos + 2*vel + 3/2000*force + 9/800*pos*pos),
           Not(306 <=
               2000/3*pos +
               2000*vel +
               15/2*(pos + vel)*(pos + vel) +
               2*force +
               15*pos*pos),
           Not(313/2 <=
               1000/3*pos +
               4000/3*vel +
               15/4*
               (-1/400 +
                pos +
                2*vel +
                3/2000*force +
                9/800*pos*pos)*
               (-1/400 +
                pos +
                2*vel +
                3/2000*force +
                9/800*pos*pos) +
               15/2*(pos + vel)*(pos + vel) +
               3/2*force +
               45/4*pos*pos)))
added
adding force >= -1
added
adding force <= 1
added
adding Not(181/400 <= pos + 2*vel + 3/2000*force + 9/800*pos*pos)
added
adding Not(306 <=
    2000/3*pos +
    2000*vel +
    15/2*(pos + vel)*(pos + vel) +
    2*force +
    15*pos*pos)
added
noModels check of ante1 /\ ante2: sat
added hyp
new ctrl Pred (diff b/w updated state inv and current ctrl pred):
 And(force >= -1,
    force <= 1,
    Not(181/400 <=
        pos + 2*vel + 3/2000*force + 9/800*pos*pos),
    Not(306 <=
        2000/3*pos +
        2000*vel +
        15/2*(pos + vel)*(pos + vel) +
        2*force +
        15*pos*pos),
    Not(313/2 <=
        1000/3*pos +
        4000/3*vel +
        15/4*
        (-1/400 + pos + 2*vel + 3/2000*force + 9/800*pos*pos)*
        (-1/400 + pos + 2*vel + 3/2000*force + 9/800*pos*pos) +
        15/2*(pos + vel)*(pos + vel) +
        3/2*force +
        45/4*pos*pos))

----------------
iteration 4

-- Refining the state invariant for node singleton
--- arc: action
wcp =
 ForAll([posX, velX],
       Implies(And(velX ==
                   vel + force*3/2000 -
                   1/400*(1 - (3*pos*3*pos)/2),
                   posX == pos + vel),
               And(Not(9/20 <= posX),
                   Not(9/20 <= posX + velX),
                   Exists(force,
                          And(force >= -1,
                              force <= 1,
                              Not(181/400 <=
                                  posX +
                                  2*velX +
                                  3/2000*force +
                                  9/800*posX*posX),
                              Not(306 <=
                                  2000/3*posX +
                                  2000*velX +
                                  15/2*
                                  (posX + velX)*
                                  (posX + velX) +
                                  2*force +
                                  15*posX*posX),
                              Not(313/2 <=
                                  1000/3*posX +
                                  4000/3*velX +
                                  15/4*
                                  (-1/400 +
                                   posX +
                                   2*velX +
                                   3/2000*force +
                                   9/800*posX*posX)*
                                  (-1/400 +
                                   posX +
                                   2*velX +
                                   3/2000*force +
                                   9/800*posX*posX) +
                                  15/2*
                                  (posX + velX)*
                                  (posX + velX) +
                                  3/2*force +
                                  45/4*posX*posX))))))
wcp_simp =
 And(Not(9/20 <= pos + vel),
    Not(181/400 <=
        pos + 2*vel + 3/2000*force + 9/800*pos*pos),
    Exists(force,
           And(force >= -1,
               force <= 1,
               Not(183/400 <=
                   pos +
                   3*vel +
                   3/1000*force +
                   9/400*pos*pos +
                   3/2000*force +
                   9/800*(pos + vel)*(pos + vel)),
               Not(311 <=
                   2000/3*pos +
                   8000/3*vel +
                   3*force +
                   45/2*pos*pos +
                   15/2*
                   (-1/400 +
                    pos +
                    2*vel +
                    3/2000*force +
                    9/800*pos*pos)*
                   (-1/400 +
                    pos +
                    2*vel +
                    3/2000*force +
                    9/800*pos*pos) +
                   2*force +
                   15*(pos + vel)*(pos + vel)),
               Not(959/6 <=
                   1000/3*pos +
                   5000/3*vel +
                   2*force +
                   15*pos*pos +
                   15/4*
                   (-3/400 +
                    pos +
                    3*vel +
                    3/1000*force +
                    9/400*pos*pos +
                    3/2000*force +
                    9/800*(pos + vel)*(pos + vel))*
                   (-3/400 +
                    pos +
                    3*vel +
                    3/1000*force +
                    9/400*pos*pos +
                    3/2000*force +
                    9/800*(pos + vel)*(pos + vel)) +
                   15/2*
                   (-1/400 +
                    pos +
                    2*vel +
                    3/2000*force +
                    9/800*pos*pos)*
                   (-1/400 +
                    pos +
                    2*vel +
                    3/2000*force +
                    9/800*pos*pos) +
                   3/2*force +
                   45/4*(pos + vel)*(pos + vel)))))
wcp_simp after QE (wcp_qe) =
And(Not(9/20 <= pos + vel),
    Not(181/400 <=
        pos + 2*vel + 3/2000*force + 9/800*pos*pos),
    Exists(force,
           And(force >= -1,
               force <= 1,
               Not(183/400 <=
                   pos +
                   3*vel +
                   3/1000*force +
                   9/400*pos*pos +
                   3/2000*force +
                   9/800*(pos + vel)*(pos + vel)),
               Not(311 <=
                   2000/3*pos +
                   8000/3*vel +
                   3*force +
                   45/2*pos*pos +
                   15/2*
                   (-1/400 +
                    pos +
                    2*vel +
                    3/2000*force +
                    9/800*pos*pos)*
                   (-1/400 +
                    pos +
                    2*vel +
                    3/2000*force +
                    9/800*pos*pos) +
                   2*force +
                   15*(pos + vel)*(pos + vel)),
               Not(959/6 <=
                   1000/3*pos +
                   5000/3*vel +
                   2*force +
                   15*pos*pos +
                   15/4*
                   (-3/400 +
                    pos +
                    3*vel +
                    3/1000*force +
                    9/400*pos*pos +
                    3/2000*force +
                    9/800*(pos + vel)*(pos + vel))*
                   (-3/400 +
                    pos +
                    3*vel +
                    3/1000*force +
                    9/400*pos*pos +
                    3/2000*force +
                    9/800*(pos + vel)*(pos + vel)) +
                   15/2*
                   (-1/400 +
                    pos +
                    2*vel +
                    3/2000*force +
                    9/800*pos*pos)*
                   (-1/400 +
                    pos +
                    2*vel +
                    3/2000*force +
                    9/800*pos*pos) +
                   3/2*force +
                   45/4*(pos + vel)*(pos + vel)))))
strengthened ctrl pred
And(force >= -1,
    force <= 1,
    Not(181/400 <=
        pos + 2*vel + 3/2000*force + 9/800*pos*pos),
    Not(306 <=
        2000/3*pos +
        2000*vel +
        15/2*(pos + vel)*(pos + vel) +
        2*force +
        15*pos*pos),
    Not(313/2 <=
        1000/3*pos +
        4000/3*vel +
        15/4*
        (-1/400 + pos + 2*vel + 3/2000*force + 9/800*pos*pos)*
        (-1/400 + pos + 2*vel + 3/2000*force + 9/800*pos*pos) +
        15/2*(pos + vel)*(pos + vel) +
        3/2*force +
        45/4*pos*pos),
    Not(9/20 <= pos + vel),
    Exists(force,
           And(force >= -1,
               force <= 1,
               Not(183/400 <=
                   pos +
                   3*vel +
                   3/1000*force +
                   9/400*pos*pos +
                   3/2000*force +
                   9/800*(pos + vel)*(pos + vel)),
               Not(311 <=
                   2000/3*pos +
                   8000/3*vel +
                   3*force +
                   45/2*pos*pos +
                   15/2*
                   (-1/400 +
                    pos +
                    2*vel +
                    3/2000*force +
                    9/800*pos*pos)*
                   (-1/400 +
                    pos +
                    2*vel +
                    3/2000*force +
                    9/800*pos*pos) +
                   2*force +
                   15*(pos + vel)*(pos + vel)),
               Not(959/6 <=
                   1000/3*pos +
                   5000/3*vel +
                   2*force +
                   15*pos*pos +
                   15/4*
                   (-3/400 +
                    pos +
                    3*vel +
                    3/1000*force +
                    9/400*pos*pos +
                    3/2000*force +
                    9/800*(pos + vel)*(pos + vel))*
                   (-3/400 +
                    pos +
                    3*vel +
                    3/1000*force +
                    9/400*pos*pos +
                    3/2000*force +
                    9/800*(pos + vel)*(pos + vel)) +
                   15/2*
                   (-1/400 +
                    pos +
                    2*vel +
                    3/2000*force +
                    9/800*pos*pos)*
                   (-1/400 +
                    pos +
                    2*vel +
                    3/2000*force +
                    9/800*pos*pos) +
                   3/2*force +
                   45/4*(pos + vel)*(pos + vel)))))
SN noModels:
-- []
-- [Not(9/20 <= pos), Not(9/20 <= pos + vel), Exists(force,
       And(force >= -1,
           force <= 1,
           Not(181/400 <=
               pos + 2*vel + 3/2000*force + 9/800*pos*pos),
           Not(306 <=
               2000/3*pos +
               2000*vel +
               15/2*(pos + vel)*(pos + vel) +
               2*force +
               15*pos*pos),
           Not(313/2 <=
               1000/3*pos +
               4000/3*vel +
               15/4*
               (-1/400 +
                pos +
                2*vel +
                3/2000*force +
                9/800*pos*pos)*
               (-1/400 +
                pos +
                2*vel +
                3/2000*force +
                9/800*pos*pos) +
               15/2*(pos + vel)*(pos + vel) +
               3/2*force +
               45/4*pos*pos)))]
-- Not(And(force >= -1,
        force <= 1,
        Not(181/400 <=
            pos + 2*vel + 3/2000*force + 9/800*pos*pos),
        Not(306 <=
            2000/3*pos +
            2000*vel +
            15/2*(pos + vel)*(pos + vel) +
            2*force +
            15*pos*pos),
        Not(313/2 <=
            1000/3*pos +
            4000/3*vel +
            15/4*
            (-1/400 +
             pos +
             2*vel +
             3/2000*force +
             9/800*pos*pos)*
            (-1/400 +
             pos +
             2*vel +
             3/2000*force +
             9/800*pos*pos) +
            15/2*(pos + vel)*(pos + vel) +
            3/2*force +
            45/4*pos*pos),
        Not(9/20 <= pos + vel),
        Exists(force,
               And(force >= -1,
                   force <= 1,
                   Not(183/400 <=
                       pos +
                       3*vel +
                       3/1000*force +
                       9/400*pos*pos +
                       3/2000*force +
                       9/800*(pos + vel)*(pos + vel)),
                   Not(311 <=
                       2000/3*pos +
                       8000/3*vel +
                       3*force +
                       45/2*pos*pos +
                       15/2*
                       (-1/400 +
                        pos +
                        2*vel +
                        3/2000*force +
                        9/800*pos*pos)*
                       (-1/400 +
                        pos +
                        2*vel +
                        3/2000*force +
                        9/800*pos*pos) +
                       2*force +
                       15*(pos + vel)*(pos + vel)),
                   Not(959/6 <=
                       1000/3*pos +
                       5000/3*vel +
                       2*force +
                       15*pos*pos +
                       15/4*
                       (-3/400 +
                        pos +
                        3*vel +
                        3/1000*force +
                        9/400*pos*pos +
                        3/2000*force +
                        9/800*(pos + vel)*(pos + vel))*
                       (-3/400 +
                        pos +
                        3*vel +
                        3/1000*force +
                        9/400*pos*pos +
                        3/2000*force +
                        9/800*(pos + vel)*(pos + vel)) +
                       15/2*
                       (-1/400 +
                        pos +
                        2*vel +
                        3/2000*force +
                        9/800*pos*pos)*
                       (-1/400 +
                        pos +
                        2*vel +
                        3/2000*force +
                        9/800*pos*pos) +
                       3/2*force +
                       45/4*(pos + vel)*(pos + vel))))))
noModels chcked ante1
adding ante2 to solver
adding Not(9/20 <= pos)
added
adding Not(9/20 <= pos + vel)
added
adding Exists(force,
       And(force >= -1,
           force <= 1,
           Not(181/400 <=
               pos + 2*vel + 3/2000*force + 9/800*pos*pos),
           Not(306 <=
               2000/3*pos +
               2000*vel +
               15/2*(pos + vel)*(pos + vel) +
               2*force +
               15*pos*pos),
           Not(313/2 <=
               1000/3*pos +
               4000/3*vel +
               15/4*
               (-1/400 +
                pos +
                2*vel +
                3/2000*force +
                9/800*pos*pos)*
               (-1/400 +
                pos +
                2*vel +
                3/2000*force +
                9/800*pos*pos) +
               15/2*(pos + vel)*(pos + vel) +
               3/2*force +
               45/4*pos*pos)))
added
noModels check of ante1 /\ ante2: sat
added hyp
SC context: [Not(9/20 <= pos), Not(9/20 <= pos + vel), Exists(force,
       And(force >= -1,
           force <= 1,
           Not(181/400 <=
               pos + 2*vel + 3/2000*force + 9/800*pos*pos),
           Not(306 <=
               2000/3*pos +
               2000*vel +
               15/2*(pos + vel)*(pos + vel) +
               2*force +
               15*pos*pos),
           Not(313/2 <=
               1000/3*pos +
               4000/3*vel +
               15/4*
               (-1/400 +
                pos +
                2*vel +
                3/2000*force +
                9/800*pos*pos)*
               (-1/400 +
                pos +
                2*vel +
                3/2000*force +
                9/800*pos*pos) +
               15/2*(pos + vel)*(pos + vel) +
               3/2*force +
               45/4*pos*pos)))]
conjuncts: [force >= -1, force <= 1, Not(181/400 <= pos + 2*vel + 3/2000*force + 9/800*pos*pos), Not(306 <=
    2000/3*pos +
    2000*vel +
    15/2*(pos + vel)*(pos + vel) +
    2*force +
    15*pos*pos), Not(313/2 <=
    1000/3*pos +
    4000/3*vel +
    15/4*
    (-1/400 + pos + 2*vel + 3/2000*force + 9/800*pos*pos)*
    (-1/400 + pos + 2*vel + 3/2000*force + 9/800*pos*pos) +
    15/2*(pos + vel)*(pos + vel) +
    3/2*force +
    45/4*pos*pos), Not(9/20 <= pos + vel), Exists(force,
       And(force >= -1,
           force <= 1,
           Not(183/400 <=
               pos +
               3*vel +
               3/1000*force +
               9/400*pos*pos +
               3/2000*force +
               9/800*(pos + vel)*(pos + vel)),
           Not(311 <=
               2000/3*pos +
               8000/3*vel +
               3*force +
               45/2*pos*pos +
               15/2*
               (-1/400 +
                pos +
                2*vel +
                3/2000*force +
                9/800*pos*pos)*
               (-1/400 +
                pos +
                2*vel +
                3/2000*force +
                9/800*pos*pos) +
               2*force +
               15*(pos + vel)*(pos + vel)),
           Not(959/6 <=
               1000/3*pos +
               5000/3*vel +
               2*force +
               15*pos*pos +
               15/4*
               (-3/400 +
                pos +
                3*vel +
                3/1000*force +
                9/400*pos*pos +
                3/2000*force +
                9/800*(pos + vel)*(pos + vel))*
               (-3/400 +
                pos +
                3*vel +
                3/1000*force +
                9/400*pos*pos +
                3/2000*force +
                9/800*(pos + vel)*(pos + vel)) +
               15/2*
               (-1/400 +
                pos +
                2*vel +
                3/2000*force +
                9/800*pos*pos)*
               (-1/400 +
                pos +
                2*vel +
                3/2000*force +
                9/800*pos*pos) +
               3/2*force +
               45/4*(pos + vel)*(pos + vel))))]
SN noModels:
-- []
-- [Not(9/20 <= pos), Not(9/20 <= pos + vel), Exists(force,
       And(force >= -1,
           force <= 1,
           Not(181/400 <=
               pos + 2*vel + 3/2000*force + 9/800*pos*pos),
           Not(306 <=
               2000/3*pos +
               2000*vel +
               15/2*(pos + vel)*(pos + vel) +
               2*force +
               15*pos*pos),
           Not(313/2 <=
               1000/3*pos +
               4000/3*vel +
               15/4*
               (-1/400 +
                pos +
                2*vel +
                3/2000*force +
                9/800*pos*pos)*
               (-1/400 +
                pos +
                2*vel +
                3/2000*force +
                9/800*pos*pos) +
               15/2*(pos + vel)*(pos + vel) +
               3/2*force +
               45/4*pos*pos))), force <= 1, Not(181/400 <= pos + 2*vel + 3/2000*force + 9/800*pos*pos), Not(306 <=
    2000/3*pos +
    2000*vel +
    15/2*(pos + vel)*(pos + vel) +
    2*force +
    15*pos*pos), Not(313/2 <=
    1000/3*pos +
    4000/3*vel +
    15/4*
    (-1/400 + pos + 2*vel + 3/2000*force + 9/800*pos*pos)*
    (-1/400 + pos + 2*vel + 3/2000*force + 9/800*pos*pos) +
    15/2*(pos + vel)*(pos + vel) +
    3/2*force +
    45/4*pos*pos), Exists(force,
       And(force >= -1,
           force <= 1,
           Not(183/400 <=
               pos +
               3*vel +
               3/1000*force +
               9/400*pos*pos +
               3/2000*force +
               9/800*(pos + vel)*(pos + vel)),
           Not(311 <=
               2000/3*pos +
               8000/3*vel +
               3*force +
               45/2*pos*pos +
               15/2*
               (-1/400 +
                pos +
                2*vel +
                3/2000*force +
                9/800*pos*pos)*
               (-1/400 +
                pos +
                2*vel +
                3/2000*force +
                9/800*pos*pos) +
               2*force +
               15*(pos + vel)*(pos + vel)),
           Not(959/6 <=
               1000/3*pos +
               5000/3*vel +
               2*force +
               15*pos*pos +
               15/4*
               (-3/400 +
                pos +
                3*vel +
                3/1000*force +
                9/400*pos*pos +
                3/2000*force +
                9/800*(pos + vel)*(pos + vel))*
               (-3/400 +
                pos +
                3*vel +
                3/1000*force +
                9/400*pos*pos +
                3/2000*force +
                9/800*(pos + vel)*(pos + vel)) +
               15/2*
               (-1/400 +
                pos +
                2*vel +
                3/2000*force +
                9/800*pos*pos)*
               (-1/400 +
                pos +
                2*vel +
                3/2000*force +
                9/800*pos*pos) +
               3/2*force +
               45/4*(pos + vel)*(pos + vel))))]
-- Not(force >= -1)
noModels chcked ante1
adding ante2 to solver
adding Not(9/20 <= pos)
added
adding Not(9/20 <= pos + vel)
added
adding Exists(force,
       And(force >= -1,
           force <= 1,
           Not(181/400 <=
               pos + 2*vel + 3/2000*force + 9/800*pos*pos),
           Not(306 <=
               2000/3*pos +
               2000*vel +
               15/2*(pos + vel)*(pos + vel) +
               2*force +
               15*pos*pos),
           Not(313/2 <=
               1000/3*pos +
               4000/3*vel +
               15/4*
               (-1/400 +
                pos +
                2*vel +
                3/2000*force +
                9/800*pos*pos)*
               (-1/400 +
                pos +
                2*vel +
                3/2000*force +
                9/800*pos*pos) +
               15/2*(pos + vel)*(pos + vel) +
               3/2*force +
               45/4*pos*pos)))
added
adding force <= 1
added
adding Not(181/400 <= pos + 2*vel + 3/2000*force + 9/800*pos*pos)
added
adding Not(306 <=
    2000/3*pos +
    2000*vel +
    15/2*(pos + vel)*(pos + vel) +
    2*force +
    15*pos*pos)
added
adding Not(313/2 <=
    1000/3*pos +
    4000/3*vel +
    15/4*
    (-1/400 + pos + 2*vel + 3/2000*force + 9/800*pos*pos)*
    (-1/400 + pos + 2*vel + 3/2000*force + 9/800*pos*pos) +
    15/2*(pos + vel)*(pos + vel) +
    3/2*force +
    45/4*pos*pos)
added
adding Exists(force,
       And(force >= -1,
           force <= 1,
           Not(183/400 <=
               pos +
               3*vel +
               3/1000*force +
               9/400*pos*pos +
               3/2000*force +
               9/800*(pos + vel)*(pos + vel)),
           Not(311 <=
               2000/3*pos +
               8000/3*vel +
               3*force +
               45/2*pos*pos +
               15/2*
               (-1/400 +
                pos +
                2*vel +
                3/2000*force +
                9/800*pos*pos)*
               (-1/400 +
                pos +
                2*vel +
                3/2000*force +
                9/800*pos*pos) +
               2*force +
               15*(pos + vel)*(pos + vel)),
           Not(959/6 <=
               1000/3*pos +
               5000/3*vel +
               2*force +
               15*pos*pos +
               15/4*
               (-3/400 +
                pos +
                3*vel +
                3/1000*force +
                9/400*pos*pos +
                3/2000*force +
                9/800*(pos + vel)*(pos + vel))*
               (-3/400 +
                pos +
                3*vel +
                3/1000*force +
                9/400*pos*pos +
                3/2000*force +
                9/800*(pos + vel)*(pos + vel)) +
               15/2*
               (-1/400 +
                pos +
                2*vel +
                3/2000*force +
                9/800*pos*pos)*
               (-1/400 +
                pos +
                2*vel +
                3/2000*force +
                9/800*pos*pos) +
               3/2*force +
               45/4*(pos + vel)*(pos + vel))))
added
