from sampling_fo2.parser.fol_parser import parse
from sampling_fo2.fol.syntax import *
from sampling_fo2.network.constraint import CardinalityConstraint
from sampling_fo2.problems import WFOMCSProblem
from sampling_fo2.wfomc import Algo, wfomc

from symengine import expand, var


sentence = parse(
    '\\forall X: (S(X) <-> (A(X) | N(X))) & \\forall X: (\\exists Y: (S(X) -> F(X, Y))) & \\forall X: (\\forall Y: (F(X, Y) -> (Ind(Y) & S(X))))'
    '& \\forall Y: (\\exists X: (Ind(Y) -> F(X, Y)))'
    '& \\forall X: (\\forall Y: ((A(X) & F(X, Y)) <-> Fa(X, Y)))'
    '& \\forall X: (\\forall Y: ((N(X) & F(X, Y)) <-> Fn(X, Y)))'
)
F = Pred('F', 2)
Fa = Pred('Fa', 2)
Fn = Pred('Fn', 2)
S = Pred('S', 1)
A = Pred('A', 1)
B = Pred('B', 1)
N = Pred('N', 1)
D = Pred('D', 1)
Ind = Pred('Ind', 1)
Ia = Pred('Ia', 1)
Id = Pred('Id', 1)
Ib = Pred('Ib', 1)
In = Pred('In', 1)

i1 = Const('i1')
i2 = Const('i2')
i3 = Const('i3')
i4 = Const('i4')

domain = [i1, i2, i3, i4]

cc = CardinalityConstraint()

xf = var('xf')
xa = var('xa')
xb = var('xb')
xn = var('xn')
xd = var('xd')
weights = {
    Fa: (xa + xa ** 2 + xa ** 3, 1),
    Fn: (xn + xn ** 2, 1),
    # F: (1, 1),
    # S: (1, 1),
    # A: (1, 1),
    # B: (1, 1),
    # Ind: (1, 1)
}

p = WFOMCSProblem(sentence, domain, weights, cc, evidence)
res = wfomc(p, use_partition_constraint=True)
print(expand(res))