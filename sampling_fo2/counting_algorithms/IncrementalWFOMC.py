from typing import Callable
from functools import reduce

from sampling_fo2.cell_graph import CellGraph
from sampling_fo2.utils import RingElement, Rational
from sampling_fo2.utils.polynomial import expand
from sampling_fo2.fol.syntax import Const, Pred, QFFormula

def incremental_wfomc(formula: QFFormula,
                      domain: set[Const],
                      get_weight: Callable[[Pred], tuple[RingElement, RingElement]],
                      leq_pred: Pred = None) -> RingElement:
    cell_graph = CellGraph(formula, get_weight, leq_pred)
    # cell_graph.show()
    cells = cell_graph.get_cells()
    n_cells = len(cells)
    domain_size = len(domain)
    
    cell_weights = cell_graph.get_all_weights()[0]
    edge_weights = cell_graph.get_all_weights()[1]

    table = dict(
        (tuple(int(k == i) for k in range(n_cells)),
         cell_weights[i])
         for i in range(n_cells)
    )
    for _ in range(domain_size - 1):
        old_table = table
        table = dict()
        for j in range(n_cells):
            w = cell_weights[j]
            for ivec, w_old in old_table.items():
                w_new = w_old * w * reduce(
                    lambda x, y: x * y,
                    (edge_weights[j][l] ** int(ivec[l]) for l in range(n_cells)),
                    Rational(1, 1)
                )
                w_new = expand(w_new)
                
                ivec = list(ivec)
                ivec[j] += 1
                ivec = tuple(ivec)

                w_new = w_new + table.get(ivec, Rational(0, 1))
                table[tuple(ivec)] = w_new
    ret = sum(table.values())
    return ret