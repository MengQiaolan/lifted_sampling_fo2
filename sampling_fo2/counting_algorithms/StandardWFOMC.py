from __future__ import annotations
from logzero import logger
from typing import Callable

from sampling_fo2.cell_graph.cell_graph import CellGraph, Cell
from sampling_fo2.utils import MultinomialCoefficients, multinomial, RingElement, Rational
from sampling_fo2.fol.syntax import Const, Pred, QFFormula

def get_config_weight_standard(cell_graph: CellGraph,
                               cell_config: dict[Cell, int]) -> RingElement:
    res = Rational(1, 1)
    for cell, n in cell_config.items():
        if n > 0:
            # NOTE: nullary weight is multiplied once
            res = res * cell_graph.get_nullary_weight(cell)
            break
    for i, (cell_i, n_i) in enumerate(cell_config.items()):
        if n_i == 0:
            continue
        res = res * cell_graph.get_cell_weight(cell_i) ** n_i
        res = res * cell_graph.get_two_table_weight(
            (cell_i, cell_i)
        ) ** (n_i * (n_i - 1) // 2)
        for j, (cell_j, n_j) in enumerate(cell_config.items()):
            if j <= i:
                continue
            if n_j == 0:
                continue
            res = res * cell_graph.get_two_table_weight(
                (cell_i, cell_j)
            ) ** (n_i * n_j)
    # logger.debug('Config weight: %s', res)
    return res

def standard_wfomc(formula: QFFormula,
                   domain: set[Const],
                   get_weight: Callable[[Pred], tuple[RingElement, RingElement]]) -> RingElement:
    cell_graph = CellGraph(formula, get_weight)
    # cell_graph.show()
    cells = cell_graph.get_cells()
    n_cells = len(cells)
    domain_size = len(domain)
    MultinomialCoefficients.setup(domain_size)

    res = Rational(0, 1)
    for partition in multinomial(n_cells, domain_size):
        coef = MultinomialCoefficients.coef(partition)
        cell_config = dict(zip(cells, partition))
        res = res + coef * get_config_weight_standard(
            cell_graph, cell_config
        )
    return res