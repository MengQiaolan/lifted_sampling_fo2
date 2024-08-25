from __future__ import annotations

from logzero import logger
from typing import Callable
from contexttimer import Timer

from sampling_fo2.cell_graph.cell_graph import OptimizedCellGraph
from sampling_fo2.utils import MultinomialCoefficients, multinomial_less_than, RingElement, Rational
from sampling_fo2.fol.syntax import Const, Pred, QFFormula
from sampling_fo2.utils.polynomial import expand

def fast_wfomc(formula: QFFormula,
                 domain: set[Const],
                 get_weight: Callable[[Pred], tuple[RingElement, RingElement]],
                 modified_cell_sysmmetry: bool = False) -> RingElement:
    domain_size = len(domain)
    MultinomialCoefficients.setup(domain_size)
    
    with Timer() as t:
        opt_cell_graph = OptimizedCellGraph(
            formula, get_weight, domain_size, modified_cell_sysmmetry
        )
    logger.info('Optimized cell graph construction time: %s', t.elapsed)

    cliques = opt_cell_graph.cliques
    nonind = opt_cell_graph.nonind
    i2_ind = opt_cell_graph.i2_ind
    nonind_map = opt_cell_graph.nonind_map

    res = Rational(0, 1)
    with Timer() as t:
        for partition in multinomial_less_than(len(nonind), domain_size):
            mu = tuple(partition)
            if sum(partition) < domain_size:
                mu = mu + (domain_size - sum(partition), )
            coef = MultinomialCoefficients.coef(mu)
            body = Rational(1, 1)

            for i, clique1 in enumerate(cliques):
                for j, clique2 in enumerate(cliques):
                    if i in nonind and j in nonind:
                        if i < j:
                            body = body * opt_cell_graph.get_two_table_weight(
                                (clique1[0], clique2[0])
                            ) ** (partition[nonind_map[i]] *
                                  partition[nonind_map[j]])

            for l in nonind:
                body = body * opt_cell_graph.get_J_term(
                    l, partition[nonind_map[l]]
                )
                if not modified_cell_sysmmetry:
                    body = body * opt_cell_graph.get_cell_weight(
                        cliques[l][0]
                    ) ** partition[nonind_map[l]]

            opt_cell_graph.setup_term_cache()
            mul = opt_cell_graph.get_term(len(i2_ind), 0, partition)
            res = res + coef * mul * body
            res = expand(res)
            
    logger.info('WFOMC time: %s', t.elapsed)

    return res
