from __future__ import annotations
from collections import defaultdict

from enum import Enum
import os
import argparse
import logging
import logzero
import math

from logzero import logger
from typing import Callable
from contexttimer import Timer

from sampling_fo2.utils import MultinomialCoefficients, multinomial, \
    multinomial_less_than, RingElement, Rational, round_rational
from sampling_fo2.utils.polynomial import coeff_dict, create_vars, expand
from sampling_fo2.cell_graph import CellGraph, CellGraphv2, Cell, OptimizedCellGraph, OptimizedCellGraphv2, OptimizedCellGraphv2_forSymCell
from sampling_fo2.context import WFOMCContext
from sampling_fo2.parser import parse_input
from sampling_fo2.problems import WFOMCSProblem

from sampling_fo2.fol.syntax import Const, Pred, QFFormula, \
    PREDS_FOR_EXISTENTIAL, AUXILIARY_PRED_NAME, top, X, Y
from sampling_fo2.fol.utils import new_predicate, quantified_formula_update, exactly_one_qf


class Algo(Enum):
    STANDARD = 'standard'
    FASTER = 'faster'
    FASTERv2 = 'fasterv2'

    def __str__(self):
        return self.value


def get_config_weight_standard_faster(config: list[int],
                                      cell_weights: list[RingElement],
                                      edge_weights: list[list[RingElement]]) \
        -> RingElement:
    res = Rational(1, 1)
    for i, n_i in enumerate(config):
        if n_i == 0:
            continue
        n_i = Rational(n_i, 1)
        res *= cell_weights[i] ** n_i
        res *= edge_weights[i][i] ** (n_i * (n_i - 1) // Rational(2, 1))
        for j, n_j in enumerate(config):
            if j <= i:
                continue
            if n_j == 0:
                continue
            n_j = Rational(n_j, 1)
            res *= edge_weights[i][j] ** (n_i * n_j)
    return res

def get_config_weight_standard_faster_2(config: list[int],
                                      cell_weights: list[RingElement],
                                      edge_weights: list[list[RingElement]]) \
        -> RingElement:
    res = Rational(1, 1)
    bi = Rational(1, 1)
    for i, n_i in enumerate(config):
        if n_i == 0:
            continue
        n_i = Rational(n_i, 1)
        res *= cell_weights[i] ** n_i
        
        tmp_bi = Rational(1, 1)
        for j, n_j in enumerate(config):
            if n_j == 0:
                continue
            if i == j:
                n_j -= 1
            n_j = Rational(n_j, 1)
            tmp_bi *= edge_weights[i][j] ** (n_j)
        
        from symengine import var, expand
        print(config, tmp_bi)
        tmp_bi_2 = Rational(0, 1)
        for degrees, coef in coeff_dict(expand(tmp_bi), [var('a1'), var('a2')]):
            if degrees[0] == sum(config)-1 or degrees[1] == sum(config)-1:
                print("delete ", coef)
                continue
            tmp_bi_2 += coef
        print(tmp_bi_2)
        bi *= (tmp_bi_2) ** n_i
    # bi = math.sqrt(bi)
    res *= Rational(bi, 1)
    return res


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


def faster_wfomc(opt_cell_graph: OptimizedCellGraphv2,
                 domain_size: int) -> RingElement:

    cliques = opt_cell_graph.cliques
    nonind = opt_cell_graph.nonind
    i2_ind = opt_cell_graph.i2_ind
    nonind_map = opt_cell_graph.nonind_map

    res = Rational(0, 1)
    with Timer() as t:
        for partition in multinomial_less_than(len(nonind), domain_size):
            mu = tuple(partition)
            mu = mu + (domain_size - sum(partition),)
            coef = MultinomialCoefficients.coef(mu)
            body = Rational(1, 1)

            # weight of 'r' for (i, j) in nonind
            for i, clique1 in enumerate(cliques):
                for j, clique2 in enumerate(cliques):
                    if i in nonind and j in nonind:
                        if i < j:
                            body = body * opt_cell_graph.get_two_table_weight(
                                (clique1[0], clique2[0])
                            ) ** (partition[nonind_map[i]] *
                                  partition[nonind_map[j]])
            # weight of 'J' for i in nonind
            for i in nonind:
                body = body * opt_cell_graph.get_J_term(
                    i, partition[nonind_map[i]]
                )

            opt_cell_graph.reset_term_cache()
            mul = opt_cell_graph.get_g_term(len(i2_ind), domain_size - sum(partition), 0, partition)
            res = res + coef * mul * body
    logger.info('WFOMC time: %s', t.elapsed)
    return res

def standard_wfomc(formula: QFFormula,
                   domain_size: int,
                   get_weight: Callable[[Pred], tuple[RingElement, RingElement]]) -> RingElement:
    cell_graph = CellGraphv2(formula, get_weight)
    # cell_graph.show()
    cells = cell_graph.get_cells()
    n_cells = len(cells)
    MultinomialCoefficients.setup(domain_size)

    res = Rational(0, 1)
    for partition in multinomial(n_cells, domain_size):
        coef = MultinomialCoefficients.coef(partition)
        res = res + coef * get_config_weight_standard_faster_2(
            partition, 
            cell_graph.get_all_weights_v2()[0], 
            cell_graph.get_all_weights_v2()[1])
    return res


def precompute_ext_weight(cell_graph: CellGraph, domain_size: int,
                          context: WFOMCContext) \
        -> dict[frozenset[tuple[Cell, frozenset[Pred], int]], RingElement]:
    existential_weights = defaultdict(lambda: Rational(0, 1))
    cells = cell_graph.get_cells()
    eu_configs = []
    for cell in cells:
        config = []
        for domain_pred, tseitin_preds in context.domain_to_evidence_preds.items():
            if cell.is_positive(domain_pred):
                config.append((
                    cell.drop_preds(
                        prefixes=PREDS_FOR_EXISTENTIAL), tseitin_preds
                ))
        eu_configs.append(config)

    cell_weights, edge_weights = cell_graph.get_all_weights()

    for partition in multinomial(len(cells), domain_size):
        # res = get_config_weight_standard(
        #     cell_graph, dict(zip(cells, partition))
        # )
        res = get_config_weight_standard_faster(
            partition, cell_weights, edge_weights
        )
        eu_config = defaultdict(lambda: 0)
        for idx, n in enumerate(partition):
            for config in eu_configs[idx]:
                eu_config[config] += n
        eu_config = dict(
            (k, v) for k, v in eu_config.items() if v > 0
        )
        existential_weights[
            frozenset((*k, v) for k, v in eu_config.items())
        ] += (Rational(MultinomialCoefficients.coef(partition), 1) * res)
    # remove duplications
    for eu_config in existential_weights.keys():
        dup_factor = Rational(MultinomialCoefficients.coef(
            tuple(c[2] for c in eu_config)
        ), 1)
        existential_weights[eu_config] /= dup_factor
    return existential_weights


def wfomc(context: WFOMCContext, domain_size: int = None, algo: Algo = Algo.STANDARD) -> Rational:
    if domain_size is None:
        domain_size = len(context.domain)
    MultinomialCoefficients.setup(domain_size)
    if algo == Algo.STANDARD:
        res = standard_wfomc(
            context.formula, domain_size, context.get_weight
        )
    elif algo == Algo.FASTER:
        raise NotImplementedError
    elif algo == Algo.FASTERv2:
        with Timer() as t:
            opt_cell_graph = OptimizedCellGraphv2(context.formula, 
                                                  context.get_weight)
        logger.info('Optimized cell graph construction time: %s', t.elapsed)
        res = faster_wfomc(opt_cell_graph, domain_size)
    res = context.decode_result(res)
    return res

def count_distribution(problem: WFOMCSProblem, 
                       cells: list[Cell],
                       domain_sizes: list[int] = None) -> dict[tuple[int, ...], Rational]:
    logger.info('Compute count distribution for cells: %s', cells) 
    
    ctx = WFOMCContext(problem)
    if domain_sizes is None:
        domain_sizes = [len(problem.domain)]
    with Timer() as t:
        # NOTE domain_sizes must be sorted in descending order
        MultinomialCoefficients.setup(domain_sizes[0])
        opt_cell_graph = OptimizedCellGraphv2_forSymCell(ctx.formula, 
                                                         cells,
                                                         ctx.get_weight)
    logger.info('Optimized cell graph construction time: %s', t.elapsed)
    res = 0
    for n in domain_sizes:
        wfomc = faster_wfomc(opt_cell_graph, n)
        wfomc = ctx.decode_result(wfomc)
        res += wfomc
    
    count_dist = opt_cell_graph.get_count_dist(res)
    return count_dist

def parse_args():
    parser = argparse.ArgumentParser(
        description='WFOMC for MLN',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--input', '-i', type=str, required=True,
                        help='mln file')
    parser.add_argument('--output_dir', '-o', type=str,
                        default='./check-points')
    parser.add_argument('--algo', '-a', type=Algo,
                        choices=list(Algo), default=Algo.FASTERv2)
    parser.add_argument('--domain_recursive',
                        action='store_true', default=False,
                        help='use domain recursive algorithm '
                             '(only for existential quantified MLN)')
    parser.add_argument('--debug', action='store_true', default=False)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    # import sys
    # sys.setrecursionlimit(int(1e6))
    args = parse_args()
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    if args.debug:
        logzero.loglevel(logging.DEBUG)
    else:
        logzero.loglevel(logging.INFO)
    logzero.logfile('{}/log.txt'.format(args.output_dir), mode='w')

    with Timer() as t:
        problem = parse_input(args.input)
    context = WFOMCContext(problem)
    logger.info('Parse input: %s', t)

    res = wfomc(
        context, algo=args.algo
    )
    logger.info('WFOMC (arbitrary precision): %s', res)
    round_val = round_rational(res)
    logger.info('WFOMC (round): %s (exp(%s))', round_val, round_val.ln())
