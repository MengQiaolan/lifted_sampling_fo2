from __future__ import annotations
from collections import defaultdict, Counter

from enum import Enum
from functools import reduce
from itertools import product
import math
import os
import argparse
import logging
import logzero
import pynauty

from logzero import logger
from typing import Callable
from contexttimer import Timer

from sampling_fo2.cell_graph.cell_graph import OptimizedCellGraph, OptimizedCellGraphWithPC
from sampling_fo2.network.constraint import PartitionConstraint
from sampling_fo2.problems import WFOMCSProblem

from sampling_fo2.utils import MultinomialCoefficients, multinomial, \
    multinomial_less_than, RingElement, Rational, round_rational
from sampling_fo2.cell_graph import CellGraph, Cell
from sampling_fo2.context import WFOMCContext
from sampling_fo2.parser import parse_input
from sampling_fo2.fol.syntax import Const, Pred, QFFormula, PREDS_FOR_EXISTENTIAL
from sampling_fo2.utils.polynomial import coeff_dict, create_vars, expand


class Algo(Enum):
    STANDARD = 'standard'
    FASTER = 'faster'
    FASTERv2 = 'fasterv2'
    INCREMENTAL = 'inc'
    DP = 'dp'

    def __str__(self):
        return self.value

class IGCache(object):
    def init(self, domain_size: int):
        self.cache = []
        self.cache_hit_count = []
        for _ in range(domain_size+1):
            self.cache.append({})
            self.cache_hit_count.append(0)
    
    def get(self, level: int, color_kind: tuple[int], color_count: tuple[int], can_label):
        if color_kind not in self.cache[level]:
            self.cache[level][color_kind] = {}
            return None
        if color_count not in self.cache[level][color_kind]:
            self.cache[level][color_kind][color_count] = {}
            return None
        if can_label not in self.cache[level][color_kind][color_count]:
            return None
        self.cache_hit_count[level] += 1
        return self.cache[level][color_kind][color_count][can_label]
    
    def set(self, level: int, color_kind: tuple[int], color_count: tuple[int], can_label, value):
        if color_kind not in self.cache[level]:
            self.cache[level][color_kind] = {}
        if color_count not in self.cache[level][color_kind]:
            self.cache[level][color_kind][color_count] = {}
        self.cache[level][color_kind][color_count][can_label] = value

# cache for isormophic graph
IG_CACHE = IGCache()
# edge weight matrix of original cell graph
ORIGINAL_WEIGHT_EDGE = []
# how many layers is needed when convert edge-colored graph
COLOR_EDGE_d = 0
# the number of colors of edge
NUM_EDGE_COLOR = 0
# key: edge weight, value: edge color
EDGE_COLOR_MAP = {1:0}
# edge color matrix of original cell graph (ORIGINAL_WEIGHT_EDGE + EDGE_COLOR_MAP = COLOR_EDGE)
COLOR_EDGE = []
# the global no. of color of vertex
NO_VERTEX_COLOR = 0
# the number of vertries of the original cell graph
NUM_CELLS = 0
# the number of vertries of the extend cell graph
NUM_EXT_CELLS = 0
# key: vertex weight, value: vertex color
VERTEX_COLOR_MAP: dict[any, int] = {}
# adjacency_dict
ADJACENCY_DICT = {}
# reduce the call of pynauty.certificate
CACHE_FOR_NAUTY = {} 

def get_vertex_color(weight):
    # weight = int(weight)
    global NO_VERTEX_COLOR
    if weight not in VERTEX_COLOR_MAP:
        VERTEX_COLOR_MAP[weight] = NO_VERTEX_COLOR
        NO_VERTEX_COLOR += 1
    return VERTEX_COLOR_MAP[weight]

def edgeWeight_To_edgeColor():
    global NUM_EDGE_COLOR, EDGE_COLOR_MAP, COLOR_EDGE
    NUM_EDGE_COLOR = 0
    EDGE_COLOR_MAP = {1:0}
    COLOR_EDGE = []
    
    for lst in ORIGINAL_WEIGHT_EDGE:
        tmp_list = []
        for w in lst:
            if w not in EDGE_COLOR_MAP:
                NUM_EDGE_COLOR += 1
                EDGE_COLOR_MAP[w] = NUM_EDGE_COLOR
            tmp_list.append(EDGE_COLOR_MAP[w])
        COLOR_EDGE.append(tmp_list)
    
    global COLOR_EDGE_d, NUM_EXT_CELLS
    # COLOR_EDGE_d = math.ceil(math.sqrt(no_edge_color))
    COLOR_EDGE_d = math.ceil(math.log2(NUM_EDGE_COLOR+1))
    NUM_EXT_CELLS = NUM_CELLS * COLOR_EDGE_d
    
def cellWeight_To_vertexColor(cell_weights):
    vertex_colors = []
    for w in cell_weights:
        vertex_colors.append(get_vertex_color(w))
    color_dict = Counter(vertex_colors)
    color_kind = tuple(sorted(color_dict))
    color_count = tuple(color_dict[num] for num in color_kind)
    return vertex_colors, color_kind, color_count

def calculate_adjacency_dict():
    # Generate new edges
    adjacency_dict = {}
    for i in range(NUM_EXT_CELLS):
        adjacency_dict[i] = []
    
    c2layers = {}
    for k in range(NUM_EDGE_COLOR+1):
        bi = bin(k)
        bi = bi[2:]
        bi = bi[::-1]
        layers = [i for i in range(len(bi)) if bi[i] == '1']
        c2layers[k] = layers
    
    for i in range(NUM_CELLS):
        for j in range(NUM_CELLS):
            layers = c2layers[COLOR_EDGE[i][j]]
            for l in layers:
                adjacency_dict[l*NUM_CELLS+i].append(l*NUM_CELLS+j)
    
    # The vertical threads (each corresponding to one vertex of the original graph) 
    # can be connected using either paths or cliques.
    for i in range(NUM_CELLS):
        clique = [i + j*NUM_CELLS for j in range(COLOR_EDGE_d)]
        for ii in clique:
            for jj in clique:
                if ii == jj:
                    continue
                adjacency_dict[ii].append(jj)
    global ADJACENCY_DICT
    ADJACENCY_DICT = adjacency_dict

def create_graph():
    global GRAPH
    GRAPH = pynauty.Graph(NUM_EXT_CELLS, 
                          directed=False, 
                          adjacency_dict=ADJACENCY_DICT)

def adjust_vertex_coloring(colored_vertices):
    '''
    Adjust the color no. of vertices to make the color no. start from 0 and be continuous.
    
    Args:
        colored_vertices: list[int]
            The color no. of vertices.
            
        Returns:
            new_colored_vertices: list[int]
                The adjusted color no. of vertices.
            num_color: int
                The number of colors. 
    
    Example:
        colored_vertices = [7, 5, 7, 3, 5, 7]
        new_colored_vertices, num_color = adjust_vertex_coloring(colored_vertices)
        print(new_colored_vertices)  # [0, 1, 0, 2, 1, 0]
        print(num_color)  # 3
    '''
    color_map = {}
    new_colored_vertices = []
    num_color = 0
    for c in colored_vertices:
        if c not in color_map:
            color_map[c] = num_color
            num_color += 1
        new_colored_vertices.append(color_map[c])
    return new_colored_vertices, num_color

def extend_vertex_coloring(colored_vertices, no_color):
    '''
    Extend the vertex set to convert colored edge
    
    Args:
        colored_vertices: list[int]
            The color no. of vertices.
        no_color: int
            The number of colors.
    
    Returns:
        vertex_coloring: list[set[int]]
            The color set of vertices.
    
    Example:
        colored_vertices = [0, 1, 0, 2, 1, 0]
        no_color = 3
        vertex_coloring = extend_vertex_coloring(colored_vertices, no_color)
        print(vertex_coloring)  # [{0, 2, 5}, {1, 4}, {3}]
    '''
    # Extend the vertex set to convert colored edge
    ext_colored_vertices = []
    for i in range(COLOR_EDGE_d):
        ext_colored_vertices += [x + no_color * i for x in colored_vertices]
    
    # Get color set of vertices
    no_color *= COLOR_EDGE_d
    vertex_coloring = [ set() for _ in range(no_color)]
    for i in range(len(ext_colored_vertices)):
        c = ext_colored_vertices[i]
        vertex_coloring[c].add(i)
    
    return vertex_coloring

def update_graph(colored_vertices):
    # for speed up, we have modified the function 'set_vertex_coloring' in graph.py of pynauty
    GRAPH.set_vertex_coloring(colored_vertices)
    return GRAPH

def dfs_wfomc(cell_weights, domain_size):
    if domain_size == 1:
        return sum(cell_weights)
    
    res = 0
    for l in range(NUM_CELLS):
        w_l = cell_weights[l]
        new_cell_weights = [cell_weights[i] * ORIGINAL_WEIGHT_EDGE[l][i] for i in range(NUM_CELLS)]
        
        # GCD is complex when calculating symbolic expression
        # gcd, new_cell_weights = get_gcd(new_cell_weights)
        
        original_vertex_colors, vertex_color_kind, vertex_color_count = cellWeight_To_vertexColor(new_cell_weights) # convert cell weights to vertex colors
        adjust_vertex_colors, no_color = adjust_vertex_coloring(original_vertex_colors) # # adjust the color no. of vertices to make them start from 0 and be continuous
        
        
        # can_label = tuple(adjust_vertex_colors) # using this line and disable following code will cancel optimziation of isomorphism
        # cache for nauty
        if tuple(adjust_vertex_colors) not in CACHE_FOR_NAUTY:
            can_label = pynauty.certificate(update_graph(extend_vertex_coloring(adjust_vertex_colors, no_color)))
            CACHE_FOR_NAUTY[tuple(adjust_vertex_colors)] = can_label
        else:
            can_label = CACHE_FOR_NAUTY[tuple(adjust_vertex_colors)]
            
        
        value = IG_CACHE.get(domain_size-1, vertex_color_kind, vertex_color_count, can_label)
        if value is None:
            value = dfs_wfomc(new_cell_weights, domain_size - 1)
            value = expand(value)
            IG_CACHE.set(domain_size-1, vertex_color_kind, vertex_color_count, can_label, value)
        res += (w_l * value) #* gcd**(domain_size-1)
    
    return res

def dp_wfomc_init(domain_size, cell_weights, edge_weights):
    IG_CACHE.init(domain_size)
    global ORIGINAL_WEIGHT_EDGE, NUM_CELLS
    # not change in the same problem
    NUM_CELLS = len(cell_weights)
    ORIGINAL_WEIGHT_EDGE = edge_weights
    edgeWeight_To_edgeColor()
    calculate_adjacency_dict()
    create_graph()

def dp_wfomc(formula: QFFormula,
             domain: set[Const],
             get_weight: Callable[[Pred], tuple[RingElement, RingElement]],
             leq_pred: Pred = None) -> RingElement:
    domain_size = len(domain)
    cell_graph = CellGraph(formula, get_weight, leq_pred)
    cell_weights = cell_graph.get_all_weights()[0]
    edge_weights = cell_graph.get_all_weights()[1]
    
    dp_wfomc_init(domain_size, cell_weights, edge_weights)
    return dfs_wfomc(cell_weights, domain_size)



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


def faster_wfomc_with_pc(formula: QFFormula,
                 domain: set[Const],
                 get_weight: Callable[[Pred], tuple[RingElement, RingElement]],
                 partition_constraint: PartitionConstraint) -> RingElement:
    domain_size = len(domain)
    with Timer() as t:
        opt_cell_graph = OptimizedCellGraphWithPC(
            formula, get_weight, domain_size, partition_constraint
        )
    logger.info('Optimized cell graph construction time: %s', t.elapsed)

    cliques = opt_cell_graph.cliques
    nonind = opt_cell_graph.nonind
    nonind_map = opt_cell_graph.nonind_map

    pred_partitions: list[list[int]] = list(num for _, num in partition_constraint.partition)
    # partition to cliques
    partition_cliques: dict[int, list[int]] = opt_cell_graph.partition_cliques

    res = Rational(0, 1)
    with Timer() as t:
        for configs in product(
            *(list(multinomial_less_than(len(partition_cliques[idx]), constrained_num)) for
              idx, constrained_num in enumerate(pred_partitions))
        ):
            coef = Rational(1, 1)
            remainings = list()
            # config for the cliques
            overall_config = list(0 for _ in range(len(cliques)))
            # {clique_idx: [number of elements of pred1, pred2, ..., predk]}
            clique_configs = defaultdict(list)
            for idx, (constrained_num, config) in enumerate(zip(pred_partitions, configs)):
                remainings.append(constrained_num - sum(config))
                mu = tuple(config) + (constrained_num - sum(config), )
                coef = coef * MultinomialCoefficients.coef(mu)
                for num, clique_idx in zip(config, partition_cliques[idx]):
                    overall_config[clique_idx] = overall_config[clique_idx] + num
                    clique_configs[clique_idx].append(num)

            body = opt_cell_graph.get_i1_weight(
                remainings, overall_config
            )

            for i, clique1 in enumerate(cliques):
                for j, clique2 in enumerate(cliques):
                    if i in nonind and j in nonind:
                        if i < j:
                            body = body * opt_cell_graph.get_two_table_weight(
                                (clique1[0], clique2[0])
                            ) ** (overall_config[nonind_map[i]] *
                                  overall_config[nonind_map[j]])

            for l in nonind:
                body = body * opt_cell_graph.get_J_term(
                    l, tuple(clique_configs[nonind_map[l]])
                )
            res = res + coef * body

            # opt_cell_graph.setup_term_cache()
            # res = res + coef * body
            # mul = opt_cell_graph.get_term(len(i2_ind), 0, partition)
            # res = res + coef * mul * body
    logger.info('WFOMC time: %s', t.elapsed)

    return res


def faster_wfomc(formula: QFFormula,
                 domain: set[Const],
                 get_weight: Callable[[Pred], tuple[RingElement, RingElement]],
                 modified_cell_sysmmetry: bool = False) -> RingElement:
    domain_size = len(domain)
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
    logger.info('WFOMC time: %s', t.elapsed)

    return res

def standard_wfomc(formula: QFFormula,
                   domain: set[Const],
                   get_weight: Callable[[Pred], tuple[RingElement, RingElement]]) -> RingElement:
    cell_graph = CellGraph(formula, get_weight)
    # cell_graph.show()
    cells = cell_graph.get_cells()
    n_cells = len(cells)
    domain_size = len(domain)

    res = Rational(0, 1)
    for partition in multinomial(n_cells, domain_size):
        coef = MultinomialCoefficients.coef(partition)
        cell_config = dict(zip(cells, partition))
        # logger.debug(
        #     '=' * 15 + ' Compute WFOMC for the partition %s ' + '=' * 15,
        #     dict(filter(lambda x: x[1] != 0, cell_config.items())
        # ))
        res = res + coef * get_config_weight_standard(
            cell_graph, cell_config
        )
    return res


def incremental_wfomc(formula: QFFormula,
                      domain: set[Const],
                      get_weight: Callable[[Pred], tuple[RingElement, RingElement]],
                      leq_pred: Pred = None) -> RingElement:
    cell_graph = CellGraph(formula, get_weight, leq_pred)
    # cell_graph.show()
    cells = cell_graph.get_cells()
    n_cells = len(cells)
    domain_size = len(domain)

    table = dict(
        (tuple(int(k == i) for k in range(n_cells)),
         cell_graph.get_cell_weight(cell))
        for i, cell in enumerate(cells)
    )
    for _ in range(domain_size - 1):
        old_table = table
        table = dict()
        for j, cell in enumerate(cells):
            w = cell_graph.get_cell_weight(cell)
            for ivec, w_old in old_table.items():
                w_new = w_old * w * reduce(
                    lambda x, y: x * y,
                    (cell_graph.get_two_table_weight((cell, cells[k])) ** int(ivec[k]) for k in range(n_cells)),
                    Rational(1, 1)
                )
                ivec = list(ivec)
                ivec[j] += 1
                ivec = tuple(ivec)

                w_new = w_new + table.get(ivec, Rational(0, 1))
                table[tuple(ivec)] = w_new
    ret = sum(table.values())

    # if leq_pred is not None:
    #     ret *= Rational(math.factorial(domain_size), 1)
    return ret



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


def wfomc(problem: WFOMCSProblem, algo: Algo = Algo.STANDARD,
          use_partition_constraint: bool = False) -> Rational:
    if use_partition_constraint:
        algo = Algo.FASTERv2
    # both standard and faster WFOMCs need precomputation
    if algo == Algo.STANDARD or algo == Algo.FASTER or \
            algo == algo.FASTERv2:
        MultinomialCoefficients.setup(len(problem.domain))

    context = WFOMCContext(problem, use_partition_constraint)
    leq_pred = Pred('LEQ', 2)
    if leq_pred in context.formula.preds() and algo != Algo.DP:
        logger.info('Linear order axiom with the predicate LEQ is found')
        logger.info('Invoke incremental WFOMC')
        algo = Algo.INCREMENTAL

    with Timer() as t:
        if algo == Algo.STANDARD:
            res = standard_wfomc(
                context.formula, context.domain, context.get_weight
            )
        elif algo == Algo.FASTER:
            res = faster_wfomc(
                context.formula, context.domain, context.get_weight
            )
        elif algo == Algo.FASTERv2:
            if context.contain_partition_constraint():
                res = faster_wfomc_with_pc(
                    context.formula, context.domain,
                    context.get_weight,
                    context.partition_constraint,
                )
            else:
                res = faster_wfomc(
                    context.formula, context.domain, context.get_weight, True
                )
        elif algo == Algo.INCREMENTAL:
            res = incremental_wfomc(
                context.formula, context.domain,
                context.get_weight, leq_pred
            )
        elif algo == Algo.DP:
            logger.info('Invoke DP WFOMC')
            res = dp_wfomc(
                context.formula, context.domain, context.get_weight, leq_pred
            )
    res = context.decode_result(res)
    logger.info('WFOMC time: %s', t.elapsed)
    return res


def count_distribution(context: WFOMCContext, preds: list[Pred],
                       algo: Algo = Algo.STANDARD) \
        -> dict[tuple[int, ...], Rational]:
    pred2weight = {}
    pred2sym = {}
    syms = create_vars('x0:{}'.format(len(preds)))
    for sym, pred in zip(syms, preds):
        if pred in pred2weight:
            continue
        weight = context.get_weight(pred)
        pred2weight[pred] = (weight[0] * sym, weight[1])
        pred2sym[pred] = sym
    context.weights.update(pred2weight)
    if algo == Algo.STANDARD:
        res = standard_wfomc(
            context.formula, context.domain, context.get_weight
        )
    elif algo == Algo.FASTER:
        res = faster_wfomc(
            context.formula, context.domain, context.get_weight
        )

    symbols = [pred2sym[pred] for pred in preds]
    count_dist = {}
    res = expand(res)
    for degrees, coef in coeff_dict(res, symbols):
        count_dist[degrees] = coef
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
                        choices=list(Algo), default=Algo.FASTER)
    parser.add_argument('--use_partition_constraint', '-p',
                        action='store_true', default=False,
                        help='use partition constraint to handle unary evidence')
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
    logger.info('Parse input: %ss', t)

    res = wfomc(
        problem, algo=args.algo, use_partition_constraint=args.use_partition_constraint,
    )
    logger.info('WFOMC (arbitrary precision): %s', res)
    round_val = round_rational(res)
    logger.info('WFOMC (round): %s (exp(%s))', round_val, round_val.ln())
