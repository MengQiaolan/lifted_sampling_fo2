from __future__ import annotations

import random
import numpy as np
import argparse
import os
import logzero
import logging
import pickle

from enum import Enum
from tqdm import tqdm
from logzero import logger
from contexttimer import Timer
from collections import defaultdict

from sampling_fo2.context import WFOMSContext
from sampling_fo2.context.existential_context import \
    BlockType, ExistentialContext, ExistentialTwoTable
from sampling_fo2.fol.syntax import AtomicFormula, Const, a, b, \
    AUXILIARY_PRED_NAME, PREDS_FOR_EXISTENTIAL
from sampling_fo2.utils import MultinomialCoefficients, multinomial, \
    Rational, RingElement, coeff_monomial, round_rational, expand, \
    multinomial_less_than
from sampling_fo2.parser import parse_input
from sampling_fo2.cell_graph import Cell, CellGraph, \
    OOptimizedCellGraph, OptimizedCellGraphWoClique
from sampling_fo2.utils.polynomial import choices

from sampling_fo2.wfomc import get_config_weight_standard_faster, \
    get_config_weight_standard

class Samplerv2(object):
    def __init__(self, context: WFOMSContext):
        self.context: WFOMSContext = context
        get_weight = self.context.get_weight

        self.domain: list[Const] = list(self.context.domain)
        self.domain_size: int = len(self.domain)
        logger.debug('domain: %s', self.domain)
        
        self.cell_graph: CellGraph = CellGraph(
            self.context.formula, get_weight
        )
        MultinomialCoefficients.setup(self.domain_size)
        self.configs, self.weights = self._get_config_weights(
            self.cell_graph, self.domain_size)
        
        print("---------- 1 -------------")
        print(self.context.formula)
        for c in self.cell_graph.cells:
            print(self.cell_graph.get_cell_weight(c), ": ", c)
        print(self.configs, self.weights)

        self.opt_cell_graph: OOptimizedCellGraph = OOptimizedCellGraph(
            self.context.formula, get_weight, self.domain_size
        )
        self.noind_clique_configs, self.noind_clique_weights = \
            self._get_noind_clique_config_weights(self.opt_cell_graph, self.domain_size)

        if self.context.contain_existential_quantifier():
            # Precomputed weights for cell configs
            self.uni_cell_graph: CellGraph = CellGraph(
                self.context.uni_formula, get_weight
            )
            
            self.configs, self.weights = self._adjust_config_weights(
                self.configs, self.weights,
                self.cell_graph, self.uni_cell_graph
            )
            # print()
            # print("---------- 2 ------------- 1-type config")
            # print(self.context.uni_formula)
            # for c in self.uni_cell_graph.cells:
            #     print(self.uni_cell_graph.get_cell_weight(c), ": ", c)
            # print(self.configs, self.weights)
            self.cell_graph = self.uni_cell_graph
        
        wfomc = sum(self.noind_clique_weights)
        if wfomc == 0:
            raise RuntimeError(
                'Unsatisfiable formula!!'
            )
        round_val = round_rational(wfomc)
        logger.info('wfomc (round):%s (exp(%s))',
                    round_val, round_val.ln())
        self.cells = self.opt_cell_graph.get_cells()
        
        if self.context.contain_existential_quantifier():
            self.block_cell_graph: CellGraph = CellGraph(
                self.context.block_encoded_formula, get_weight
            )
            # Precomputed weights for cell + block configuration
            # NOTE: here the cell + block is the `CELL` in the sampling paper
            
            self.cb_weights: dict[
                frozenset[tuple[Cell, BlockType, int]], RingElement
            ] = dict()
            logger.info('Pre-compute the weights for existential quantifiers')
            # time: [n] * C(n, n + |CELL| - 1)
            for n in range(1, self.domain_size):
                self.cb_weights.update(
                    self._precompute_cb_weight(
                        self.block_cell_graph, n, self.context)
                )
            
            # num of keys of cb_weights is 
            # multinomial(|1-types|*|block-types|, 1) + ... + multinomial(|1-types|*|block-types|, n-1)
            # = 
            # multinomial(|CELLs|, 1) + ... + multinomial(|CELLs|, n-1)
            
            logger.debug('pre-computed weights for existential quantifiers:\n%s',
                         self.cb_weights)
            print("---------- 3 -------------")
            print(self.context.block_encoded_formula)
            for c in self.block_cell_graph.cells:
                print(self.block_cell_graph.get_cell_weight(c), ": ", c)
            print("--")
            for k,v in self.cb_weights.items():
                print(k, ":", v)

        self.cells = self.cell_graph.get_cells()
        # for measuring performance
        self.t_sampling = 0
        self.t_assigning = 0
        self.t_sampling_models = 0
    
    def _precompute_cb_weight(self, cell_graph: CellGraph, domain_size: int,
                              context: WFOMSContext) \
            -> dict[frozenset[tuple[Cell, BlockType, int]], RingElement]:
        cb_weights = defaultdict(lambda: Rational(0, 1))
        cells = cell_graph.get_cells()
        # cell + block configuration, i.e., the `CELL` configuration in
        # the sampling paper
        cb_configs = []
        for cell in cells:
            config = []
            for domain_pred, block_type in context.blockpred_to_blocktype.items():
                if cell.is_positive(domain_pred):
                    config.append((
                        cell.drop_preds(prefixes=PREDS_FOR_EXISTENTIAL),
                        block_type
                    ))
            cb_configs.append(config)

        cell_weights, edge_weights = cell_graph.get_all_weights()

        print(cells)
        print(cell_weights)
        print(domain_size)
        for partition in multinomial(len(cells), domain_size):
            res = get_config_weight_standard_faster(
                partition, cell_weights, edge_weights
            )
            cb_config = defaultdict(lambda: 0)
            for idx, n in enumerate(partition):
                for config in cb_configs[idx]:
                    cb_config[config] += n
            cb_config = dict(
                (k, v) for k, v in cb_config.items() if v > 0
            )
            cb_weights[
                frozenset((*k, v) for k, v in cb_config.items())
            ] += (Rational(MultinomialCoefficients.coef(partition), 1) * res)
            print(partition)
            print(cb_config)
            print("key: ", frozenset((*k, v) for k, v in cb_config.items()))
            print("val: ", cb_weights[frozenset((*k, v) for k, v in cb_config.items())])
        print("cb_weights:")
        for k,v in cb_weights.items():
            print(k, ":", v)
        # remove duplications
        for cb_config in cb_weights.keys():
            dup_factor = Rational(MultinomialCoefficients.coef(
                tuple(c[2] for c in cb_config)
            ), 1)
            cb_weights[cb_config] /= dup_factor
        
        print("cb_weights:")
        for k,v in cb_weights.items():
            print(k, ":", v)
        return cb_weights
    
    def _sample_ext_evidences(self, cell_assignment: list[Cell],
                              cell_weight: RingElement) \
            -> dict[tuple[int, int], frozenset[AtomicFormula]]:
        ext_context = ExistentialContext(
            cell_assignment, self.context.binary_ext_preds)

        # Get the total weight of the current configuration
        cell_config = tuple(cell_assignment.count(cell) for cell in self.cells)
        total_weight = self.weights[self.configs.index(
            cell_config)] / MultinomialCoefficients.coef(cell_config)

        pair_evidences: dict[tuple[int, int],
                             frozenset[AtomicFormula]] = dict()
        q = Rational(1, 1)
        print(ext_context.cell_config)
        print(ext_context.cb_config)
        
        # 已经确定了所有 element 的 1-type 和 block-type（默认是满的）
        # 即确定了 cell-config 和 cb-config
        
        # time: n * |CELL| * C(n_i, n_i + 4^|ext_preds| - 1)
        
        # 如果 cb_config 中的 config 和 cell_config 的一一对应，
        # 那就满足所有的 existential 条件了，不用再处理
        # cb_config 是关于 CELL 的，一共有 |CELL| = |1-types|*|block-types| 个 unit
        # cell_config 是关于 1-types 的，一共有 |1-types| 个 unit
        # 这两个 config 一一对应的意思就是
        # cb_config 中 「1-type + 空 block」 的项能和 cell_config 中一一对应
        # 一般来说，最开始的时候
        # cb_config 应该是「1-type + 满 block」 的项和 cell_config 中一一对应
        # 满 block 就是所有的 ext_preds 都是 positive 的（待满足的）
        while not ext_context.all_satisfied():
            print("======== ======== ======== ======== ======== ======== ======== ======== ======== ======== ======== ======== ======== ======== ======== ========")
            print(ext_context.cb_config)
            # 选择一个 CELL 来处理（注意，这里一定会选择 block 不是空且 cfg 大于 0 的 CELL）
            # TODO
            selected_cell, selected_block = ext_context.select_cell_block_type()
            # 从 ext_context 的一些记录中（cell_config, cb_config, cb_elements）
            # 删去选中的 CELL 的相关计数（-1）, 并挑一个 element
            selected_idx = ext_context.reduce_element(
                selected_cell, selected_block)
            logger.debug('select element: %s, cell: %s, block type: %s',
                         selected_idx, selected_cell, selected_block)
            etable_weights: dict[
                Cell, dict[ExistentialTwoTable, RingElement]
            ] = dict()
            
            print(ext_context.cb_config)
            
            # filter all impossible existential 2tables
            for cell in self.cells:
                cell_pair = (selected_cell, cell)
                weights = dict()
                for etable in self.context.etables:
                    evidences = etable.get_evidences()
                    if self.cell_graph.satisfiable(cell_pair, evidences):
                        weights[etable] = self.cell_graph.get_two_table_weight(
                            cell_pair, evidences
                        )
                etable_weights[cell] = weights

            # 遍历每一个 「selected CELL 与每一类 CELL 的 ext-2-tables 的 config（etable_config）」
            # 这里的 etable_config 的 key 是 CELL (|1-types|*|block-types|)
            # 考虑 selected CELL 与每一个 CELL 的 ext-relation 的 2-tables 的 config (4^|ext_preds|)
            # 假如 CELL_i 的 config 是 n_i （n_i 是已经确定的，在 cb-config 中有）
            # 那么 CELL_i 的 ext-relation 的 2-tables 的 config 有 C(n_i, n_i + 4^|ext_preds| - 1) 种情况
            # 一共要循环 |CELL| * C(n_i, n_i + 4^|ext_preds| - 1) 次
            for etable_config in ext_context.iter_etable_config(
                etable_weights
            ):
                print("======== ======== ======== ======== ======== ======== ======== ========")
                print("========")
                for k,v in etable_config.items():
                    print(k, ". ")
                    for k2,v2 in v.items():
                        print(" ", k2, ":", v2)
                print("====end====")
                
                # etable_config 的 key 是 CELL （1-type + block-type）
                # etable_config_per_cell 的 key 是 1-type
                # overall_etable_config 的 key 是 ext-relation 的 2-type，也即是前两个所有 v 的和
                
                etable_config_per_cell = defaultdict(
                    lambda: defaultdict(lambda: 0)
                )
                # overall_etable_config 记录了所有关于 ext-relation 的 2-tables 的 config
                # overall_etable_config 不关心 CELL，只关心 ext-relation 的 2-tables
                # 它用来判断本轮循环中的 etable_config 是否能够满足 selected CELL 的 block 要求
                # 因为 selected CELL 的 block（非空）肯定会对 ext-2-table 产生要求
                # （blocl对应的ext-2-table 的 config 必须大于0）
                # 所以用 overall_etable_config 来判断 etable_config 是否合理
                overall_etable_config = defaultdict(lambda: 0)
                for (cell, _), config in etable_config.items():
                    for etable, num in config.items():
                        etable_config_per_cell[cell][etable] += num
                        overall_etable_config[etable] += num

                # for k, v in etable_config_per_cell.items():
                #     print(k, ":")
                #     for k2,v2 in v.items():
                #         print(" ", k2, ":", v2)
                # for k, v in overall_etable_config.items():
                #     print(k, ":", v)
                
                # 判断在 overall_etable_config 所有的（num大于0） pair 中，
                # selected_block 中包含的 ext_preds 能不能在某个或者某几个的 pair 的 a/b 中找到
                # 如果可以，那么返回 true，继续往下执行
                # 如果不可以，说明我们选择的 CELL 不满足这个 config
                # 即这个 config 不可能出现（不合理的）
                if not ext_context.satisfied(selected_block, overall_etable_config):
                    continue
                
                print("---- continue ----")
                
                # ext-2-table config 的系数
                coeff = Rational(1, 1)
                for _, config in etable_config.items():
                    coeff *= Rational(MultinomialCoefficients.coef(
                        tuple(config.values())), 1)

                # ext-2-table config 的 2-table 的 weight
                total_weight_etable = Rational(1, 1)
                for cell, config in etable_config_per_cell.items():
                    for etable, num in config.items():
                        total_weight_etable *= (
                            etable_weights[cell][etable] ** Rational(num, 1)
                        )
                
                # 采用了当前 etable_config 之后，新的 CELL_config (cb_config)
                # 和对应的 CELL_config 的 weight
                reduced_cb_config = ext_context.reduce_cb_config(etable_config)
                reduced_weight = self.cb_weights[reduced_cb_config]
                print(ext_context.cb_config)
                print(reduced_cb_config)
                
                
                
                cell_weight = self.cell_graph.get_cell_weight(selected_cell)
                w = self.context.decode_result(
                    q * total_weight_etable * cell_weight *
                    coeff * reduced_weight
                )
                # logger.debug(eb_config)
                # logger.debug('%s %s', w, total_weight)
                if random.random() < w / total_weight:
                    logger.debug('selected etable config:\n%s', etable_config)
                    etable_indices = ext_context.sample_and_update(
                        etable_config)
                    logger.debug('sampled evidences in this step:')
                    for etable, indices in etable_indices.items():
                        for idx in indices:
                            # NOTE: the element order in pair evidences matters!
                            if selected_idx < idx:
                                pair_evidences[(selected_idx, idx)
                                               ] = etable.get_evidences()
                            else:
                                pair_evidences[(idx, selected_idx)
                                               ] = etable.get_evidences(True)
                            logger.debug('(%s, %s): %s',
                                         selected_idx, idx, etable)
                    # Now the ebtype assignement has been determined!
                    total_weight = w / coeff
                    q *= (cell_weight * total_weight_etable)
                    break
                else:
                    total_weight -= w
        return pair_evidences
    
    def _adjust_config_weights(self, configs: list[tuple[int, ...]],
                               weights: list[RingElement],
                               src_cell_graph: CellGraph,
                               dest_cell_graph: CellGraph) -> \
            tuple[list[tuple[int, ...]], list[Rational]]:
        src_cells = src_cell_graph.get_cells()
        dest_cells = dest_cell_graph.get_cells()
        mapping_mat = np.zeros(
            (len(src_cells), len(dest_cells)), dtype=np.int32)
        for idx, cell in enumerate(src_cells):
            dest_idx = dest_cells.index(
                cell.drop_preds(prefixes=PREDS_FOR_EXISTENTIAL)
            )
            mapping_mat[idx, dest_idx] = 1
        
        adjusted_config_weight = defaultdict(lambda: Rational(0, 1))
        for config, weight in zip(configs, weights):
            adjusted_config_weight[tuple(np.dot(
                config, mapping_mat).tolist())] += weight
        return list(adjusted_config_weight.keys()), \
            list(adjusted_config_weight.values())


    def _compute_wmc_prod(
        self, cell_assignment: list[Cell],
        pair_evidences: dict[tuple[int, int], frozenset[AtomicFormula]] = None
    ) -> list[RingElement]:
        wmc_prod = [Rational(1, 1)]
        n_elements = len(cell_assignment)
        # compute from back to front
        for i in range(n_elements - 1, -1, -1):
            for j in range(n_elements - 1, max(i, 0), -1):
                cell_pair = (cell_assignment[i], cell_assignment[j])
                pair = (i, j)
                if pair_evidences is not None and pair in pair_evidences:
                    twotable_weight = self.opt_cell_graph.get_two_table_weight(
                        cell_pair, frozenset(pair_evidences[pair])
                    )
                else:
                    twotable_weight = self.opt_cell_graph.get_two_table_weight(
                        cell_pair)
                prod = wmc_prod[0] * twotable_weight
                wmc_prod.insert(0, prod)
        return wmc_prod

    def _get_unary_atoms(self, cell_assignment: list[Cell]) -> set[AtomicFormula]:
        sampled_atoms = set()
        for idx, cell in enumerate(cell_assignment):
            evidences = cell.get_evidences(self.domain[idx])
            positive_lits = filter(lambda lit: lit.positive, evidences)
            sampled_atoms.update(set(positive_lits))
        return sampled_atoms

    def _get_weight_poly(self, weight: RingElement):
        if self.context.contain_cardinality_constraint():
            return coeff_monomial(expand(weight), self.monomial)
        return weight

    def _sample_binary_atoms(self, cell_assignment: list[Cell],
                             cell_weight: RingElement,
                             binary_evidences: frozenset[AtomicFormula] = None,
                             pair_evidences: dict[tuple[int, int],
                                                  frozenset[AtomicFormula]] = None) -> set[AtomicFormula]:
        # NOTE: here the element order matters in pair_evidences!!!
        if pair_evidences is None:
            pair_evidences = defaultdict(list)
            if binary_evidences is not None:
                for evidence in binary_evidences:
                    # NOTE: we always deal with the index of domain elements here!
                    pair_index = tuple(self.domain.index(c)
                                       for c in evidence.atom.args)
                    assert len(pair_index) == 2
                    if pair_index[0] < pair_index[1]:
                        evidence = AtomicFormula(evidence.pred()(
                            a, b), evidence.positive)
                    else:
                        pair_index = (pair_index[1], pair_index[0])
                        evidence = AtomicFormula(evidence.pred()(
                            b, a), evidence.positive)
                    pair_evidences[pair_index].append(evidence)
        wmc_prod = self._compute_wmc_prod(cell_assignment, pair_evidences)
        total_weight = self.context.decode_result(cell_weight * wmc_prod[0])
        q = Rational(1, 1)
        idx = 1
        sampled_atoms = set()
        for i, cell_1 in enumerate(cell_assignment):
            for j, cell_2 in enumerate(cell_assignment):
                if i >= j:
                    continue
                logger.debug('Sample the atom consisting of %s(%s) and %s(%s)',
                             i, self.domain[i], j, self.domain[j])
                # go through all two tables
                evidences = None
                if (i, j) in pair_evidences:
                    evidences = frozenset(pair_evidences[(i, j)])
                twotables_with_weight = self.opt_cell_graph.get_two_tables(
                    (cell_1, cell_2), evidences
                )
                # compute the sampling distribution
                for twotable, twotable_weight in twotables_with_weight:
                    gamma_w = self.context.decode_result(
                        cell_weight * q * twotable_weight * wmc_prod[idx]
                    )
                    if random.random() < gamma_w / total_weight:
                        sampled_raw_atoms = [
                            lit for lit in twotable if lit.positive
                        ]
                        r_hat = twotable_weight
                        total_weight = gamma_w
                        break
                    else:
                        total_weight -= gamma_w
                # replace to real domain elements
                sampled_atoms_replaced = set(
                    self._replace_consts(
                        atom,
                        {a: self.domain[i], b: self.domain[j]}
                    ) for atom in sampled_raw_atoms
                )
                sampled_atoms.update(sampled_atoms_replaced)
                # update q
                q *= r_hat
                # move forward
                idx += 1
                logger.debug(
                    'sampled atoms at this step: %s', sampled_atoms_replaced
                )
                logger.debug('updated q: %s', q)
        return sampled_atoms

    def sample_on_config(self, config):
        logger.debug('sample on cell configuration %s', config)
        # shuffle domain elements
        random.shuffle(self.domain)
        with Timer() as t:
            cell_assignment, cell_weight = self._assign_cell(
                self.opt_cell_graph, dict(
                    zip(self.cells, config))
            )
            sampled_atoms: set = self._remove_aux_atoms(
                self._get_unary_atoms(cell_assignment)
            )
            logger.debug('initial unary atoms: %s', sampled_atoms)
            self.t_assigning += t.elapsed

        pair_evidences = None
        if self.context.contain_existential_quantifier():
            # pair_evidences = self._sample_ext_evidences(
            #     cell_assignment, cell_weight)
            # logger.debug('sampled existential quantified literals: %s',
            #              pair_evidences)
            raise NotImplementedError

        with Timer() as t:
            sampled_atoms.update(
                self._sample_binary_atoms(
                    cell_assignment, cell_weight,
                    pair_evidences=pair_evidences,
                )
            )
            self.t_sampling_models += t.elapsed
        return self._remove_aux_atoms(sampled_atoms)

    def _remove_aux_atoms(self, atoms):
        # only return atoms with the predicate in the original input
        return set(
            filter(lambda atom: not atom.pred.name.startswith(AUXILIARY_PRED_NAME), atoms)
        )

    def _replace_consts(self, term, replacement):
        if isinstance(term, AtomicFormula):
            args = [replacement.get(a) for a in term.args]
            return term.pred(*args) if term.positive else ~term.pred(*args)
        else:
            raise RuntimeError(
                'Unknown type to replace constant %s', type(term)
            )

    def _get_config_weights(self, cell_graph: CellGraph, domain_size: int) \
            -> tuple[list[tuple[int, ...]], list[Rational]]:
        configs = []
        weights = []
        cells = cell_graph.get_cells()
        n_cells = len(cells)
        for partition in multinomial(n_cells, domain_size):
            coef = MultinomialCoefficients.coef(partition)
            cell_config = dict(zip(cells, partition))
            weight = coef * get_config_weight_standard(
                cell_graph, cell_config
            )
            weight = self.context.decode_result(weight)
            if weight != 0:
                configs.append(partition)
                weights.append(weight)
        return configs, weights

    def _get_noind_clique_config_weights(self, opt_cell_graph: OOptimizedCellGraph, domain_size: int) \
            -> tuple[list[tuple[int, ...]], list[Rational]]:

        cliques = opt_cell_graph.cliques
        i1_ind = opt_cell_graph.i1_ind
        i2_ind = opt_cell_graph.i2_ind
        nonind = opt_cell_graph.nonind
        k, l = len(i1_ind), len(i2_ind)

        noind_clique_configs, noind_clique_weights = [], []
        for partition in multinomial_less_than(len(nonind), domain_size):
            mu = tuple(partition)
            mu = mu + (domain_size - sum(partition),)
            coef = MultinomialCoefficients.coef(mu)
            body = Rational(1, 1)
            
            for i in range(k+l, len(cliques)):
                for j in range(i, len(cliques)):
                    body = body * opt_cell_graph.get_two_table_weight(
                        (cliques[i][0], cliques[j][0])
                    ) ** (partition[i-k-l] *
                          partition[j-k-l])
                            
            for i in range(k+l, len(cliques)):
                body = body * opt_cell_graph.get_J_term(
                    i, partition[i-k-l]
                )

            # weight of I1 and I2
            opt_cell_graph.reset_term_cache()
            mul = opt_cell_graph.get_g_term(l, 0, partition)

            # config of sysmmetric cliques
            noind_clique_configs.append(partition)
            noind_clique_weights.append(coef * mul * body)

        return noind_clique_configs, noind_clique_weights

    def _assign_cell(self, cell_graph: CellGraph,
                     config: dict[Cell, int]) -> tuple[list[Cell], RingElement]:
        cell_assignment = list()
        w = Rational(1, 1)
        for cell, n in config.items():
            for _ in range(n):
                cell_assignment.append(cell)
                w = w * cell_graph.get_cell_weight(cell)
        return cell_assignment, w
    
    def sample_in_clique(self, clique_idx, remian):
        config_in_clique = ()
        cliques = self.opt_cell_graph.cliques
        clique_size = len(cliques[clique_idx])
        for i in range(clique_size):
            if remian == 0:
                config_in_clique = config_in_clique + (0,)
                continue
            if i == clique_size - 1:
                config_in_clique = config_in_clique + (remian,)
                continue
            w = self.opt_cell_graph.d_term_cache[(clique_idx, remian, i)]
            sampled_n = choices(list(range(remian+1)), weights=w)[0]
            remian = remian - sampled_n
            config_in_clique = config_in_clique + (sampled_n,)
        return config_in_clique
        
    def sample_i1_clique(self, remain, sampled_noind_clique_config):
        sampled_i1_clique_config = ()
        i1_ind = self.opt_cell_graph.i1_ind
        nonind = self.opt_cell_graph.nonind
        nonind_map = self.opt_cell_graph.nonind_map
        cliques = self.opt_cell_graph.cliques
        for i in reversed(i1_ind):
            if i == i1_ind[0]:
                sampled_i1_clique_config = (remain,) + sampled_i1_clique_config
            else:
                i1_ind_weights = []
                for nval in range(remain):
                    tmp_weight = MultinomialCoefficients.comb(remain, nval)

                    tmp_weight *= self.opt_cell_graph.cache_for_1ind[i] ** (remain - nval)

                    tmp = self.opt_cell_graph.get_cell_weight(cliques[i][0])
                    for j in nonind:
                        tmp = tmp * self.opt_cell_graph.get_two_table_weight(
                            (cliques[i][0], cliques[j][0])
                        ) ** (sampled_noind_clique_config[nonind_map[j]])
                    tmp_weight = tmp ** nval

                    i1_ind_weights.append(tmp_weight)

                sampled_i1 = choices(list(range(remain)), weights=i1_ind_weights)[0]
                sampled_i1_clique_config = (sampled_i1,) + sampled_i1_clique_config
                remain = remain - sampled_i1
        return sampled_i1_clique_config
    
    def sample_i2_clique(self, remain):
        sampled_i2_clique_config = ()
        bign = 0
        l = len(self.opt_cell_graph.i2_ind)
        for i in range(l):
            i2_n = self.opt_cell_graph.configs_for_2ind_sample[(l - i, bign)]
            i2_w = self.opt_cell_graph.weights_for_2ind_sample[(l - i, bign)]
            sampled_i2_idx = choices(list(range(len(i2_n))), weights=i2_w)[0]
            sampled_i2_n = i2_n[sampled_i2_idx]
            sampled_i2_clique_config = (sampled_i2_n,) + sampled_i2_clique_config
            remain = remain - sampled_i2_n
            bign = bign + sampled_i2_n
        return sampled_i2_clique_config, remain

    # def sample(self, k=1):
    #     samples = []
    #     sampled_idx_list = choices(list(range(len(self.noind_clique_configs))),
    #                                weights=self.noind_clique_weights, k=k)
        
    #     for sampled_idx in tqdm(sampled_idx_list):
            
    #         # TODO cache relation with nonind
            
    #         # sample noind clique config
    #         sampled_noind_clique_config = self.noind_clique_configs[sampled_idx]
    #         sampled_clique_config = sampled_noind_clique_config
    #         remain = self.domain_size - sum(sampled_clique_config)
            
    #         # sample i1 and i2 clique config
    #         self.opt_cell_graph.setup_term_cache()
    #         self.opt_cell_graph.get_term(len(self.opt_cell_graph.i2_ind), 0, sampled_noind_clique_config)
            
    #         sampled_i2_clique_config, remain = self.sample_i2_clique(remain)
    #         sampled_clique_config = sampled_i2_clique_config + sampled_clique_config

    #         sampled_i1_clique_config = self.sample_i1_clique(remain, sampled_noind_clique_config)
    #         sampled_clique_config = sampled_i1_clique_config + sampled_clique_config

    #         logger.debug('sampled clique config: %s', sampled_clique_config)
    #         logger.debug('sampled I1 clique config: %s', sampled_i1_clique_config)
    #         logger.debug('sampled I2 clique config: %s', sampled_i2_clique_config)
    #         logger.debug('sampled NI clique config: %s', sampled_noind_clique_config)

    #         # sample in clique
    #         sampled_config = ()
    #         for i, n in zip(self.opt_cell_graph.i1_ind, sampled_i1_clique_config):
    #             sampled_config = sampled_config + self.sample_in_clique(i, n)
    #         for i, n in zip(self.opt_cell_graph.i2_ind, sampled_i2_clique_config):
    #             sampled_config = sampled_config + self.sample_in_clique(i, n)
    #         for i, n in zip(self.opt_cell_graph.nonind, sampled_noind_clique_config):
    #             sampled_config = sampled_config + self.sample_in_clique(i, n)

    #         samples.append(self.sample_on_config(sampled_config))

    #     return samples


    def sample_on_config(self, config):
        logger.debug('sample on cell configuration %s', config)
        # shuffle domain elements
        random.shuffle(self.domain)
        with Timer() as t:
            cell_assignment, cell_weight = self._assign_cell(
                self.cell_graph, dict(
                    zip(self.cells, config))
            )
            sampled_atoms: set = self._remove_aux_atoms(
                self._get_unary_atoms(cell_assignment)
            )
            logger.debug('initial unary atoms: %s', sampled_atoms)
            self.t_assigning += t.elapsed

        pair_evidences = None
        if self.context.contain_existential_quantifier():
            pair_evidences = self._sample_ext_evidences(
                cell_assignment, cell_weight)
            logger.debug('sampled existential quantified literals: %s',
                         pair_evidences)

        with Timer() as t:
            sampled_atoms.update(
                self._sample_binary_atoms(
                    cell_assignment, cell_weight,
                    pair_evidences=pair_evidences,
                )
            )
            self.t_sampling_models += t.elapsed
        return self._remove_aux_atoms(sampled_atoms)

    def sample(self, k=1):
        samples = []
        print(self.configs)
        print(self.weights)
        sampled_idx = choices(
            list(range(len(self.configs))), weights=self.weights, k=k)

        self.t_assigning = 0
        self.t_sampling = 0
        self.t_sampling_models = 0
        # TODO: do it parallelly!
        for idx in tqdm(sampled_idx):
            samples.append(self.sample_on_config(
                self.configs[idx]
            ))
        logger.info('elapsed time for assigning cell type: %s',
                    self.t_assigning)
        logger.info('elapsed time for sampling possible worlds: %s',
                    self.t_sampling_models)
        return samples


def parse_args():
    parser = argparse.ArgumentParser(
        description='Sampler for MLN',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--input', '-i', type=str, required=True,
                        help='mln file')
    parser.add_argument('--n_samples', '-k', type=int, required=True)
    parser.add_argument('--output_dir', '-o', type=str,
                        default='./outputs')
    parser.add_argument('--show_samples', '-s',
                        action='store_true', default=False)
    parser.add_argument('--debug', '-d', action='store_true', default=False)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    if args.debug:
        logzero.loglevel(logging.DEBUG)
        args.show_samples = True
    else:
        logzero.loglevel(logging.INFO)
    logzero.logfile('{}/log.txt'.format(args.output_dir), mode='w')

    with Timer() as t:
        problem = parse_input(args.input)
    context = WFOMSContext(problem)
    logger.info('Parse input: %ss', t)

    with Timer() as total_t:
        with Timer() as t:
            sampler = Samplerv2(context)
        logger.info('elapsed time for initializing sampler: %s', t.elapsed)
        samples = sampler.sample(args.n_samples)
        logger.info('total time for sampling: %s', total_t.elapsed)
    save_file = os.path.join(args.output_dir, 'samples.pkl')
    with open(save_file, 'wb') as f:
        pickle.dump(samples, f)
    logger.info('Samples are saved in %s', save_file)
    if args.show_samples:
        logger.info('Samples:')
        for s in samples:
            logger.info(sorted(str(i) for i in s))
