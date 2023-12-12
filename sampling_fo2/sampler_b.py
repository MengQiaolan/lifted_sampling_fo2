from __future__ import annotations

import random
import numpy as np
import argparse
import os
import logzero
import logging
import pickle

from tqdm import tqdm
from logzero import logger
from contexttimer import Timer
from collections import defaultdict

from sampling_fo2.context import WFOMSContext, WFOMSContextv2
from sampling_fo2.context.existential_context import \
    BlockType, ExistentialContext, ExistentialTwoTable
from sampling_fo2.fol.syntax import AtomicFormula, Const, a, b, \
    AUXILIARY_PRED_NAME, PREDS_FOR_EXISTENTIAL
from sampling_fo2.utils import MultinomialCoefficients, multinomial, \
    Rational, RingElement, coeff_monomial, round_rational, expand
from sampling_fo2.parser import parse_input
from sampling_fo2.cell_graph import Cell, CellGraph
from sampling_fo2.utils.polynomial import choices

from sampling_fo2.wfomc import get_config_weight_standard_faster, \
    get_config_weight_standard


class Sampler(object):
    def __init__(self, context: WFOMSContextv2):
        self.context: WFOMSContextv2 = context
        get_weight = self.context.get_weight
        
        self.domain: list[Const] = list(self.context.domain)
        self.domain_size: int = len(self.domain)
        logger.debug('domain: %s', self.domain)
        self.cell_graph: CellGraph = CellGraph(
            self.context.skolemized_qf_formula, get_weight
        )
        MultinomialCoefficients.setup(self.domain_size)
        self.configs, self.weights = self._get_config_weights(
            self.cell_graph, self.domain_size)

        if self.context.contain_existential_quantifier():
            # Precomputed weights for cell configs
            self.uni_cell_graph: CellGraph = CellGraph(
                self.context.universal_qf_formula, get_weight
            )
            # self.uni_cell_graph.show()
            self.configs, self.weights = self._adjust_config_weights(
                self.configs, self.weights,
                self.cell_graph, self.uni_cell_graph
            )
            self.cell_graph = self.uni_cell_graph
        wfomc = sum(self.weights)
        if wfomc == 0:
            raise RuntimeError(
                'Unsatisfiable formula!!'
            )
        round_val = round_rational(wfomc)
        logger.info('wfomc (round):%s (exp(%s))',
                    round_val, round_val.ln())
        logger.debug('Configuration weight (round): %s', list(zip(self.configs, [
            round_rational(w) for w in self.weights
        ])))
        self.cells = self.cell_graph.get_cells()

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
            for n in range(1, self.domain_size):
                self.cb_weights.update(
                    self._precompute_cb_weight(
                        self.block_cell_graph, n, self.context)
                )
            logger.debug('pre-computed weights for existential quantifiers:\n%s',
                         self.cb_weights)

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
            for block_pred, block_type in context.blockpred_to_blocktype.items():
                if cell.is_positive(block_pred):
                    config.append((
                        cell.drop_preds(prefixes=PREDS_FOR_EXISTENTIAL),
                        block_type
                    ))
            cb_configs.append(config)

        cell_weights, edge_weights = cell_graph.get_all_weights()

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
        # remove duplications
        for cb_config in cb_weights.keys():
            dup_factor = Rational(MultinomialCoefficients.coef(
                tuple(c[2] for c in cb_config)
            ), 1)
            cb_weights[cb_config] /= dup_factor
        return cb_weights

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

    def _sample_ext_evidences(self, cell_assignment: list[Cell],
                              cell_weight: RingElement) \
            -> dict[tuple[int, int], frozenset[AtomicFormula]]:
        ext_config = ExistentialContext(
            cell_assignment, self.context.binary_ext_preds)

        # Get the total weight of the current configuration
        cell_config = tuple(cell_assignment.count(cell) for cell in self.cells)
        total_weight = self.weights[self.configs.index(
            cell_config)] / MultinomialCoefficients.coef(cell_config)

        pair_evidences: dict[tuple[int, int],
                             frozenset[AtomicFormula]] = dict()
        q = Rational(1, 1)
        while not ext_config.all_satisfied():
            selected_cell, selected_block = ext_config.select_cell_block_type()
            print(selected_block)
            # print(ext_config.cb_config.items())
            selected_idx = ext_config.reduce_element(
                selected_cell, selected_block)
            # print("-> ", ext_config.cb_config.items())
            # print()
            logger.debug('select element: %s, cell: %s, block type: %s',
                         selected_idx, selected_cell, selected_block)

            etable_weights: dict[
                Cell, dict[ExistentialTwoTable, RingElement]
            ] = dict()
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

            for etable_config in ext_config.iter_etable_config(
                etable_weights
            ):
                print("-"*100)
                for k,v in etable_config.items():
                    print(k)
                    for k2,v2 in v.items():
                        print(k2,v2)
                #     break
                cell_weight = self.cell_graph.get_cell_weight(selected_cell)
                etable_config_per_cell = defaultdict(
                    lambda: defaultdict(lambda: 0)
                )
                overall_etable_config = defaultdict(lambda: 0)
                for (cell, _), config in etable_config.items():
                    for etable, num in config.items():
                        etable_config_per_cell[cell][etable] += num
                        overall_etable_config[etable] += num

                if not ext_config.satisfied(selected_block, overall_etable_config):
                    continue
                print("@"*100)
                coeff = Rational(1, 1)
                for _, config in etable_config.items():
                    coeff *= Rational(MultinomialCoefficients.coef(
                        tuple(config.values())), 1)

                total_weight_etable = Rational(1, 1)
                for cell, config in etable_config_per_cell.items():
                    for etable, num in config.items():
                        total_weight_etable *= (
                            etable_weights[cell][etable] ** Rational(num, 1)
                        )

                reduced_cb_config = ext_config.reduce_cb_config(etable_config)
                # print()
                # print(reduced_cb_config)
                # print(ext_config.cb_config.items())
                reduced_weight = self.cb_weights[reduced_cb_config]
                # print(q, total_weight_ebtype, utype_weight, coeff,
                #       reduced_weight)
                # print(expand(q * total_weight_ebtype * utype_weight * coeff *
                #       reduced_weight))
                w = self.context.decode_result(
                    q * total_weight_etable * cell_weight *
                    coeff * reduced_weight
                )
                # print(ext_config.cb_config.items())
                # print("-"*100)
                # print(w / total_weight)
                # logger.debug(eb_config)
                # logger.debug('%s %s', w, total_weight)
                if random.random() < w / total_weight:
                    print("@"*100)
                    logger.debug('selected etable config:\n%s', etable_config)
                    etable_indices = ext_config.sample_and_update(
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
                    twotable_weight = self.cell_graph.get_two_table_weight(
                        cell_pair, frozenset(pair_evidences[pair])
                    )
                else:
                    twotable_weight = self.cell_graph.get_two_table_weight(
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
                twotables_with_weight = self.cell_graph.get_two_tables(
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

    def _assign_cell(self, cell_graph: CellGraph,
                     config: dict[Cell, int]) -> tuple[list[Cell], RingElement]:
        cell_assignment = list()
        w = Rational(1, 1)
        for cell, n in config.items():
            for _ in range(n):
                cell_assignment.append(cell)
                w = w * cell_graph.get_cell_weight(cell)
        return cell_assignment, w

    def sample(self, k=1):
        samples = []
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
    context = WFOMSContextv2(problem)
    logger.info('Parse input: %ss', t)

    with Timer() as total_t:
        with Timer() as t:
            sampler = Sampler(context)
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