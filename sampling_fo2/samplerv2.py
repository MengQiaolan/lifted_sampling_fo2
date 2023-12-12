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
    OOptimizedCellGraph
from sampling_fo2.utils.polynomial import choices

class Samplerv2(object):
    def __init__(self, context: WFOMSContext):
        self.context: WFOMSContext = context
        get_weight = self.context.get_weight

        self.domain: list[Const] = list(self.context.domain)
        self.domain_size: int = len(self.domain)
        logger.debug('domain: %s', self.domain)

        self.opt_cell_graph: OOptimizedCellGraph = OOptimizedCellGraph(
            self.context.formula, get_weight, self.domain_size
        )
        self.noind_clique_configs, self.noind_clique_weights = \
            self._get_noind_clique_config_weights(self.opt_cell_graph, self.domain_size)

        if self.context.contain_existential_quantifier():
            raise NotImplementedError
        
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
            raise NotImplementedError

        # for measuring performance
        self.t_sampling = 0
        self.t_assigning = 0
        self.t_sampling_models = 0

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
                for nval in range(remain+1):
                    tmp_weight = MultinomialCoefficients.comb(remain, nval)
                    tmp_weight *= self.opt_cell_graph.cache_for_1ind[i] ** (remain - nval)
                    tmp = self.opt_cell_graph.get_cell_weight(cliques[i][0])
                    for j in nonind:
                        tmp = tmp * self.opt_cell_graph.get_two_table_weight(
                            (cliques[i][0], cliques[j][0])
                        ) ** (sampled_noind_clique_config[nonind_map[j]])
                    tmp_weight = tmp_weight * tmp ** nval
                    i1_ind_weights.append(tmp_weight)

                sampled_i1 = choices(list(range(remain+1)), weights=i1_ind_weights)[0]
                sampled_i1_clique_config = (sampled_i1,) + sampled_i1_clique_config
                remain = remain - sampled_i1
        return sampled_i1_clique_config
    
    def sample_i2_clique(self, remain):
        sampled_i2_clique_config = ()
        bign = 0 # sampled number
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

    def sample(self, k=1):
        samples = []
        sampled_idx_list = choices(list(range(len(self.noind_clique_configs))),
                                   weights=self.noind_clique_weights, k=k)
        
        for sampled_idx in tqdm(sampled_idx_list):
            
            # TODO cache relation with nonind
            
            # sample noind clique config
            sampled_noind_clique_config = self.noind_clique_configs[sampled_idx]
            sampled_clique_config = sampled_noind_clique_config
            remain = self.domain_size - sum(sampled_clique_config)
            
            # sample i1 and i2 clique config
            self.opt_cell_graph.reset_term_cache()
            self.opt_cell_graph.get_g_term(len(self.opt_cell_graph.i2_ind), 0, sampled_noind_clique_config)
            
            sampled_i2_clique_config, remain = self.sample_i2_clique(remain)
            sampled_clique_config = sampled_i2_clique_config + sampled_clique_config

            sampled_i1_clique_config = self.sample_i1_clique(remain, sampled_noind_clique_config)
            sampled_clique_config = sampled_i1_clique_config + sampled_clique_config

            logger.debug('sampled clique config: %s', sampled_clique_config)
            logger.debug('sampled I1 clique config: %s', sampled_i1_clique_config)
            logger.debug('sampled I2 clique config: %s', sampled_i2_clique_config)
            logger.debug('sampled NI clique config: %s', sampled_noind_clique_config)

            # sample in clique
            sampled_config = ()
            for i, n in zip(self.opt_cell_graph.i1_ind, sampled_i1_clique_config):
                sampled_config = sampled_config + self.sample_in_clique(i, n)
            for i, n in zip(self.opt_cell_graph.i2_ind, sampled_i2_clique_config):
                sampled_config = sampled_config + self.sample_in_clique(i, n)
            for i, n in zip(self.opt_cell_graph.nonind, sampled_noind_clique_config):
                sampled_config = sampled_config + self.sample_in_clique(i, n)

            samples.append(self.sample_on_config(sampled_config))

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
