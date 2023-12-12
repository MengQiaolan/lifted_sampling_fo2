from __future__ import annotations

import pandas as pd
import functools
import networkx as nx

from typing import Callable, Dict, FrozenSet, List, Tuple
from logzero import logger
from sympy import Poly
from copy import deepcopy
from sampling_fo2.cell_graph.utils import conditional_on

from sampling_fo2.fol.syntax import AtomicFormula, Const, Pred, QFFormula, a, b, c, X, Y
from sampling_fo2.utils import Rational, RingElement
from sampling_fo2.utils.multinomial import MultinomialCoefficients, multinomial
from sampling_fo2.utils.polynomial import choices, create_vars, expand, coeff_dict

from .components import Cell, TwoTable

def ground_on_tuple(formula: QFFormula,
                        c1: Const, c2: Const = None) -> QFFormula:
    variables = formula.vars()
    if len(variables) > 2:
        raise RuntimeError(
            "Can only ground out FO2"
        )
    if len(variables) == 1:
        constants = [c1]
    else:
        if c2 is not None:
            constants = [c1, c2]
        else:
            constants = [c1, c1]
    substitution = dict(zip(variables, constants))
    gnd_formula = formula.substitute(substitution)
    return gnd_formula

def ground_on_tuple_v2(formula: QFFormula,
                        c1: Const, c2: Const = None) -> QFFormula:
    variables = formula.vars()
    if len(variables) > 2:
        raise RuntimeError(
            "Can only ground out FO2"
        )
    if len(variables) == 1:
        constants = [c1]
    else:
        if c2 is not None:
            constants = [c1, c2]
        else:
            constants = [c1, c1]
    substitution = dict(zip(variables, constants))
    
    substitution = {X: a, Y: b}
    
    gnd_formula = formula.substitute(substitution)
    return gnd_formula

def build_cells_from_formula(formula):
    preds = tuple(formula.preds())
    gnd_formula_cc = ground_on_tuple(formula, c)
    cells = []
    code = {}
    for model in gnd_formula_cc.models():
        for lit in model:
            code[lit.pred] = lit.positive
        cells.append(Cell(tuple(code[p] for p in preds), preds))
    return cells

class CellGraph(object):
    """
    Cell graph that handles cells (i.e., 1-types, in the sampling paper) and the wmc between them.
    """

    def __init__(self, formula: QFFormula,
                 get_weight: Callable[[Pred], Tuple[RingElement, RingElement]]):
        """
        Cell graph that handles cells (1-types) and the WMC between them

        :param sentence CNF: the sentence in the form of CNF
        :param get_weight Callable[[Pred], Tuple[mpq, mpq]]: the weighting function
        :param conditional_formulas List[CNF]: the optional conditional formula appended in WMC computing
        """
        self.formula: QFFormula = formula
        self.get_weight: Callable[[Pred],
                                  Tuple[RingElement, RingElement]] = get_weight
        self.preds: Tuple[Pred] = tuple(self.formula.preds())
        logger.debug('prednames: %s', self.preds)

        gnd_formula_ab1: QFFormula = ground_on_tuple(
            self.formula, a, b
        )
        gnd_formula_ab2: QFFormula = ground_on_tuple(
            self.formula, b, a
        )
        self.gnd_formula_ab: QFFormula = \
            gnd_formula_ab1 & gnd_formula_ab2
        self.gnd_formula_cc: QFFormula = ground_on_tuple(
            self.formula, c
        )
        logger.info('ground a b: %s', self.gnd_formula_ab)
        logger.info('ground c: %s', self.gnd_formula_cc)

        # build cells
        self.cells: List[Cell] = self.build_cells()
        # filter cells
        logger.info('the number of valid cells: %s',
                    len(self.cells))

        self.cell_weights: Dict[Cell, Poly] = self._compute_cell_weights()
        self.two_tables: Dict[Tuple[Cell, Cell],
                              TwoTable] = self._build_two_tables()

    def show(self):
        logger.info(str(self))

    def build_cells(self):
        return build_cells_from_formula(self.formula)

    def __str__(self):
        s = 'CellGraph:\n'
        s += 'predicates: {}\n'.format(self.preds)
        cell_weight_df = []
        twotable_weight_df = []
        for _, cell1 in enumerate(self.cells):
            cell_weight_df.append(
                [str(cell1), self.get_cell_weight(cell1)]
            )
            twotable_weight = []
            for _, cell2 in enumerate(self.cells):
                # if idx1 < idx2:
                #     twotable_weight.append(0)
                #     continue
                twotable_weight.append(
                    self.get_two_table_weight(
                        (cell1, cell2))
                )
            twotable_weight_df.append(twotable_weight)
        cell_str = [str(cell) for cell in self.cells]
        cell_weight_df = pd.DataFrame(cell_weight_df, index=None,
                                      columns=['Cell', 'Weight'])
        twotable_weight_df = pd.DataFrame(twotable_weight_df, index=cell_str,
                                          columns=cell_str)
        s += 'cell weights: \n'
        s += cell_weight_df.to_markdown() + '\n'
        s += '2table weights: \n'
        s += twotable_weight_df.to_markdown()
        return s

    def __repr__(self):
        return str(self)

    def get_cells(self, cell_filter: Callable[[Cell], bool] = None) -> List[Cell]:
        if cell_filter is None:
            return self.cells
        return list(filter(cell_filter, self.cells))

    @functools.lru_cache(maxsize=None, typed=True)
    def get_cell_weight(self, cell: Cell) -> Poly:
        if cell not in self.cell_weights:
            logger.warning(
                "Cell %s not found", cell
            )
            return 0
        return self.cell_weights.get(cell)

    def _check_existence(self, cells: Tuple[Cell, Cell]):
        if cells not in self.two_tables:
            raise ValueError(
                f"Cells {cells} not found, note that the order of cells matters!"
            )

    @functools.lru_cache(maxsize=None, typed=True)
    def get_two_table_weight(self, cells: Tuple[Cell, Cell],
                             evidences: FrozenSet[AtomicFormula] = None) -> RingElement:
        self._check_existence(cells)
        return self.two_tables.get(cells).get_weight(evidences)

    def get_all_weights(self) -> Tuple[List[RingElement], List[RingElement]]:
        cell_weights = []
        twotable_weights = []
        for i, cell_i in enumerate(self.cells):
            cell_weights.append(self.get_cell_weight(cell_i))
            twotable_weight = []
            for j, cell_j in enumerate(self.cells):
                if i > j:
                    twotable_weight.append(Rational(1, 1))
                else:
                    twotable_weight.append(self.get_two_table_weight(
                        (cell_i, cell_j)
                    ))
            twotable_weights.append(twotable_weight)
        return cell_weights, twotable_weights

    @functools.lru_cache(maxsize=None, typed=True)
    def satisfiable(self, cells: Tuple[Cell, Cell],
                    evidences: FrozenSet[AtomicFormula] = None) -> bool:
        self._check_existence(cells)
        return self.two_tables.get(cells).satisfiable(evidences)

    @functools.lru_cache(maxsize=None)
    def get_two_tables(self, cells: Tuple[Cell, Cell],
                       evidences: FrozenSet[AtomicFormula] = None) \
            -> Tuple[FrozenSet[AtomicFormula], RingElement]:
        self._check_existence(cells)
        return self.two_tables.get(cells).get_two_tables(evidences)

    def _compute_cell_weights(self):
        weights = dict()
        for cell in self.cells:
            weight = Rational(1, 1)
            for i, pred in zip(cell.code, cell.preds):
                if pred.arity > 0:
                    if i:
                        weight = weight * self.get_weight(pred)[0]
                    else:
                        weight = weight * self.get_weight(pred)[1]
            weights[cell] = weight
        return weights

    @functools.lru_cache(maxsize=None)
    def get_nullary_weight(self, cell: Cell) -> RingElement:
        weight = Rational(1, 1)
        for i, pred in zip(cell.code, cell.preds):
            if pred.arity == 0:
                if i:
                    weight = weight * self.get_weight(pred)[0]
                else:
                    weight = weight * self.get_weight(pred)[1]
        return weight

    def _build_two_tables(self):
        # build a pd.DataFrame containing all model as well as the weight
        models = dict()
        gnd_lits = self.gnd_formula_ab.atoms()
        gnd_lits = gnd_lits.union(
            frozenset(map(lambda x: ~x, gnd_lits))
        )
        for model in self.gnd_formula_ab.models():
            weight = Rational(1, 1)
            for lit in model:
                # ignore the weight appearing in cell weight
                if (not (len(lit.args) == 1 or all(arg == lit.args[0]
                                                   for arg in lit.args))):
                    weight *= (self.get_weight(lit.pred)[0] if lit.positive else
                               self.get_weight(lit.pred)[1])
            models[frozenset(model)] = weight
        # build twotable tables
        tables = dict()
        for i, cell in enumerate(self.cells):
            models_1 = conditional_on(models, gnd_lits, cell.get_evidences(a))
            for j, other_cell in enumerate(self.cells):
                if i > j:
                    tables[(cell, other_cell)] = tables[(other_cell, cell)]
                models_2 = conditional_on(models_1, gnd_lits,
                                          other_cell.get_evidences(b))
                tables[(cell, other_cell)] = TwoTable(
                    models_2, gnd_lits
                )
        return tables

class CellGraphv2(CellGraph):
    def __init__(self, formula: QFFormula,
                 get_weight: Callable[[Pred], Tuple[RingElement, RingElement]]):
        super().__init__(formula, get_weight)
    
    def get_all_weights_v2(self) -> Tuple[List[RingElement], List[RingElement]]:
        cell_weights = []
        twotable_weights = []
        for i, cell_i in enumerate(self.cells):
            cell_weights.append(self.get_cell_weight(cell_i))
            twotable_weight = []
            for j, cell_j in enumerate(self.cells):
                twotable_weight.append(self.get_two_table_weight(
                    (cell_i, cell_j)
                ))
            twotable_weights.append(twotable_weight)
        return cell_weights, twotable_weights

    
    def _build_two_tables(self):
        # build a pd.DataFrame containing all model as well as the weight
        models = dict()
        gnd_lits = self.gnd_formula_ab.atoms()
        
        
        new_gnd_formula_ab = ground_on_tuple_v2(self.formula, a, b)
        gnd_lits = new_gnd_formula_ab.atoms()
        
        
        gnd_lits = gnd_lits.union(
            frozenset(map(lambda x: ~x, gnd_lits))
        )
        #for model in self.gnd_formula_ab.models():
        for model in new_gnd_formula_ab.models():
            weight = Rational(1, 1)
            for lit in model:
                # ignore the weight appearing in cell weight
                if (not (len(lit.args) == 1 or all(arg == lit.args[0]
                                                   for arg in lit.args))):
                    # weight *= (self.get_weight(lit.pred)[0] if lit.positive else
                    #            self.get_weight(lit.pred)[1])
                    if lit.positive:
                        weight *= self.get_weight(lit.pred)[0]
                    else:
                        from symengine import var
                        if lit.args[0] == a and lit.pred.name == "R1":
                            weight *= var('a1')
                        elif lit.args[0] == a and lit.pred.name == "R2":
                            weight *= var('a2')
                        else:
                            weight *= self.get_weight(lit.pred)[1]
            models[frozenset(model)] = weight
        # for k,v in models.items():
        #     print(k, v)
        # build twotable tables
        tables = dict()
        for i, cell in enumerate(self.cells):
            models_1 = conditional_on(models, gnd_lits, cell.get_evidences(a))
            for j, other_cell in enumerate(self.cells):
                if i > j:
                    tables[(cell, other_cell)] = tables[(other_cell, cell)]
                models_2 = conditional_on(models_1, gnd_lits,
                                          other_cell.get_evidences(b))
                tables[(cell, other_cell)] = TwoTable(
                    models_2, gnd_lits
                )
        # for k,v in tables.items():
        #     print(k, v.get_two_tables())
        return tables

    

class OptimizedCellGraph(CellGraph):
    def __init__(self, formula: QFFormula,
                 get_weight: Callable[[Pred], Tuple[RingElement, RingElement]],
                 domain_size: int,
                 modified_cell_symmetry: bool = False):
        """
        Optimized cell graph for FastWFOMC
        :param formula: the formula to be grounded
        :param get_weight: a function that returns the weight of a predicate
        :param domain_size: the domain size
        """
        super().__init__(formula, get_weight)
        self.modified_cell_symmetry = modified_cell_symmetry
        self.domain_size: int = domain_size
        MultinomialCoefficients.setup(self.domain_size)

        if self.modified_cell_symmetry:
            # 1. find independent sets
            i1_ind_set, i2_ind_set, nonind_set = self.find_independent_sets()
            # 2. build symmetric cliques
            self.cliques, [self.i1_ind, self.i2_ind, self.nonind] = \
                self.build_symmetric_cliques([i1_ind_set, i2_ind_set, nonind_set])
            self.nonind_map: dict[int, int] = dict(zip(self.nonind, range(len(self.nonind))))
        else:
            raise NotImplementedError

        logger.info("Found i1 independent cliques: %s", self.i1_ind)
        logger.info("Found i2 independent cliques: %s", self.i2_ind)
        logger.info("Found non-independent cliques: %s", self.nonind)

        self.term_cache = dict()

    def build_symmetric_cliques(self, cell_indices_list) -> List[List[Cell]]:
        i1_ind_set = deepcopy(cell_indices_list[0])
        cliques: list[list[Cell]] = []
        ind_idx: list[list[int]] = []
        for cell_indices in cell_indices_list:
            idx_list = []
            while len(cell_indices) > 0:
                cell_idx = cell_indices.pop()
                clique = [self.cells[cell_idx]]
                # for cell in I1 independent set, we dont need to built sysmmetric cliques
                if cell_idx not in i1_ind_set:
                    for other_cell_idx in cell_indices:
                        other_cell = self.cells[other_cell_idx]
                        if self._matches(clique, other_cell):
                            clique.append(other_cell)
                    for other_cell in clique[1:]:
                        cell_indices.remove(self.cells.index(other_cell))
                cliques.append(clique)
                idx_list.append(len(cliques) - 1)
            ind_idx.append(idx_list)
        logger.info("Built %s symmetric cliques: %s", len(cliques), cliques)
        return cliques, ind_idx

    def find_independent_sets(self) -> tuple[list[int], list[int], list[int], list[int]]:
        g = nx.Graph()
        g.add_nodes_from(range(len(self.cells)))
        for i in range(len(self.cells)):
            for j in range(i + 1, len(self.cells)):
                if self.get_two_table_weight(
                        (self.cells[i], self.cells[j])
                ) != Rational(1, 1):
                    g.add_edge(i, j)

        self_loop = set()
        for i in range(len(self.cells)):
            if self.get_two_table_weight((self.cells[i], self.cells[i])) != Rational(1, 1):
                self_loop.add(i)

        if len(g.nodes-self_loop) == 0:
            i1_ind = {}
        else:
            i1_ind = set(nx.maximal_independent_set(g.subgraph(g.nodes-self_loop)))
        g_ind = set(nx.maximal_independent_set(g, nodes=i1_ind))
        i2_ind = g_ind.difference(i1_ind)
        non_ind = g.nodes - i1_ind - i2_ind
        logger.info("Found i1 independent set: %s", i1_ind)
        logger.info("Found i2 independent set: %s", i2_ind)
        logger.info("Found non-independent set: %s", non_ind)
        return list(i1_ind), list(i2_ind), list(non_ind)

    def _matches(self, clique, other_cell) -> bool:
        cell = clique[0]
        if not self.modified_cell_symmetry:
            if self.get_cell_weight(cell) != self.get_cell_weight(other_cell) or \
                    self.get_two_table_weight((cell, cell)) != self.get_two_table_weight((other_cell, other_cell)):
                return False

        if len(clique) > 1:
            third_cell = clique[1]
            r = self.get_two_table_weight((cell, third_cell))
            for third_cell in clique:
                if r != self.get_two_table_weight((other_cell, third_cell)):
                    return False

        for third_cell in self.get_cells():
            if other_cell == third_cell or third_cell in clique:
                continue
            r = self.get_two_table_weight((cell, third_cell))
            if r != self.get_two_table_weight((other_cell, third_cell)):
                return False
        return True

    def reset_term_cache(self):
        self.term_cache = dict()

    @functools.lru_cache(maxsize=None)
    def get_g_term(self, iv: int, bign: int, partition: tuple[int]) -> RingElement:
        """
        param iv: the index of the current clique in I2
        param bign: the number of elements that has been assigned to (iv+1,...,k+l)-th clique
        param partition: the partition of cliques in non-independent set
        """
        if (iv, bign) in self.term_cache:
            return self.term_cache[(iv, bign)]

        if iv == 0:
            accum = Rational(0, 1)
            for j in self.i1_ind:
                tmp = self.get_cell_weight(self.cliques[j][0])
                for i in self.nonind:
                    tmp = tmp * self.get_two_table_weight(
                        (self.cliques[i][0], self.cliques[j][0])) ** partition[self.nonind_map[i]]
                accum = accum + tmp
            accum = accum ** (self.domain_size - sum(partition) - bign)
            self.term_cache[(iv, bign)] = accum
            return accum
        else:
            sumtoadd = 0
            s = self.i2_ind[len(self.i2_ind) - iv]
            for nval in range(self.domain_size - sum(partition) - bign + 1):
                smul = MultinomialCoefficients.comb(
                    self.domain_size - sum(partition) - bign, nval
                )
                smul = smul * self.get_J_term(s, nval)
                if not self.modified_cell_symmetry:
                    smul = smul * self.get_cell_weight(self.cliques[s][0]) ** nval

                for i in self.nonind:
                    smul = smul * self.get_two_table_weight(
                        (self.cliques[i][0], self.cliques[s][0])
                    ) ** (partition[self.nonind_map[i]] * nval)
                smul = smul * self.get_g_term(
                    iv - 1, bign + nval, partition
                )
                sumtoadd = sumtoadd + smul
            self.term_cache[(iv, bign)] = sumtoadd
            return sumtoadd

    @functools.lru_cache(maxsize=None)
    def get_J_term(self, l: int, nhat: int) -> RingElement:
        """
        param l: the index of the clique
        param nhat: the number of elements that are in the clique (configuration of the clique)
        return: the weight of the clique
        """
        if len(self.cliques[l]) == 1:
            thesum = self.get_two_table_weight(
                (self.cliques[l][0], self.cliques[l][0])
            ) ** (int(nhat * (nhat - 1) / 2))
            if self.modified_cell_symmetry:
                thesum = thesum * self.get_cell_weight(self.cliques[l][0]) ** nhat
        else:
            r = self.get_two_table_weight(
                (self.cliques[l][0], self.cliques[l][1]))
            thesum = (
                (r ** MultinomialCoefficients.comb(nhat, 2)) *
                self.get_d_term(l, nhat)
            )
        return thesum

    @functools.lru_cache(maxsize=None)
    def get_d_term(self, l: int, n: int, cur: int = 0) -> RingElement:
        clique_size = len(self.cliques[l])
        r = self.get_two_table_weight((self.cliques[l][0], self.cliques[l][1]))
        s = self.get_two_table_weight((self.cliques[l][0], self.cliques[l][0]))
        if cur == clique_size - 1:
            if self.modified_cell_symmetry:
                w = self.get_cell_weight(self.cliques[l][cur]) ** n
                s = self.get_two_table_weight((self.cliques[l][cur], self.cliques[l][cur]))
                ret = w * (s / r) ** MultinomialCoefficients.comb(n, 2)
            else:
                ret = (s / r) ** MultinomialCoefficients.comb(n, 2)
        else:
            ret = 0
            for ni in range(n + 1):
                mult = MultinomialCoefficients.comb(n, ni)
                if self.modified_cell_symmetry:
                    w = self.get_cell_weight(self.cliques[l][cur]) ** ni
                    s = self.get_two_table_weight((self.cliques[l][cur], self.cliques[l][cur]))
                    mult = mult * w
                mult = mult * ((s / r) ** MultinomialCoefficients.comb(ni, 2))
                mult = mult * self.get_d_term(l, n - ni, cur + 1)
                ret = ret + mult
        return ret
    

class OptimizedCellGraphv2(CellGraph):
    def __init__(self, formula: QFFormula,
                 get_weight: Callable[[Pred], Tuple[RingElement, RingElement]]):
        """
        Optimized cell graph for FastWFOMC
        :param formula: the formula to be grounded
        :param get_weight: a function that returns the weight of a predicate
        :param domain_size: the domain size
        """
        super().__init__(formula, get_weight)

        # 1. find independent sets
        i1_ind_set, i2_ind_set, nonind_set = self.find_independent_sets()
        
        # 2. build symmetric cliques
        self.cliques, [self.i1_ind, self.i2_ind, self.nonind] = \
            self.build_symmetric_cliques([i1_ind_set, i2_ind_set, nonind_set])
        self.nonind_map: dict[int, int] = dict(zip(self.nonind, range(len(self.nonind))))

        logger.info("Found i1 independent cliques: %s", self.i1_ind)
        logger.info("Found i2 independent cliques: %s", self.i2_ind)
        logger.info("Found non-independent cliques: %s", self.nonind)
        
        # for c in self.cliques:
        #     print("-----")
        #     for b in c:
        #         print(b)
        
        self.g_term_cache = dict()
        self.d_term_cache = dict()

    def build_symmetric_cliques(self, cell_indices_list) -> List[List[Cell]]:
        i1_ind_set = deepcopy(cell_indices_list[0])
        cliques: list[list[Cell]] = []
        ind_idx: list[list[int]] = []
        for cell_indices in cell_indices_list:
            idx_list = []
            while len(cell_indices) > 0:
                cell_idx = cell_indices.pop()
                clique = [self.cells[cell_idx]]
                # for cell in I1 independent set, we dont need to built sysmmetric cliques
                if cell_idx not in i1_ind_set:
                    for other_cell_idx in cell_indices:
                        other_cell = self.cells[other_cell_idx]
                        if self._matches(clique, other_cell):
                            clique.append(other_cell)
                    for other_cell in clique[1:]:
                        cell_indices.remove(self.cells.index(other_cell))
                cliques.append(clique)
                idx_list.append(len(cliques) - 1)
            ind_idx.append(idx_list)
        logger.info("Built %s symmetric cliques: %s", len(cliques), cliques)
        return cliques, ind_idx

    def find_independent_sets(self) -> tuple[list[int], list[int], list[int], list[int]]:
        g = nx.Graph()
        g.add_nodes_from(range(len(self.cells)))
        for i in range(len(self.cells)):
            for j in range(i + 1, len(self.cells)):
                if self.get_two_table_weight(
                        (self.cells[i], self.cells[j])
                ) != Rational(1, 1):
                    g.add_edge(i, j)

        self_loop = set()
        for i in range(len(self.cells)):
            if self.get_two_table_weight((self.cells[i], self.cells[i])) != Rational(1, 1):
                self_loop.add(i)

        if len(g.nodes-self_loop) == 0:
            i1_ind = {}
        else:
            i1_ind = set(nx.maximal_independent_set(g.subgraph(g.nodes-self_loop)))
        g_ind = set(nx.maximal_independent_set(g, nodes=i1_ind))
        # i2_ind = g_ind.difference(i1_ind)
        i2_ind = set()
        non_ind = g.nodes - i1_ind - i2_ind
        logger.info("Found i1 independent set: %s", i1_ind)
        logger.info("Found i2 independent set: %s", i2_ind)
        logger.info("Found non-independent set: %s", non_ind)
        return list(i1_ind), list(i2_ind), list(non_ind)

    def _matches(self, clique, other_cell) -> bool:
        cell = clique[0]
        if len(clique) > 1:
            third_cell = clique[1]
            r = self.get_two_table_weight((cell, third_cell))
            for third_cell in clique:
                if r != self.get_two_table_weight((other_cell, third_cell)):
                    return False

        for third_cell in self.get_cells():
            if other_cell == third_cell or third_cell in clique:
                continue
            r = self.get_two_table_weight((cell, third_cell))
            if r != self.get_two_table_weight((other_cell, third_cell)):
                return False
        return True

    def reset_term_cache(self):
        self.g_term_cache = dict()
        self.d_term_cache = dict()

    @functools.lru_cache(maxsize=None)
    def get_g_term(self, iv: int, rest: int, bign: int, partition: tuple[int]) -> RingElement:
        """
        param iv: the index of the current clique in I2
        param bign: the number of elements that has been assigned to (iv+1,...,k+l)-th clique
        param partition: the partition of cliques in non-independent set
        """
        # if (iv, bign) in self.g_term_cache:
        #     return self.g_term_cache[(iv, bign)]

        if iv == 0:
            accum = Rational(0, 1)
            for j in self.i1_ind:
                tmp = self.get_cell_weight(self.cliques[j][0])
                for i in self.nonind:
                    tmp = tmp * self.get_two_table_weight(
                        (self.cliques[i][0], self.cliques[j][0])) ** partition[self.nonind_map[i]]
                accum = accum + tmp
            accum = accum ** rest
            # self.g_term_cache[(iv, bign)] = accum
            return accum
        else:
            sumtoadd = 0
            s = self.i2_ind[len(self.i2_ind) - iv]
            for nval in range(rest + 1):
                smul = MultinomialCoefficients.comb(
                    rest, nval
                )
                smul = smul * self.get_J_term(s, nval)

                for i in self.nonind:
                    smul = smul * self.get_two_table_weight(
                        (self.cliques[i][0], self.cliques[s][0])
                    ) ** (partition[self.nonind_map[i]] * nval)
                smul = smul * self.get_g_term(
                    iv - 1, rest - nval, bign + nval, partition
                )
                sumtoadd = sumtoadd + smul
            # self.g_term_cache[(iv, bign)] = sumtoadd
            return expand(sumtoadd)

    @functools.lru_cache(maxsize=None)
    def get_J_term(self, l: int, nhat: int) -> RingElement:
        """
        param l: the index of the clique
        param nhat: the number of elements that are in the clique (configuration of the clique)
        """
        if len(self.cliques[l]) == 1:
            thesum = self.get_two_table_weight(
                (self.cliques[l][0], self.cliques[l][0])
            ) ** (int(nhat * (nhat - 1) / 2))
            thesum = thesum * self.get_cell_weight(self.cliques[l][0]) ** nhat
        else:
            r = self.get_two_table_weight(
                (self.cliques[l][0], self.cliques[l][1]))
            thesum = (
                (r ** MultinomialCoefficients.comb(nhat, 2)) *
                self.get_d_term(l, nhat)
            )
        return expand(thesum)

    @functools.lru_cache(maxsize=None)
    def get_d_term(self, l: int, n: int, cur: int = 0) -> RingElement:
        clique_size = len(self.cliques[l])
        r = self.get_two_table_weight((self.cliques[l][0], self.cliques[l][1]))
        s = self.get_two_table_weight((self.cliques[l][0], self.cliques[l][0]))
        if cur == clique_size - 1:
            w = self.get_cell_weight(self.cliques[l][cur]) ** n
            s = self.get_two_table_weight((self.cliques[l][cur], self.cliques[l][cur]))
            ret = w * (s / r) ** MultinomialCoefficients.comb(n, 2)
        else:
            ret = 0
            for ni in range(n + 1):
                coef = MultinomialCoefficients.comb(n, ni)
                w = self.get_cell_weight(self.cliques[l][cur])
                s = self.get_two_table_weight((self.cliques[l][cur], self.cliques[l][cur]))
                mult = coef * (w ** ni) * ((s / r) ** MultinomialCoefficients.comb(ni, 2))
                mult = mult * self.get_d_term(l, n - ni, cur + 1)
                ret = ret + mult
        return expand(ret)

class OptimizedCellGraphv2_forSymCell(OptimizedCellGraphv2):
    def __init__(self, formula: QFFormula,
                 original_cells: list[Cell],
                 get_weight: Callable[[Pred], Tuple[RingElement, RingElement]]):
        self.original_cells = original_cells
        super().__init__(formula, get_weight)
        
    @functools.lru_cache(maxsize=None, typed=True)
    def get_cell_weight(self, cell: Cell) -> Poly:
        if cell not in self.cell_weights:
            raise KeyError
        return self.cell_weights.get(cell)

    def _compute_cell_weights(self):
        self.origcell2sym = {}
        syms = create_vars('y0:{}'.format(len(self.original_cells)))
        for sym, origcell in zip(syms, self.original_cells):
            if origcell in self.origcell2sym:
                continue
            self.origcell2sym[origcell] = sym
        
        weights = dict()
        for cell in self.cells:
            for orig_cell in self.original_cells:
                # if orig_cell is subset of cell
                if all([cell.is_positive(p) == orig_cell.is_positive(p) \
                        for p in orig_cell.preds]):   
                    weight = Rational(1, 1)         
                    for t, pred in zip(cell.code, cell.preds):
                        if pred in orig_cell.preds:
                            continue
                        if pred.arity > 0:
                            if t:
                                weight = weight * self.get_weight(pred)[0]
                            else:
                                weight = weight * self.get_weight(pred)[1]
                    weights[cell] = weight * self.origcell2sym[orig_cell]
                    break
        return weights
    
    def get_count_dist(self, res: RingElement):
        symbols = [self.origcell2sym[cell] for cell in self.original_cells]
        count_dist = {}
        print("******************")
        res = expand(res)
        print("----------------")
        for degrees, coef in coeff_dict(res, symbols):
            count_dist[degrees] = coef
        if (0,)*len(self.original_cells) in count_dist:
            del count_dist[(0,)*len(self.original_cells)]
        return count_dist
        

class OOptimizedCellGraph(OptimizedCellGraph):
    def __init__(self, formula: QFFormula,
                 get_weight: Callable[[Pred], Tuple[RingElement, RingElement]],
                 domain_size: int,
                 modified_cell_symmetry: bool = True):
        
        super().__init__(formula, get_weight, domain_size, modified_cell_symmetry)
        self.d_term_cache = dict()
        for c in self.cells:
            print("cell: ", c)
        for c in self.cliques:
            print("clique: ", c)
        print("I1: ", self.i1_ind)
        print("I2: ", self.i2_ind)
        print("NI: ", self.nonind)
        
        self.resort_cliques()
        self.resort_cells()
        
        print("---------RESORT:----------")
        for c in self.cells:
            print("cell: ", c)
        for c in self.cliques:
            print("clique: ", c)
        print("I1: ", self.i1_ind)
        print("I2: ", self.i2_ind)
        print("NI: ", self.nonind)
        
    def resort_cliques(self):
        new_cliques = []
        new_i1_ind = []
        new_i2_ind = []
        new_nonind = []
        for i in self.i1_ind:
            new_cliques.append(self.cliques[i])
            new_i1_ind.append(len(new_cliques) - 1)
        for i in self.i2_ind:
            new_cliques.append(self.cliques[i])
            new_i2_ind.append(len(new_cliques) - 1)
        for i in self.nonind:
            new_cliques.append(self.cliques[i])
            new_nonind.append(len(new_cliques) - 1)
        self.cliques = new_cliques
        self.i1_ind = new_i1_ind
        self.i2_ind = new_i2_ind
        self.nonind = new_nonind
        self.nonind_map = dict(
            zip(self.nonind, range(len(self.nonind))))
        
    def resort_cells(self):
        new_cells = []
        for clique in self.cliques:
            for cell in clique:
                new_cells.append(cell)
        self.cells = new_cells
    
    def reset_term_cache(self):
        # different partitions have different caches
        self.term_cache = dict()
        self.configs_for_2ind_sample: dict[tuple, list] = dict()
        self.weights_for_2ind_sample: dict[tuple, list]  = dict()
        self.cache_for_1ind = dict()
    
    def get_g_term(self, p: int, bign: int, partition: tuple[int]) -> RingElement:
        if (p, bign) in self.term_cache:
            return self.term_cache[(p, bign)]

        if p == 0:
            accum = Rational(0, 1)
            for i in self.i1_ind:
                self.cache_for_1ind[i] = accum
                tmp = self.get_cell_weight(self.cliques[i][0])
                for j in self.nonind:
                    tmp = tmp * self.get_two_table_weight(
                        (self.cliques[i][0], self.cliques[j][0])) ** partition[self.nonind_map[j]]
                accum = accum + tmp
            accum = accum ** (self.domain_size - sum(partition) - bign)
            self.term_cache[(p, bign)] = accum
            return accum
        else:
            sumtoadd = 0
            self.configs_for_2ind_sample[(p, bign)] = []
            self.weights_for_2ind_sample[(p, bign)] = []
            for nval in range(self.domain_size - sum(partition) - bign + 1):
                smul = MultinomialCoefficients.comb(
                    self.domain_size - sum(partition) - bign, nval
                )
                smul = smul * self.get_J_term(self.i2_ind[p-1], nval)

                for i in self.nonind:
                    smul = smul * self.get_two_table_weight(
                        (self.cliques[i][0], 
                         self.cliques[self.i2_ind[p-1]][0])
                    ) ** (partition[self.nonind_map[i]] * nval)
                smul = smul * self.get_g_term(
                    p - 1, bign + nval, partition
                )
                
                sumtoadd = sumtoadd + smul
                self.configs_for_2ind_sample[(p, bign)].append(nval)
                self.weights_for_2ind_sample[(p, bign)].append(smul)
            self.term_cache[(p, bign)] = sumtoadd
            return sumtoadd
        
    @functools.lru_cache(maxsize=None)
    def get_d_term(self, l: int, n: int, cur: int = 0) -> RingElement:
        clique_size = len(self.cliques[l])
        r = self.get_two_table_weight((self.cliques[l][0], self.cliques[l][1]))
        s = self.get_two_table_weight((self.cliques[l][0], self.cliques[l][0]))
        if cur == clique_size - 1:
            ret = (s / r) ** MultinomialCoefficients.comb(n, 2)
            if self.modified_cell_symmetry:
                ret = ret * self.get_cell_weight(self.cliques[l][cur-1]) ** n
        else:
            ret = 0
            self.d_term_cache[(l, n, cur)] = []
            for ni in range(n + 1):
                mult = MultinomialCoefficients.comb(n, ni)
                mult = mult * ((s / r) ** MultinomialCoefficients.comb(ni, 2))
                mult = mult * self.get_d_term(l, n - ni, cur + 1)
                if self.modified_cell_symmetry:
                    mult = mult * self.get_cell_weight(self.cliques[l][cur-1]) ** ni
                self.d_term_cache[(l, n, cur)].append(mult)
                ret = ret + mult
        return ret
