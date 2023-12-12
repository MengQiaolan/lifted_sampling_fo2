from __future__ import annotations
from itertools import product
from logzero import logger
from sampling_fo2.context.existential_context import BlockType, ExistentialTwoTable
from sampling_fo2.fol.syntax import top
from sampling_fo2.fol.sc2 import SC2
from sampling_fo2.fol.boolean_algebra import substitute
from sampling_fo2.fol.utils import exactly_one_qf, new_predicate, \
    convert_counting_formula

from sampling_fo2.network.constraint import CardinalityConstraint
from sampling_fo2.fol.syntax import *
from sampling_fo2.utils import Rational
from sampling_fo2.utils.third_typing import RingElement

from copy import deepcopy
from sympy.core.symbol import Symbol

def strip_formula(formula: Formula):
    """
    Strip the formula to get the quantified formula and quantifiers
    """
    quantifiers: list[Quantifier] = list()
    while not isinstance(formula, QFFormula):
        quantifiers.append(formula.quantifier_scope)
        formula = formula.quantified_formula
    return formula, quantifiers

class WFOMSContext(object):
    """
    Context for WFOMS algorithm
    """

    def __init__(self, problem):
        self.domain: set[Const] = problem.domain
        self.sentence: SC2 = problem.sentence
        self.weights: dict[Pred, tuple[Rational, Rational]] = problem.weights
        self.cardinality_constraint: CardinalityConstraint = problem.cardinality_constraint

        logger.info('sentence: \n%s', self.sentence)
        logger.info('domain: \n%s', self.domain)
        logger.info('weights:')
        for pred, w in self.weights.items():
            logger.info('%s: %s', pred, w)
        logger.info('cardinality constraint: %s', self.cardinality_constraint)

        # For existantial quantifiers existential quantified predicates,
        # they should be some converted tseitin predicates
        self.binary_ext_preds: list[Pred] = list()
        self.other_ext_preds: list[Pred] = list()
        # self.uni_var_indices: list[int] = list()
        # self.tseitin_preds: list[Pred] = list()
        # self.tseitin_to_extpred: dict[Pred, Pred] = dict()
        # self.tseitin_to_skolem: dict[Pred, Pred] = dict()
        self.block_preds: list[Pred] = list()
        self.blockpred_to_blocktype: dict[Pred, BlockType] = dict()

        self.formula: QFFormula   # skolemized sentence
        self.uni_formula: QFFormula  # universally quantified sentence
        
        self.orig_binary_ext_pred_cliques: list[list[Pred]] = list()
        self.aux_binary_ext_pred_cliques: list[list[Pred]] = list()
        self.orig_binary_ext_atom_syms: list[Pred] = list()
        self.dict_extpred2aux: dict[Pred, Pred] = dict()
        self.dict_symbol2pred: dict[Symbol, Pred] = dict()
        self.dict_pred2atomsym: dict[Symbol, Pred] = dict()
        
        self._build()
        logger.info('skolemized sentence: %s', self.formula)
        logger.info(
            'universally quantified sentence: %s', self.uni_formula)
        
        s1, s2, self.ext_formulas_for_block = self._encode_block_types()
        self.block_encoded_formula: QFFormula = self.uni_formula & s1
        self.block_encoded_formula_2: QFFormula = self.uni_formula & s2
        logger.info('block encoded sentence: %s',
                    self.block_encoded_formula)

        logger.info('weights for WFOMC:')
        for pred, w in self.weights.items():
            logger.info('%s: %s', pred, w)
        # build etables
        self.etables: list[ExistentialTwoTable] = self._build_etables()
        

    def contain_cardinality_constraint(self) -> bool:
        return self.cardinality_constraint is not None

    def contain_existential_quantifier(self) -> bool:
        return self.sentence.contain_existential_quantifier() or \
            self.sentence.contain_counting_quantifier()

    def get_weight(self, pred: Pred) -> tuple[RingElement, RingElement]:
        default = Rational(1, 1)
        if pred in self.weights:
            return self.weights[pred]
        return (default, default)

    def decode_result(self, res: RingElement):
        if not self.contain_cardinality_constraint():
            return res
        return self.cardinality_constraint.decode_poly(res)

    def _skolemize_one_formula(self, formula: QuantifiedFormula) -> QFFormula:
        """
        Only need to deal with \forall X \exists Y: P(X,Y) or \exists X: P(X)
        """
        qf_formula, quantifiers = strip_formula(formula)
        quantifier_num = len(quantifiers)

        # always introduce auxiliary predicate
        aux_pred = new_predicate(quantifier_num, AUXILIARY_PRED_NAME)
        aux_atom = aux_pred(X, Y) if quantifier_num == 2 else aux_pred(X)
        self.formula = self.formula & (qf_formula.equivalent(aux_atom))
        self.uni_formula = self.uni_formula & (qf_formula.equivalent(aux_atom))
        ext_formula = aux_atom
        ext_pred = aux_pred
        
        # for binary existential predicates symmetry
        if quantifier_num == 2:
            assert isinstance(qf_formula, AtomicFormula)
            atom_pred = qf_formula.pred
            atom_sym = qf_formula.expr
            self.dict_pred2atomsym[atom_pred] = atom_sym
            self.dict_symbol2pred[atom_sym] = atom_pred
            self.orig_binary_ext_atom_syms.append(atom_sym)
            self.dict_extpred2aux[atom_pred] = aux_pred

        if ext_pred.arity == 2:
            self.binary_ext_preds.append(ext_pred)
        else:
            self.other_ext_preds.append(ext_pred)

        if quantifier_num == 2:
            skolem_pred = new_predicate(1, SKOLEM_PRED_NAME)
            skolem_atom = skolem_pred(X)
        elif quantifier_num == 1:
            skolem_pred = new_predicate(0, SKOLEM_PRED_NAME)
            skolem_atom = skolem_pred()
        self.formula = self.formula & (skolem_atom | ~ext_formula)
        self.weights[skolem_pred] = (Rational(1, 1), Rational(-1, 1))

    def _build(self):
        self.formula = self.sentence.uni_formula
        while(not isinstance(self.formula, QFFormula)):
            self.formula = self.formula.quantified_formula

        self.ext_formulas = self.sentence.ext_formulas
        if self.sentence.contain_counting_quantifier():
            logger.info('translate SC2 to SNF')
            if not self.contain_cardinality_constraint():
                self.cardinality_constraint = CardinalityConstraint()
            for cnt_formula in self.sentence.cnt_formulas:
                uni_formula, ext_formulas, cardinality_constraint, _ = \
                    convert_counting_formula(cnt_formula, self.domain)
                self.formula = self.formula & uni_formula
                self.ext_formulas = self.ext_formulas + ext_formulas
                self.cardinality_constraint.add(*cardinality_constraint)

        if self.contain_cardinality_constraint():
            self.cardinality_constraint.build()

        self.uni_formula = self.formula
        for ext_formula in self.ext_formulas:
            self._skolemize_one_formula(ext_formula)
        if self.contain_cardinality_constraint():
            self.weights.update(
                self.cardinality_constraint.transform_weighting(
                    self.get_weight,
                )
            )

    def _encode_block_types(self):
        # Encode block type, every block type can be seen as a set of unary facts
        ext_atoms = []
        for extp in self.binary_ext_preds:
            ext_atom = extp(X, Y)
            ext_atoms.append(ext_atom)

        evidence_sentence = top
        ext_formulas_for_block: list[QuantifiedFormula] = []
        for flag in product(*([[True, False]] * len(ext_atoms))):
            domain_pred = new_predicate(1, BLOCK_PRED_NAME)
            domain_atom = domain_pred(X)
            if any(flag):
                for idx, f in enumerate(flag):
                    if not f:
                        continue
                    evidence = ext_atoms[idx]
                    skolem_pred = new_predicate(1, SKOLEM_PRED_NAME)
                    self.weights[skolem_pred] = (
                        Rational(1, 1), Rational(-1, 1))
                    skolem_lit = skolem_pred(X)
                    evidence_sentence = evidence_sentence & (
                        domain_atom | skolem_lit
                    )
                    evidence_sentence = evidence_sentence & (
                        skolem_lit | (~evidence)
                    )
                    
                    block_formula = QuantifiedFormula(Existential(Y), (~domain_atom) | evidence)
                    block_formula = QuantifiedFormula(Universal(X), block_formula)
                    ext_formulas_for_block.append(block_formula)

                block_type = BlockType(
                    pred for idx, pred in enumerate(self.binary_ext_preds) if flag[idx]
                )
            else:
                block_type = BlockType()
            self.block_preds.append(domain_pred)
            self.blockpred_to_blocktype[domain_pred] = block_type
        sentence = evidence_sentence & exactly_one_qf(self.block_preds)
        sentence_2 = exactly_one_qf(self.block_preds)
        return sentence, sentence_2, ext_formulas_for_block
    
    def _build_etables(self) -> list[ExistentialTwoTable]:
        etables = list()
        n_ext_preds = len(self.binary_ext_preds)
        # all combinations of subsets of the atoms {Zk(x)}
        for i in product(*([[True, False]] * n_ext_preds)): # 2^|e|
            for j in product(*([[True, False]] * n_ext_preds)):
                etables.append(
                    ExistentialTwoTable(i, j, tuple(self.binary_ext_preds))
                )
        # the size of etables is 4^|e| 
        # where 'e' is the number of existential predicates 
        return etables
