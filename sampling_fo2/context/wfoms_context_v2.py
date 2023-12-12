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

class WFOMSContextv2(object):
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
        
        self.binary_ext_preds: list[Pred] = list()
        self.other_ext_preds: list[Pred] = list()
        
        self.block_preds: list[Pred] = list()
        self.blockpred_to_blocktype: dict[Pred, BlockType] = dict()

        self.universal_qf_formula: QFFormula
        self.skolemized_qf_formula: QFFormula
        self.block_encoded_formula: QFFormula
        
        self.binary_ext_pred_cliques: list[list[Pred]] = list()
        self.binary_ext_pred_aux: list[Pred] = list()
        self.binary_ext_atom_syms: list[Pred] = list()
        self.dict_atomsym2extpred: dict[Symbol, Pred] = dict()
        self.dict_extpred2atomsym: dict[Symbol, Pred] = dict()
        
        self._build_sentence()
        logger.info('universally quantified sentence: %s', self.universal_qf_formula)
        logger.info('skolemized sentence: %s', self.skolemized_qf_formula)
        logger.info('block encoded sentence: %s', self.block_encoded_formula)
        
        # build etables
        self.etables: list[ExistentialTwoTable] = self._build_etables()
            
    def judge_pred_symmetry(self, pred_1: Pred, pred_2: Pred, 
                            formula: Formula) -> bool:    
        atomsym_1r = self.dict_extpred2atomsym[pred_1]
        atomsym_2r = self.dict_extpred2atomsym[pred_2]
        
        atomsym_1s_x = Symbol(pred_1.name+'(X,X)')
        atomsym_2s_x = Symbol(pred_2.name+'(X,X)')
        
        atomsym_1s_y = Symbol(pred_1.name+'(Y,Y)')
        atomsym_2s_y = Symbol(pred_2.name+'(Y,Y)')
        
        if self.get_weight(pred_1) != self.get_weight(pred_2):
            return False
        
        qf_formula, _ = strip_formula(formula)
        expr = qf_formula.expr
        expr_1 = deepcopy(expr)
        expr_2 = deepcopy(expr)
        
        dic_1 = {
            atomsym_1r: Symbol('p'), 
            atomsym_2r: Symbol('q'), 
            atomsym_1s_x: Symbol('u_x'), 
            atomsym_2s_x: Symbol('v_x'), 
            atomsym_1s_y: Symbol('u_y'), 
            atomsym_2s_y: Symbol('v_y'), 
        }
        
        dic_2 = {
            atomsym_1r: Symbol('q'), 
            atomsym_2r: Symbol('p'), 
            atomsym_1s_x: Symbol('v_x'), 
            atomsym_2s_x: Symbol('u_x'), 
            atomsym_1s_y: Symbol('v_y'), 
            atomsym_2s_y: Symbol('u_y'), 
        }
        
        expr_1 = substitute(expr_1, dic_1)
        expr_2 = substitute(expr_2, dic_2)
        
        return expr_1 == expr_2
        

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
        assert isinstance(qf_formula, AtomicFormula)
        ext_atom = qf_formula
        ext_pred = qf_formula.pred
        
        self.universal_qf_formula = self.universal_qf_formula & (ext_atom | ~ext_atom)
        
        # for binary existential predicates symmetry
        if quantifier_num == 2:
            assert isinstance(qf_formula, AtomicFormula)
            atom_pred = qf_formula.pred
            atom_sym = qf_formula.expr
            self.dict_extpred2atomsym[atom_pred] = atom_sym
            self.dict_atomsym2extpred[atom_sym] = atom_pred
            self.binary_ext_atom_syms.append(atom_sym)

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
        
        self.skolemized_qf_formula = self.skolemized_qf_formula & (skolem_atom | ~ext_atom)
        self.weights[skolem_pred] = (Rational(1, 1), Rational(-1, 1))

    def _build_sentence(self):
        self.universal_qf_formula = self.sentence.uni_formula
        while(not isinstance(self.universal_qf_formula, QFFormula)):
            self.universal_qf_formula = self.universal_qf_formula.quantified_formula

        self.ext_formulas = self.sentence.ext_formulas
        if self.sentence.contain_counting_quantifier():
            logger.info('translate SC2 to SNF')
            if not self.contain_cardinality_constraint():
                self.cardinality_constraint = CardinalityConstraint()
            for cnt_formula in self.sentence.cnt_formulas:
                uni_formula, ext_formulas, cardinality_constraint, _ = \
                    convert_counting_formula(cnt_formula, self.domain)
                self.universal_qf_formula = self.universal_qf_formula & uni_formula
                self.ext_formulas = self.ext_formulas + ext_formulas
                self.cardinality_constraint.add(*cardinality_constraint)

        if self.contain_cardinality_constraint():
            self.cardinality_constraint.build()

        self.skolemized_qf_formula = self.universal_qf_formula
        for ext_formula in self.ext_formulas:
            self._skolemize_one_formula(ext_formula)
        if self.contain_cardinality_constraint():
            self.weights.update(
                self.cardinality_constraint.transform_weighting(
                    self.get_weight,
                )
            )
        
        # for binary existential predicates symmetry
        while self.binary_ext_atom_syms:
            atomsym_1 = self.binary_ext_atom_syms.pop()
            pred_1 = self.dict_atomsym2extpred[atomsym_1]
            flag = True
            for c in self.binary_ext_pred_cliques:
                pred_2 = c[0]
                if self.judge_pred_symmetry(pred_1, pred_2, 
                                            self.sentence.uni_formula):
                    c.append(self.dict_atomsym2extpred[atomsym_1])
                    flag = False
                    break
            if flag:
                self.binary_ext_pred_cliques.append([pred_1])
                
        for idx, _ in enumerate(self.binary_ext_pred_cliques):
            self.binary_ext_pred_aux.append(new_predicate(1, AUXILIARY_PRED_NAME))
        
        self.binary_ext_preds = []
        self.binary_ext_preds = [pred for clique in self.binary_ext_pred_cliques for pred in clique]
        self._encode_block_types()
        
        for pred_clique in self.binary_ext_pred_cliques:
            print(pred_clique)
            # 假如pred_clique=[p1,p2,...pm],求出所有的 (p2|~p1),(p3|((~p1) & (~p2)))...,(pm|((~p1) & (~p2) & ... & (~pm-1)))
            consequent = top
            for i in range(len(pred_clique)):
                if i != 0:
                    self.block_encoded_formula = self.block_encoded_formula & (pred_clique[i](X,Y) | consequent)
                consequent = consequent & (~pred_clique[i](X,Y))
    
    def _encode_block_types(self):
        ext_atoms: list = []
        for pred_clique in self.binary_ext_pred_cliques:
            for ext_pred in pred_clique:
                ext_atoms.append(ext_pred(X, Y))
        
        res:list[list[list[bool]]] = []
        for c in self.binary_ext_pred_cliques:
            tmp: list[list[bool]] = []
            for i in range(len(c)+1):
                tmp.append([True]*i + [False]*(len(c)-i))
            res.append(tmp)
        
        self.block_value_list = []
        for block_type in product(*res):
            self.block_value_list.append(tuple([item for sublist in block_type for item in sublist]))
           
        print(self.binary_ext_preds) 

        evidence_sentence = top
        for block_value in self.block_value_list:
            block_pred = new_predicate(1, BLOCK_PRED_NAME)
            block_atom = block_pred(X)
            if any(block_value):
                for idx, f in enumerate(block_value):
                    if not f:
                        continue
                    evidence = ext_atoms[idx]
                    skolem_pred = new_predicate(1, SKOLEM_PRED_NAME)
                    self.weights[skolem_pred] = (
                        Rational(1, 1), Rational(-1, 1))
                    skolem_lit = skolem_pred(X)
                    evidence_sentence = evidence_sentence & (
                        block_atom | skolem_lit
                    )
                    evidence_sentence = evidence_sentence & (
                        skolem_lit | (~evidence)
                    )

                block_type = BlockType(
                    pred for idx, pred in enumerate(self.binary_ext_preds) if block_value[idx]
                )
            else:
                block_type = BlockType()
            self.block_preds.append(block_pred)
            self.blockpred_to_blocktype[block_pred] = block_type
            print(block_value)
            print(block_pred, block_type)
            sentence = evidence_sentence & exactly_one_qf(self.block_preds)
        self.block_encoded_formula = self.universal_qf_formula & sentence
    
    def _build_etables(self) -> list[ExistentialTwoTable]:
        etables = list()
        
        # n_ext_preds = len(self.binary_ext_preds)  
        # all combinations of subsets of the atoms {Zk(x)}
        # for i in product(*([[True, False]] * n_ext_preds)): # 2^|e|
        #     for j in product(*([[True, False]] * n_ext_preds)):
        #         etables.append(
        #             ExistentialTwoTable(i, j, tuple(self.binary_ext_preds))
        #         )
        # the size of etables is 4^|e| 
        # where 'e' is the number of existential predicates 
        
        for i in self.block_value_list:
            for j in self.block_value_list:
                etables.append(
                    ExistentialTwoTable(
                        tuple(not e for e in i),
                        tuple(not e for e in j),
                        tuple(self.binary_ext_preds))
                )
        for e in etables:
            print(e)
        return etables
