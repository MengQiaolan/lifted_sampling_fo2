from __future__ import annotations
import sys
from enum import Enum
import os
import argparse
import logging
import logzero
from logzero import logger
from contexttimer import Timer

from sampling_fo2.problems import WFOMCSProblem
from sampling_fo2.utils import Rational, round_rational
from sampling_fo2.context import WFOMCContext
from sampling_fo2.parser import parse_input
from sampling_fo2.fol.syntax import Pred
from sampling_fo2.counting_algorithms import standard_wfomc, incremental_wfomc, fast_wfomc, recursive_wfomc

class Algo(Enum):
    STANDARD = 'standard'
    FAST = 'fast'
    FASTER = 'faster'
    INCREMENTAL = 'inc'
    RECURSIVE = 'rec'
    RECURSIVE_REAL = 'rec_real'
    
    def __str__(self):
        return self.value
    
def wfomc(problem: WFOMCSProblem, algo: Algo = Algo.INCREMENTAL) -> Rational:
    context = WFOMCContext(problem)
    leq_pred = Pred('LEQ', 2)
    if leq_pred in context.formula.preds() and (algo == Algo.STANDARD or 
                                                algo == Algo.FAST or 
                                                algo == Algo.FASTER):
        logger.error('LEQ predicate is not supported in this algorithm')
        raise ValueError('LEQ predicate is not supported in this algorithm')

    logger.info('Invoke WFOMC: %s', algo)
    with Timer() as t:
        if algo == Algo.STANDARD:
            res = standard_wfomc(
                context.formula, context.domain, context.get_weight
            )
        elif algo == Algo.FAST:
            res = fast_wfomc(
                context.formula, context.domain, context.get_weight
            )
        elif algo == Algo.FASTER:
            res = fast_wfomc(
                context.formula, context.domain, context.get_weight, True
            )
        elif algo == Algo.INCREMENTAL:
            res = incremental_wfomc(
                context.formula, context.domain, context.get_weight, leq_pred
            )
        elif algo == Algo.RECURSIVE or algo == Algo.RECURSIVE_REAL:
            res = recursive_wfomc(
                context.formula, context.domain, context.get_weight, leq_pred, 
                algo == Algo.RECURSIVE_REAL
            )
        else:
            raise ValueError('Invalid algorithm: {}'.format(algo))
    res = context.decode_result(res)
    logger.info('WFOMC time: %s', t.elapsed)
    return res

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
                        choices=list(Algo), default=Algo.FAST)
    parser.add_argument('--domain_recursive',
                        action='store_true', default=False,
                        help='use domain recursive algorithm '
                             '(only for existential quantified MLN)')
    parser.add_argument('--debug', action='store_true', default=False)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    sys.setrecursionlimit(int(1e6))
    args = parse_args()
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    if args.debug:
        logzero.loglevel(logging.DEBUG)
    else:
        logzero.loglevel(logging.CRITICAL)
        logzero.loglevel(logging.INFO)
    logzero.logfile('{}/log.txt'.format(args.output_dir), mode='w')

    with Timer() as t:
        problem = parse_input(args.input)
    logger.info('Parse input: %ss', t)

    res = wfomc(problem, algo=args.algo)
    logger.info('WFOMC (arbitrary precision): %s', res)
    round_val = round_rational(res)
    logger.critical('WFOMC (round): %s (exp(%s))', round_val, round_val.ln())