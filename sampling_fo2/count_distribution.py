import argparse
import logging
import logzero

from sampling_fo2.context import WFOMCContext
from sampling_fo2.parser import parse_mln_constraint
from sampling_fo2.utils.polynomial import coeff_dict, create_vars, expand
from sampling_fo2.wfomc import Algo
from sampling_fo2.utils import Rational
from sampling_fo2.fol.syntax import Pred
from sampling_fo2.counting_algorithms import standard_wfomc, fast_wfomc

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
        res = standard_wfomc(context, Algo.STANDARD)
    elif algo == Algo.FASTER:
        res = fast_wfomc(
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
    parser.add_argument('--debug', action='store_true', default=False)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    if args.debug:
        logzero.loglevel(logging.DEBUG)
    else:
        logzero.loglevel(logging.INFO)
    mln, tree_constraint, cardinality_constraint = parse_mln_constraint(
        args.input
    )
    print(cardinality_constraint.pred2card)
    context = WFOMCContext(mln, tree_constraint, cardinality_constraint)
    count_dist = count_distribution(
        context.sentence, context.get_weight,
        context.domain, context.mln.preds(),
        context.tree_constraint, context.cardinality_constraint
    )
    print(count_dist)
