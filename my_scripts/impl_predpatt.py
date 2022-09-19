

import argparse
from tqdm import tqdm

from predpatt import PredPatt
from predpatt import load_conllu


def load_conll(file_path: str):
    # conll_example = [ud_parse for sent_id, ud_parse in load_conllu('/workspace/data/users/zanchangtong1/3_XIE/data/En-Zh/test.En')][0]
    # print(conll_example.pprint(K=3))
    return [ud_parse for sent_id, ud_parse in load_conllu(file_path)]

def predpatt_(ppatt_obj):
    print(" ".join([token.text for token in ppatt_obj.tokens]))
    # print(ppatt.pprint(color=True))
    print(ppatt.pprint())
    # 示例:1：
    # print(ppatt.pprint(color=True))
    # 示例2：
    # for predicate in ppatt_obj.instances:
    #     print("\t%s [%s-%s-%d]" % (" ".join(token.text for token in predicate.tokens),
    #                             predicate.root.text, predicate.root.gov_rel, predicate.root.position))
    #     for argument in predicate.arguments:
    #         print("\t\t%s [%s-%s-%d]" % (" ".join(token.text for token in argument.tokens),
    #                                     argument.root.text, argument.root.gov_rel, argument.root.position))

    # 示例3：


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='open file and do some actions')
    parser.add_argument('--input', default='/workspace/data/users/zanchangtong1/3_XIE/data/En-Zh/test.En', type=str, help='raw file path')
    # parser.add_argument('--output', default='/workspace/data/users/zanchangtong1/3_XIE/data/En-Zh/all.conllu.clean.En', type=str, help='modified file path')
    args = parser.parse_args()

    from predpatt import PredPattOpts
    from predpatt.util.ud import dep_v1, dep_v2
    resolve_relcl = True  # relative clauses
    resolve_appos = True  # appositional modifiers
    resolve_amod = True   # adjectival modifiers
    resolve_conj = True   # conjuction
    resolve_poss = True   # possessives
    ud = dep_v1.VERSION   # the version of UD
    opts = PredPattOpts(
        resolve_relcl=resolve_relcl,
        resolve_appos=resolve_appos,
        resolve_amod=resolve_amod,
        resolve_conj=resolve_conj,
        resolve_poss=resolve_poss,
        ud=ud
        )
    examples=load_conll(args.input)
    for example in tqdm(examples):
        ppatt = PredPatt(example, opts)
        predpatt_(ppatt)


