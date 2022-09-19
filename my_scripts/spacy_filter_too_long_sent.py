# 2022/03/01 zct
# Filter sentence-pairs of parallel corpus: len(sent) > max_len, with spacy tokenization
# Example: python spacy_filter_too_long_sent.py --src_lang_path /workspace/data/users/zanchangtong1/3_XIE/data/En-Zh/all.En --tgt_lang_path /workspace/data/users/zanchangtong1/3_XIE/data/En-Zh/all.Zh --max_len 50
import argparse
from tqdm import tqdm
import spacy

from predpatt import PredPatt
from predpatt import load_conllu

class len_filter():
    def __init__(self, max_len=50):
        self.nlp = spacy.load("en_core_web_sm")
        self.max_len = max_len
        self.total_num = 0
        self.filtered_idx = []

    def too_long(self, sentence: str):
        return len(self.nlp(sentence)) > self.max_len

    def filter_w_idx(self, tgt_path, filtered_tgt_path):

        with open(tgt_path, 'r', encoding='utf-8') as tgt, \
            open(filtered_tgt_path, 'w', encoding='utf-8') as tgt_2:

            print('>> Start filter: len(sent) > {} in file {}'.format(self.max_len, tgt_path))
            j = 0
            for (i, sent) in tqdm(enumerate(tgt)):
                if self.filtered_idx[j] == i:
                    tgt_2.write(sent)
                    tgt_2.flush()
                else:
                    continue
            print('>> Left {} sentences!!!'.format(len(self.filtered_idx)))

    def filter(self, src_path: str, tgt_path: str):

        filtered_src_path = src_path + '.filtered'
        filtered_tgt_path = tgt_path + '.filtered'

        with open(src_path, 'r', encoding='utf-8') as src, \
            open(filtered_src_path, 'w', encoding='utf-8') as src_2:
            print('>> Start filter: len(sent) > {} in file {}'.format(self.max_len, src_path))
            for (i, sent) in tqdm(enumerate(src)):
                self.total_num += 1

                if not self.too_long(sent):
                    self.filtered_idx.append(i)
                    src_2.write(sent)
                    src_2.flush()
                else:
                    continue
            print('>> Left {} sentences!!!'.format(len(self.filtered_idx)))

        self.filter_w_idx(tgt_path, filtered_tgt_path)
        print('>> Finash!!!')

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='open file and do some actions')
    parser.add_argument('--src_lang_path', default='/workspace/data/users/zanchangtong1/3_XIE/data/En-Zh/all.En', type=str, help='Source side file path.')
    parser.add_argument('--tgt_lang_path', default='/workspace/data/users/zanchangtong1/3_XIE/data/En-Zh/all.Zh', type=str, help='Target side file path.')
    parser.add_argument('--max_len', default=50, type=int, help='Filter with max length')
    args = parser.parse_args()

    Filter = len_filter(max_len = args.max_len)
    Filter.filter(args.src_lang_path, args.tgt_lang_path)


