# 规范conll格式的文件，不同句子以空行分割
import argparse
from tqdm import tqdm


def act(input_path, output_path): 
    with open(input_path, 'r', encoding='utf-8') as raw, \
        open(output_path, 'w', encoding='utf-8') as tgt:

        print("start preprocessing...")
        num=0
        for sent in tqdm(raw):
            if sent[0] != '#':
                tgt.write(sent)
                num += 1
                tgt.flush()
            else:
                continue
        
        print(">> Finish preprocessing {} sentences. ^-^".format(num))

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='open file and do some actions')
    parser.add_argument('--input', default='/workspace/data/users/zanchangtong1/3_XIE/data/En-Zh/all.conllu.En', type=str, help='raw file path')
    parser.add_argument('--output', default='/workspace/data/users/zanchangtong1/3_XIE/data/En-Zh/all.conllu.clean.En', type=str, help='modified file path')
    args = parser.parse_args()

    act(args.input, args.output)

