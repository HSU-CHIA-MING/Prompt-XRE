# 
# TODO Tokenize the chinese doc with jieba's Accurate Mode

import argparse
import jieba

def tokenize_w_jieba(sent: str):


    return tok_sent

def act(input_path, output_path, sent_num): 
    with open(input_path, 'r', encoding='utf-8') as raw, \
        open(output_path, 'w', encoding='utf-8') as tgt:
        print("start preprocessing...")
        num=0
        while 1:
            try:
                raw_sent = raw.readline()
            except UnicodeDecodeError:
                print('error:', num)

            tgt.write(raw_sent)
            num += 1
            tgt.flush()
            if num >= sent_num:
                print("finish preprocessing. ^-^")
                break

if __name__=='__main__':

    # example='93. 委员会注意到小组委员会按照大会第47 / 67 和 47 / 68 号决议的规定审议了通过由 H. Freudens chuss 先生 (奥地利) 担任主席的工作组早日审查《原则》并可能加以修订的问题。'
    parser = argparse.ArgumentParser(description='open file and do some actions')
    parser.add_argument('--input', type=str, help='raw file path')
    parser.add_argument('--output', type=str, help='modified file path')
    args = parser.parse_args()

    act(args.input, args.output)

