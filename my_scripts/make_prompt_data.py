import argparse
import os
from tqdm import tqdm

def make_prompt(input_path, output_path, prompt): 
    with open(input_path, 'r', encoding='utf-8') as raw, \
        open(output_path, 'w', encoding='utf-8') as tgt:
        print('>> start constructing prompt data')
        for line in raw:
            sentence, entity_1, entity_2 = line.strip().split('\t')
            tgt_line = prompt
            tgt_line = tgt_line.replace('<sent>', sentence)
            tgt_line = tgt_line.replace('<entity_a>', entity_1)
            tgt_line = tgt_line.replace('<entity_b>', entity_2)
            tgt.write(tgt_line + '\n')
            tgt.flush()

def make_baseline_data(input_path, output_path1, output_path2):
    with open(input_path, 'r', encoding='utf-8') as raw, \
        open(output_path1, 'w', encoding='utf-8') as src, \
        open(output_path2, 'w', encoding='utf-8') as tgt:
        print('>> start constructing baseline data')
        for line in raw:
            sentence, entity_1, entity_2 = line.strip().split('\t')
            src.write(sentence)
            tgt.write(entity_1 + ' ' + entity_2)
            src.write('\n')
            tgt.write('\n')
            src.flush()
            tgt.flush()


def make_split_raw_data(input_path, output_path1, output_path2, output_path3):
    with open(input_path, 'r', encoding='utf-8') as raw, \
        open(output_path1, 'w', encoding='utf-8') as sent, \
        open(output_path2, 'w', encoding='utf-8') as entity1, \
        open(output_path3, 'w', encoding='utf-8') as entity2:
        print('>> start constructing split_raw data')
        for line in raw:
            sentence, entity_1, entity_2 = line.strip().split('\t')
            sent.write(sentence)
            entity1.write(entity_1)
            entity2.write(entity_2)
            
            sent.write('\n')
            entity1.write('\n')
            entity2.write('\n')
            sent.flush()
            entity1.flush()
            entity2.flush()

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='open file and do some actions')
    parser.add_argument('--input', type=str, help='raw file path')
    parser.add_argument('--output', type=str, help='modified file path')
    parser.add_argument('--prompt', type=str, help='modified file path')
    parser.add_argument('--output2', type=str, help='modified file path')
    parser.add_argument('--output3', type=str, help='modified file path')
    args = parser.parse_args()
    
    if args.prompt == 'baseline':
        make_baseline_data(args.input, args.output, args.output2)
        
    elif args.prompt == 'split_raw':
        make_split_raw_data(args.input, args.output, args.output2, args.output3)
    else:
        prompts={'prompt_0':'<sent> includes <entity_a> <mask> <entity_b>', \
            'prompt_1':'<sent> contains the relationship between <entity_a> and <entity_b> is', \
            'prompt_2':'<sent> <entity_a> <entity_b>',
            }
        make_prompt(args.input, args.output, prompts[args.prompt])
