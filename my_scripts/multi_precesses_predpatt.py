import argparse
import encodings
import os
from tqdm import tqdm
from multiprocessing import Pool
import numpy as np

from predpatt import PredPatt
from predpatt import load_conllu

from predpatt import PredPattOpts
from predpatt.util.ud import dep_v1, dep_v2
resolve_relcl = True  # relative clauses
resolve_appos = True  # appositional modifiers
resolve_amod = True   # adjectival modifiers
resolve_conj = True   # conjuction
resolve_poss = True   # possessives
ud = dep_v1.VERSION   # the version of UD
# enable PredPatt to tackle various syntactic phenomena.
opts = PredPattOpts(
    resolve_relcl=resolve_relcl,
    resolve_appos=resolve_appos,
    resolve_amod=resolve_amod,
    resolve_conj=resolve_conj,
    resolve_poss=resolve_poss,
    ud=ud
    )

def count_character(symbol, file_path):
    def blocks(files, size=65536):
        while True:
            b = files.read(size)
            if not b: break
            yield b

    with open(file_path, "r",encoding="utf-8",errors='ignore') as f:
        return sum(bl.count(symbol) for bl in blocks(f))

def count_dumy_line(file_path):
    num = 0
    with open(file_path, "r",encoding="utf-8",errors='ignore') as f:
        for sent in tqdm(f):  
            if sent == '\n':
                num+=1
    return num+1

def chunk_split(pargs):

    def unit_test():
        num=0
        print('>> unit test...')
        for i in tqdm(range(pargs.nums)):
            mid_raw_path = pargs.output_dir + '/mid.raw.'+str(i)
            num += count_dumy_line(mid_raw_path)
        assert num == instance_num
        print('>> Chunks created sucessfully.')


    # counting instances number in the file 
    # split_symbol='\n\n' # 统计子字符串的个数不会精确，分chunk可能会把字符串截断。
    # instance_num = count_character(split_symbol, pargs.raw_path)+1
    print('>> Counting instance number...')
    instance_num = count_dumy_line(pargs.raw_path)
    # assert instance_num == 2000000
    print('>> Total instance num in raw file is {}.'.format(instance_num))
    instance_num_offset = int(instance_num / pargs.nums)
    instance_num_offsets = instance_num_offset * (np.array(range(pargs.nums-1))+1)
    
    total_num = 0
    with open(pargs.raw_path, 'r', encoding='utf-8') as raw:
        for i in range(pargs.nums-1):
            mid_raw_path = pargs.output_dir + '/mid.raw.'+str(i)
            with open(mid_raw_path, 'w', encoding='utf-8') as sub_raw:
                while 1:
                    sent = raw.readline()
                    if sent == '\n':
                        total_num +=1
                    if total_num == instance_num_offsets[i]:
                        break
                    sub_raw.write(sent)
                    sub_raw.flush()
                print('>> Processing instances: {} ...'.format(total_num))

        mid_raw_path = pargs.output_dir + '/mid.raw.'+str(pargs.nums-1)
        with open(mid_raw_path, 'w', encoding='utf-8') as sub_raw:
            while 1:
                sent = raw.readline()
                if not sent: 
                    break
                if sent == '\n':
                    total_num +=1
                sub_raw.write(sent)
                sub_raw.flush()
                
            print('>> Processing instances: {} ...'.format(total_num))

    # unit_test()
    return instance_num
    # print('>> Total number of instances is {}.'.format(total_num))


def load_conll(file_path: str):
    return [ud_parse for sent_id, ud_parse in load_conllu(file_path)]

def predpatt_(src_path, mid_out_path):
    examples=load_conll(src_path)
    with open(mid_out_path, 'w', encoding='utf-8') as tgt:
        for example in examples:
            # ppatt = PredPatt(example, opts)
            ppatt = PredPatt(example)
            tgt.write(' '.join([token.text for token in ppatt.tokens]))
            tgt.write('\n')
            tgt.write(ppatt.pprint())
            tgt.write('\n\n')
            tgt.flush()
            

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='tokenize the target raw text of arabic')
    parser.add_argument('--raw_path', default='/workspace/data/users/zanchangtong1/3_XIE/data/En-Zh/all.conllu.clean.En', help='the input arabic raw file')
    parser.add_argument('--output_dir', default='/workspace/data/users/zanchangtong1/3_XIE/data/En-Zh', help='tokenized arabic file')
    parser.add_argument('--nums',default=100, type=int, help='tokenized arabic file')
    pargs = parser.parse_args()

    print('>> split the raw file: {}'.format(pargs.raw_path))
    total_num = chunk_split(pargs)
    # total_num = count_dumy_line(pargs.raw_path)

    print('>> Start predpatt')
    p=Pool(pargs.nums)
    for precess_id in range(pargs.nums):    
        mid_raw_path = pargs.output_dir + '/mid.raw.'+str(precess_id)
        mid_out_path = pargs.output_dir + '/mid.out.'+str(precess_id)
        # predpatt_(mid_raw_path, mid_out_path)
        p.apply_async(predpatt_, args=(mid_raw_path, mid_out_path))
    p.close()
    p.join()

    print('>> Finish predpatt')
    output_path = pargs.output_dir + '/output.' + pargs.raw_path.split('.')[-1]
    num = 0
    pre = 0
    unparse = 0
    with open(output_path, 'w', encoding='utf-8' ) as output:
        for precess_id in tqdm(range(pargs.nums)):
            mid_out_path = pargs.output_dir + '/mid.out.'+str(precess_id)
            with open(mid_out_path, 'r', encoding='utf-8' ) as mid_output:
                for line in mid_output:
                    if line == '\n' and pre == 0:
                        num += 1
                        pre += 1
                    elif line == '\n' and pre == 1: 
                        unparse += 1
                        pre = 0
                    else:
                        pre = 0
                    output.write(line)

    # 存在无法解析的句子
    # total_num = 2000000
    # assert num == total_num
    for precess_id in range(pargs.nums):
        mid_raw_path = pargs.output_dir + '/mid.raw.'+str(precess_id)
        mid_out_path = pargs.output_dir + '/mid.out.'+str(precess_id)
        os.remove(mid_raw_path)
        os.remove(mid_out_path)

    print('>> Unparsed sentences number is {}.'.format(unparse))
    print('>> Finish....')
