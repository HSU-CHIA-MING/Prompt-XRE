import argparse
import unittest
from xml.etree.ElementInclude import default_loader
from tqdm import tqdm
import numpy as np


# def extract_pair(input_path, output_dir): 
    


#     if input_path.split('.') == 'zh_CN':
#         output_src = output_dir + '/link.en_XX'
#         output_tgt = output_dir + '/sentence.zh_CN'
#     else:
#         print( '>> Only process en_XX')
#         return 0

#     print("start preprocessing...")
#     with open(input_path, 'r', encoding='utf-8') as raw, \
#         open(output_src, 'w', encoding='utf-8') as output_src, \
#         open(output_tgt, 'w', encoding='utf-8') as output_tgt:
        
        
#         num=0
#         while 1:
#             try:
#                 raw_sent = raw.readline()
#             except UnicodeDecodeError:
#                 print('error:', num)

#             tgt.write(raw_sent)
#             num += 1
#             tgt.flush()
#             if num >= sent_num:
#                 print("finish preprocessing. ^-^")
#                 break



def exit_id(dict_list, sent_id):
    exit_id_ = False
    for dict in dict_list[1]:
        if sent_id in dict:
            exit_id_ = True
            del dict[sent_id]
    return exit_id_

def convert_list2dict(content_list):
    dict_ = {}
    for instance in content_list:
        dict_[instance[0]] = instance[1]
    return dict_

def rm_duplicate(initial_dict):

    def unit_test():
        for instance in all_link[0]:
            if exit_id(all_link[1:], instance[0]):
                assert False

    print('>> Start removing duplicate sentence id.')
    all_link = {}
    for key, value in initial_dict.items():
        all_link[key] = convert_list2dict(value)
    # TODO
    all_link = list(all_link.items())
    for i, (key, value) in enumerate(all_link):
        for instance in value.items():
            if exit_id(all_link[i+1:], instance[0]):
                del all_link[i][1][instance[0]]
    unit_test
    print('>> Removed duplicate sentence id.')
    return all_link

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='open file and do some actions')
    parser.add_argument('--top_k_link', default='/workspace/data/datasets/NLG/translation/zct_mBART_data/wmt17-enzh-conllu/top_10_link.npy', type=str, help='raw file path')
    parser.add_argument('--input', type=str, help='modified file path')
    parser.add_argument('--output_dir', type=str, help='modified file path')
    # parser.add_argument('--sent_num', type=int, default=1000000, help='modified file path')
    args = parser.parse_args()

    top_k_dict = np.load(args.top_k_link,allow_pickle = True).item()
    top_k_dict = rm_duplicate(top_k_dict)
    # extract_pair(args.input, args.output_dir)



