import argparse
from tqdm import tqdm
import numpy as np
import os
import re
import random

def load_link(file_path):
    predpatt_dict = {}
    sent_id = 0
    pre = 0
    unparse = 0
    with open(file_path, 'r', encoding='utf-8') as f:
        for sent in tqdm(f):
            if sent == '\n' and pre == 0:
                sent_id += 1
                pre += 1
            elif sent == '\n' and pre == 1: 
                unparse += 1
                pre = 0
            else:
                pre = 0

            if sent[0] == '\t' and sent[1] != '\t':
                link = sent[1:]
                if link in predpatt_dict.keys():
                    predpatt_dict[link].append([sent_id, []])
                else:
                    predpatt_dict[link] = [[sent_id, []]]

            if sent[:2] == '\t\t':
                entity = sent[2:]
                predpatt_dict[link][-1][1].append(entity)

    return predpatt_dict

def sort_link(predpatt_dict):
    return dict(sorted(predpatt_dict.items(), key=lambda item: len(item[1])))

def count_topk(dic_items, k):
    total_num = 0
    k = -k

    for link in dic_items[k:]:
        total_num += len(link[1])
    return total_num

def save_link_num(args, dic_items, n):
    c= list(reversed(dic_items))

    show = []
    for instance in c:
        show.append((instance[0], len(instance[1])))
    with open(args.output_dir + "/sorted_predpatt_nums.txt", 'w', encoding='utf-8') as f:
        for instance in show[:n]:
            f.write(instance[0] + '\t' + str(instance[1]) + '\n')
            f.flush()

def exit_key(dict, key):
    if key in dict:
        return True
    else:
        return False

def construct_dataset(links):
    global max_num
    global sorted_predpatt
    dataset = {}
    num = [0] * len(links)
    for i, link in enumerate( links):
        try:
            for item_ in sorted_predpatt[link]:
                item = [instance[:-1] for instance in item_[1][:2]]
                if exit_key(sorted_predpatt, item[0]):
                    
                    item.append(link[:-1])
                    # dataset[item_[0]] += item
                    # dataset[item_[0]].append(link[:-1])
                    dataset[item_[0]].append(item)
                else:
                    item.append(link[:-1])
                    dataset[item_[0]] = [item]
                num[i] += 1
                if num[i] >= max_num[i]:
                    break
        except KeyError:
            print("unrecognized keys")
    print(str(num))
    return dataset

def save_file(args, dataset):
    input_src = args.src_path
    output_src = os.path.join(args.output_dir, 'XIE.zh_CN-en_XX.zh_CN')
    output_tgt = os.path.join(args.output_dir, 'XIE.zh_CN-en_XX.en_XX')
    output_bpe_tgt = os.path.join(args.output_dir, 'XIE.zh_CN-en_XX.bpe.en_XX')
    with open(input_src, 'r', encoding='utf-8') as f, \
        open(output_src, 'w', encoding='utf-8') as src, \
        open(output_tgt, 'w', encoding='utf-8') as tgt, \
        open(output_bpe_tgt, 'w', encoding='utf-8') as bpe:
        for idx, line in enumerate(tqdm(f)):
            if exit_key(dataset, idx):
                for item in dataset[idx]:
                    src.write(line)
                    tgt.write('|||'.join(item))
                    tgt.write('\n')

                    bpe_line = ' '.join(item)
                    bpe_line = bpe_line.replace('?a: ','')
                    bpe_line = bpe_line.replace('?b: ','')
                    bpe_line = bpe_line.replace('?a ','')
                    bpe_line = bpe_line.replace(' ?b','')
                    bpe.write(bpe_line)
                    bpe.write('\n')
            else:
                continue
            src.flush()
            tgt.flush()
            bpe.flush()


def get_link(link):
    if link in ['?a have ?b', '?a has ?b', '?a had ?b']:
        return '?a have/has/had ?b'
    return link

def link2label(link):
    global label_dict
    return label_dict[link]

def save_prompt_dataset(args, prompt_id='prompt_0'):

    global prompts
    prompt = prompts[prompt_id]
    # input_src = args.src_path
    input_src = os.path.join(args.output_dir, 'XIE.zh_CN-en_XX.zh_CN')
    input_tgt = os.path.join(args.output_dir, 'XIE.zh_CN-en_XX.en_XX')

    src_name = prompt_id + '.XIE.zh_CN-en_XX.input0'
    tgt_name = prompt_id + '.XIE.zh_CN-en_XX.label'
    output_src = os.path.join(args.output_dir, src_name)
    output_tgt = os.path.join(args.output_dir, tgt_name)
    # output_bpe_tgt = os.path.join(args.output_dir, 'XIE.zh_CN-en_XX.bpe.en_XX')
    with open(input_src, 'r', encoding='utf-8') as src_in, \
        open(input_tgt, 'r', encoding='utf-8') as tgt_in, \
        open(output_src, 'w', encoding='utf-8') as p_src, \
        open(output_tgt, 'w', encoding='utf-8') as p_tgt:
        
        src = src_in.readlines()
        tgt = tgt_in.readlines()
        assert len(src)==len(tgt)
        samples = list(zip(src, tgt))
        random.shuffle(samples)

        for i in range(len(samples)):
            line = prompt
            info = samples[i][1].strip().split('|||')
            line = line.replace('<sent>', samples[i][0].strip())
            line = line.replace('<entity_a>', info[0].replace('?a: ', ''))
            line = line.replace('<entity_b>', info[1].replace('?b: ', ''))
            # line = re.sub('<sent>', samples[i][0].strip(), line)
            # line = re.sub('<entity_a>', info[0].replace('?a: ', ''), line)
            # line = re.sub('<entity_b>', info[1].replace('?b: ', ''), line)

            p_src.write(line + '\n')
            # link = get_link(info[2])
            p_tgt.write(str(link2label(info[2])) + '\n')

            p_src.flush()
            p_tgt.flush()

    # return True

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='open file and do some actions')
    parser.add_argument('--input', default='/workspace/data/datasets/NLG/translation/mBART_data/wmt17-enzh-conllu/output.En', type=str, help='raw file path')
    parser.add_argument('--output_dir', default='/workspace/data/users/zanchangtong1/3_XIE/data/En-Zh', type=str, help='raw file path')
    parser.add_argument('--src_path', default='/workspace/data/datasets/NLG/translation/mBART_data/wmt17-enzh/raw/raw.train.zh_CN', type=str, help='src file path')
    parser.add_argument('--ops_v2',default=False, type=bool, help='with clear flow')
    parser.add_argument('--ops_v2_conll_input', default='/workspace/data/datasets/NLG/translation/mBART_data/wmt17-enzh-conllu/output.En', type=str, help='raw file path')
    parser.add_argument('--Zh_file', default='/workspace/data/users/zanchangtong1/3_XIE/data/En-Zh/train.Zh', type=str, help='raw file path')
    args = parser.parse_args()

    if not args.ops_v2:
        # predpatt_dict = load_link(args.input)
        # np.save(args.output_dir + "/predpatt.npy", predpatt_dict) 
        # predpatt_dict = np.load(args.output_dir + "/predpatt.npy",allow_pickle = True).item()

        # sorted_predpatt = sort_link(predpatt_dict)
        # np.save(args.output_dir + "/sorted_predpatt.npy", sorted_predpatt )
        # >> load predpatt
        # sorted_predpatt = np.load(args.output_dir + "/sorted_predpatt.npy",allow_pickle = True).item()
        # dic_items = list(sorted_predpatt.items())
        
        # >> show examples:
        # topk_num = count_topk(dic_items, 10)
        # save_link_num(args, dic_items, 3000)

        # >> save top-k links:
        # reversed_dic_items =  list(reversed(dic_items))
        # print('>> Total number of last 10 link is {}.'.format(total_num))
        # np.save(args.output_dir + "/top_10_link.npy",  dict(dic_items[-10:]))
        # print('>> Saved.')
        # selected_dict = np.load(args.output_dir + "/top_10_link.npy",allow_pickle = True).item()

        # >> create and save dataset:
        links = ['?a have ?b\n', '?a has ?b\n', '?a had ?b\n', '?a to improve ?b\n', '?a said ?b\n', '?a include ?b\n', \
        '?a involving ?b\n', '?a to promote ?b\n', '?a concerning ?b\n', '?a to make ?b\n', '?a using ?b\n', \
        '?a to achieve ?b\n']
        # max_num = [50000] * len(links)
        # for i in range(3):
        #     max_num[i] = 17000
        # dataset = construct_dataset(links)
        # save_file(args, dataset)

        values = list(range(len(links)))
        for i in range(3):
            values[i] = 0
        label_dict = dict(zip([link.strip() for link in links], values))
        prompts={'raw':'<sent>\t<entity_a>\t<entity_b>', \
            'prompt_1':'<sent> contains the relationship between <entity_a> and <entity_b> is', \
            'prompt_2':'<sent> <entity_a> <entity_b>',
            }
        save_prompt_dataset(args,  prompt_id='raw')
        # save_prompt_dataset(args,  prompt_id='prompt_1')
        # save_prompt_dataset(args,  prompt_id='prompt_2')
    else:
        predpatt_dict = load_link(args.ops_v2_conll_input)
        sorted_predpatt = sort_link(predpatt_dict)
        dic_items = list(sorted_predpatt.items())
        
        filtered_link=[]
        for link in dic_items[-300:]:
            if '?a' in link[0] and '?b' in link[0] and not '?c' in link[0]:
                filtered_link.append(link)
        import en_core_web_sm
        nlp = en_core_web_sm.load() 
        
        
        for idx, link in enumerate(filtered_link):
            filtered_link[idx] = list(link)
            line = ' '.join(link[0].strip().split(' ')[1:-1])
            line = nlp(line)
            tmp=[]
            for tok in line:
                tmp.append(tok.lemma_)
            filtered_link[idx][0] = '?a ' + ' '.join(tmp) + ' ?b'
        # merage same link
        processed_link = {}
        for link in filtered_link:
            if link[0] in processed_link.keys():
                print("merage {}: {} items".format(link[0], len(link[1])))
                processed_link[link[0]] += link[1]
            else:
                print("load {}: {} items".format(link[0], len(link[1])))
                processed_link[link[0]] = link[1]
                
        del processed_link['?a ? b ?b']
        max_num = 10500
        Zh_file=open(args.Zh_file, 'r', encoding='utf-8').readlines()
        
        with open(args.output_dir + '/train.input0', 'w', encoding='utf-8') as data_0, \
            open(args.output_dir + '/train.input1', 'w', encoding='utf-8') as data_1, \
            open(args.output_dir + '/train.input2', 'w', encoding='utf-8') as data_2, \
            open(args.output_dir + '/train.label', 'w', encoding='utf-8') as label:

            for items in processed_link.items():
                for i, item in enumerate(items[1][:-500]):
                    if i > max_num:
                        break
                    data_0.write(Zh_file[item[0]])
                    label.write(items[0].replace(' ', '_') + '\n')
                    data_1.write(item[1][0].replace('?a: ', ''))
                    data_2.write(item[1][1].replace('?b: ', ''))
                    data_0.flush()
                    label.flush()
                    data_1.flush()
                    data_2.flush()

        with open(args.output_dir + '/valid.input0', 'w', encoding='utf-8') as data_0, \
            open(args.output_dir + '/valid.input1', 'w', encoding='utf-8') as data_1, \
            open(args.output_dir + '/valid.input2', 'w', encoding='utf-8') as data_2, \
            open(args.output_dir + '/valid.label', 'w', encoding='utf-8') as label:

            for items in processed_link.items():
                for i, item in enumerate(items[1][-500:-300]):
                    if i > max_num:
                        break
                    data_0.write(Zh_file[item[0]])
                    label.write(items[0].replace(' ', '_') + '\n')
                    data_1.write(item[1][0].replace('?a: ', ''))
                    data_2.write(item[1][1].replace('?b: ', ''))
                    data_0.flush()
                    label.flush()
                    data_1.flush()
                    data_2.flush()
                    
        with open(args.output_dir + '/test.input0', 'w', encoding='utf-8') as data_0, \
            open(args.output_dir + '/test.input1', 'w', encoding='utf-8') as data_1, \
            open(args.output_dir + '/test.input2', 'w', encoding='utf-8') as data_2, \
            open(args.output_dir + '/test.label', 'w', encoding='utf-8') as label:

            for items in processed_link.items():
                for i, item in enumerate(items[1][-300:]):
                    if i > max_num:
                        break
                    data_0.write(Zh_file[item[0]])
                    label.write(items[0].replace(' ', '_') + '\n')
                    data_1.write(item[1][0].replace('?a: ', ''))
                    data_2.write(item[1][1].replace('?b: ', ''))
                    data_0.flush()
                    label.flush()
                    data_1.flush()
                    data_2.flush()
        print(">> Finished...")
        