import json
import argparse
from tqdm import tqdm
import os

def extract_json(json_file: str):
    file = open(json_file, 'r', encoding='utf-8')
    a = json.load(file)
    return a
# json_file = '/workspace/data/users/zanchangtong1/3_XIE/Tools/ACE05-Processor/processed-data/English/train/AFP_ENG_20030304.0250.v2.json'
# dumy = extract_json(json_file)
# assert True

def load_sent_from_conllu(conllu_file: str):
    with open(conllu_file, 'r', encoding='utf-8') as f:
        sents = {}
        for line in f:
            line = line.strip()
            content = line.split(' = ')
            if not len(content) >= 2:
                continue
            
            if content[0] == '# sent_id':
                sent_id = int(content[1])
            elif content[0] == '# text':
                sents[sent_id] = ' = '.join(content[1:]).strip()
    return sents
# conllu_file = '/workspace/data/users/zanchangtong1/3_XIE/Tools/ACE05-Processor/processed-data/English/train/uk.gay-lesbian-bi_20050127.0311.conllu'
# dumy = load_sent_from_conllu(conllu_file)
# assert True

def write_sample(data_dir, lang, sample):
    input_path = os.path.join(data_dir, 'ace05.raw.{}.input0'.format(lang))
    label_path = os.path.join(data_dir, 'ace05.raw.{}.label'.format(lang))
    with open(input_path, 'a', encoding='utf-8') as input, \
        open(label_path, 'a', encoding='utf-8') as label:
        if not len(sample['arguments']) == 2: 
            return True
        line = sample['sent'].replace('\n', '').strip() + '\t' + sample['arguments'][0]['text'].replace('\n', '').strip() + '\t' \
            + sample['arguments'][1]['text'].replace('\n', '').strip() + '\n'
        input.write(line)
        label.write(sample['label'] + '\n')
        input.flush()
        label.flush()

def get_file_names(dirpath):
    filenames = []
    for file in os.listdir(dirpath):
        if file.endswith(".conllu"):
            filenames.append(os.path.splitext(file)[0])
    return filenames

def main_(lang, split, data_dir, target_dir): 
    file_dir = os.path.join(data_dir, split)
    filenames = get_file_names(file_dir)
    target_dir =  os.path.join(os.path.join(target_dir, lang), split)
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    print('>> extract data from {}'.format(file_dir))
    for filename in tqdm(filenames, total=len(filenames)):
        doc_file = os.path.join(file_dir, filename + '.conllu')
        relation_file = os.path.join(file_dir, filename + '.v2.json')

        doc = load_sent_from_conllu(doc_file)
        relations = extract_json(relation_file)['relations']
        for relation in relations:
            sample = {} 
            sample['label'] = relation['relation-type']
            sample['arguments'] = relation['arguments']
            sent_id = relation['sent_id']
            sample['sent'] = doc[int(sent_id)]
            write_sample(target_dir, lang, sample)

# data_dir_ = '/workspace/data/users/zanchangtong1/3_XIE/Tools/ACE05-Processor/processed-data/English'
# target_dir_ = '/workspace/data/users/zanchangtong1/3_XIE/data/ace05'
# main_('En', 'train', data_dir_, target_dir_)

def main(input_dir, output_dir): 

    langs = ['En', 'Zh', 'Ar']
    splits = ['train', 'dev', 'test']
    languages = ['English', 'Chinese', 'Arabic']
    for idx, lang in enumerate(langs):
        language = languages[idx]
        input_dir_ = os.path.join(input_dir, language)
        for split in splits:
            main_(lang, split, input_dir_, output_dir)


    print(">> Finish . ^-^")

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='open file and do some actions')
    parser.add_argument('--input_dir', default='/workspace/data/users/zanchangtong1/3_XIE/Tools/ACE05-Processor/processed-data', type=str, help='raw file path')
    parser.add_argument('--output_dir', default='/workspace/data/users/zanchangtong1/3_XIE/data/ace05', type=str, help='modified file path')
    args = parser.parse_args()

    main(args.input_dir, args.output_dir)
