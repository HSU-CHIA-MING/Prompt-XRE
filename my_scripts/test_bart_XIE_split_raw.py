import numpy as np
from numpy.core.arrayprint import BoolFormat
import torch
import os
import argparse
from torch.utils.data import dataloader
from tqdm import tqdm
from fairseq.data import data_utils, PadDataset
from fairseq.models.bart import BARTModel
from sklearn.metrics import f1_score

class my_dict:
    def __init__(self, path):
        self.dict_patch = path
        self.dict = []

        with open(self.dict_patch, 'r', encoding='utf-8') as f:
            for line in f:
                token, freq = line.strip().split(' ')
                self.dict.append(token)
        self.dict = self.dict[:-2]

    def __getitem__(self, idx):
        if idx < len(self.dict):
            return self.dict[idx]
        return '<unk>'

    def __len__(self):
        """Returns the number of symbols in the dictionary"""
        return len(self.dict)

if __name__ =="__main__":

    # os.environ["CUDA_VISIBLE_DEVICES"] = "0" 
    # 报错：RuntimeError: CUDA error: CUBLAS_STATUS_INTERNAL_ERROR when calling `cublasSgemm( handle, opa, opb, m, n, k, &alpha, a, lda, b, ldb, &beta, c, ldc)`
    parser = argparse.ArgumentParser(description='open file and do some actions')
    parser.add_argument('--sent', type=str, default='/workspace/data/users/zanchangtong1/3_XIE/data/ace05/split_raw/spm_data/test.ace05.raw.split_raw.Ar.spm.sent.input0', help='raw file path')
    parser.add_argument('--en1', type=str, default='/workspace/data/users/zanchangtong1/3_XIE/data/ace05/split_raw/spm_data/test.ace05.raw.split_raw.Ar.spm.entity1.input0', help='raw file path')
    parser.add_argument('--en2', type=str, default='/workspace/data/users/zanchangtong1/3_XIE/data/ace05/split_raw/spm_data/test.ace05.raw.split_raw.Ar.spm.entity2.input0', help='raw file path')
    parser.add_argument('--prompt_id', type=str, default='prompt_v2_0', help='raw file path')
    parser.add_argument('--label', default='/workspace/data/users/zanchangtong1/3_XIE/data/ace05/Ar/test/ace05.raw.Ar.label', type=str, help='label file path')
    parser.add_argument('--model_path', type=str,default='/workspace/data/users/zanchangtong1/3_XIE/checkpoints/ace05.mbart.dropout_0.1.split_raw.prompt_v2_0.en_XX', help='the model dir')
    parser.add_argument('--checkpoint', type=str,default='checkpoint_best.pt', help='the model path')
    parser.add_argument('--gpu_id', type=int,default=0, help='gpu_id')
    parser.add_argument('--src_lid', type=int, default=250005, help='source language id, 250001 ~ 250025: en - 250004; zh - 250025; ar - 250001')
    parser.add_argument('--tgt_lid', type=int, default=250005, help='target language id, 250001 ~ 250025')
    parser.add_argument('--label_dict', type=str, default="/workspace/data/users/zanchangtong1/3_XIE/data_bin/ace05/mbart.split_raw/En/label/dict.txt", help='label dict path')
    args = parser.parse_args()
    
    prompts = {
        'prompt_v2_1': ['<s> <Sent> <mask> Ent1 <mask> <Ent2> </s>','<s> <Sent> </s>'],
        'prompt_v2_2': ['<s> <Sent> <mask> <Ent1> <mask> <Ent2> </s>', '<s> <Ent1> <Ent2> </s>'],
        'prompt_v2_3': ['<s> <Sent> <mask> <Ent1> <mask> <Ent2> </s>', '<s> <Ent1> <mask> <Ent2> </s>'],
        'prompt_v2_4': ['<s> <Sent> <mask> <Ent1> <mask> <Ent2> </s>', '<s> <Sent> <mask> <Ent1> <mask> <Ent2> </s>'],
        'prompt_v2_5': ['<s> <Sent> </s>', '<s> ▁What ▁is ▁the ▁type ▁of ▁relationship ▁between <Ent1> ▁and <Ent2> </s>'],
        'prompt_v2_6': ['<s> ▁The ▁sentence ▁of <Sent> ▁includes <Ent1> ▁and <Ent2> </s>', '<s> ▁What ▁is ▁the ▁type ▁of ▁relationship ▁between <Ent1> ▁and <Ent2> </s>'],
        'prompt_v2_7': ['<s> ▁The ▁sentence ▁of <Sent> ▁includes <Ent1> <mask> <Ent2> </s>', '<s> ▁The ▁sentence ▁of <s> <Sent> <mask> ▁includes <Ent1> <mask> <Ent2> </s>'],
        'prompt_v2_8': ['<s> ▁The ▁sentence : " <Sent> ▁" ▁includes <Ent1> <mask> <Ent2> </s>', '<s> <Sent> </s>'],
        'prompt_v2_9': ['<s> ▁The ▁sentence : " <Sent> ▁" ▁includes <Ent1> <mask> <Ent2> </s>', '<s> <Ent1> <mask> <Ent2> </s>'],
    }
    prompt = prompts[args.prompt_id]
    
    # parser.add_argument('--baseline', type=bool, default=False, help='label dict path')
    torch.cuda.set_device(args.gpu_id)
    print('>> setup model...')
    label_dict = my_dict(args.label_dict)
    model = BARTModel.from_pretrained(model_name_or_path=args.model_path, checkpoint_file=args.checkpoint, src_lid=args.src_lid, tgt_lid=args.tgt_lid, bpe='sentencepiece', sentencepiece_model="/root/autodl-tmp/mbart.cc25/sentence.bpe.model")
    
    ncorrect, nsamples = 0, 0
    model.cuda() 
    model.eval() 

    with open(args.sent, 'r', encoding='utf-8' ) as fin_sent, \
        open(args.en1, 'r', encoding='utf-8' ) as fin_en1, \
        open(args.en2, 'r', encoding='utf-8' ) as fin_en2, \
        open(args.label, 'r', encoding='utf-8' ) as lin:
        sentences = fin_sent.readlines()
        entitys_1 = fin_en1.readlines()
        entitys_2 = fin_en2.readlines()
        labels = lin.readlines()
        
        GT_l = []
        pred_l = []
        # print('example:[{} ,{}]'.format(samples[0], labels[0]))
        for index, sentence in enumerate(sentences):
            tmp_prompt = prompt
            src_sample = tmp_prompt[0].replace('<Sent>', sentence.strip())
            src_sample = src_sample.replace('<Ent1>', entitys_1[index].strip())
            src_sample = src_sample.replace('<Ent2>', entitys_2[index].strip())
            
            tgt_sample = tmp_prompt[1].replace('<Sent>', sentence.strip())
            tgt_sample = tgt_sample.replace('<Ent1>', entitys_1[index].strip())
            tgt_sample = tgt_sample.replace('<Ent2>', entitys_2[index].strip())
            
            label = labels[index].strip()
            src_tokens = model.encode(src_sample)[:-1] # 移除一个多余的eos
            tgt_tokens = model.encode(tgt_sample)[:-1] # 移除一个多余的eos
            
            prediction = model.predict('sentence_classification_head', src_tokens, target_tokens = tgt_tokens).argmax().item()
            prediction_label = str(label_dict[prediction])
            GT_l.append(label)
            pred_l.append(prediction_label)
            ncorrect += int(prediction_label == label)
            nsamples += 1
    
    print('| Gt_l: ', str(GT_l))
    print('| pred_l: ', str(pred_l))
    print('example:[{} ,{} ,{}]'.format(src_tokens, tgt_sample, prediction_label))
    print('| Accuracy: ', float(ncorrect)/float(nsamples))
    print('| f1-score: ', f1_score(GT_l, pred_l, average='micro'))



'''
prompt_v2_0:
        [     0,    601,  87747,  17759,  30110,    581,  14098,  46684,  19295,
             53,    136, 106294, 135440, 172040,      7,    621, 101904,  11469,
          48031,     23,     70,  57309,     66,  16965,    420,  20271,     70,
          69496,   1631,      5, 250001,    581,  14098,  46684,  19295,     53,
            136, 106294, 135440, 172040,      7, 250001,    581,  14098,  46684,
          19295,     53,    136, 106294, 135440,      2, 250005,      1,      1,
              1,      1,      1,      1,      1,      1,      1,      1,      1,
              1,      1,      1,      1,      1,      1,      1,      1,      1,
              1,      1,      1,      1],
        [250005, 250001,      2]
prompt_v2_1:
        [     0,   3060,    111,   6097,     21,  34204,    136,  21507,    133,
           1055,    621,   7730,     47,    186,  54433,  33600,     31,     47,
           2363,  87143,      5,   2363,  69128,      7, 250001,   2363,      2,
         250005,      1,      1,      1,      1,      1,      1,      1,      1,
              1,      1,      1,      1,      1,      1,      1,      1,      1,
              1,      1,      1,      1,      1,      1,      1,      1,      1,
              1,      1,      1,      1,      1,      1,      1,      1,      1,
              1,      1,      1,      1,      1,      1,      1,      1,      1,
              1,      1,      1],
        [250005,      0,    601,  87747,  17759,  30110,    581,  14098,  46684,
          19295,     53,    136, 106294, 135440, 172040,      7,    621, 101904,
          11469,  48031,     23,     70,  57309,     66,  16965,    420,  20271,
             70,  69496,   1631,      5,      2,      1,      1,      1,      1,
              1,      1,      1,      1,      1,      1,      1,      1,      1,
              1,      1,      1,      1,      1,      1,      1,      1,      1,
              1,      1,      1,      1,      1,      1,      1,      1,      1,
              1,      1,      1,      1,      1,      1,      1,      1],
prompt_v2_2:
        [     0, 116338,  51521,      7,   5154,     10,  18025,   1340,   5962,
           1919,   1631,    927,   1663,    136,    764,    509, 168861,     47,
             28,  75161,      5,   1919, 250001,   1919,   1631,    927,   1663,
              2, 250005,      1,      1,      1,      1,      1,      1,      1,
              1,      1,      1,      1,      1,      1,      1,      1,      1,
              1,      1,      1,      1,      1,      1,      1,      1,      1,
              1,      1,      1,      1,      1,      1,      1,      1,      1,
              1,      1,      1,      1,      1,      1,      1,      1,      1,
              1,      1,      1],
        [250005,    581,  14098,  46684,  19295,     53,    136, 106294, 135440,
         172040,      7,    581,  14098,  46684,  19295,     53,    136, 106294,
         135440,      2],
prompt_v2_3:
        [     0,    601,  87747,  17759,  30110,    581,  14098,  46684,  19295,
             53,    136, 106294, 135440, 172040,      7,    621, 101904,  11469,
          48031,     23,     70,  57309,     66,  16965,    420,  20271,     70,
          69496,   1631,      5,    581,  14098,  46684,  19295,     53,    136,
         106294, 135440, 172040,      7, 250001,    581,  14098,  46684,  19295,
             53,    136, 106294, 135440,      2, 250005,      1,      1,      1,
              1,      1,      1,      1,      1,      1,      1,      1,      1,
              1,      1,      1,      1,      1,      1,      1,      1,      1,
              1,      1,      1],
        [250005, 216024, 250001, 216024,     25,      7,   5368,  35461,      2,
              1,      1,      1,      1,      1,      1,      1,      1,      1,
              1,      1,      1],
prompt_v2_4:
        [     0,   3060,    111,   6097,     21,  34204,    136,  21507,    133,
           1055,    621,   7730,     47,    186,  54433,  33600,     31,     47,
           2363,  87143,      5,   2363,  69128,      7, 250001,   2363,      2,
         250005,      1,      1,      1,      1,      1,      1,      1,      1,
              1,      1,      1,      1,      1,      1,      1,      1,      1,
              1,      1,      1,      1,      1,      1,      1,      1,      1,
              1,      1,      1,      1,      1,      1,      1,      1,      1,
              1,      1,      1,      1,      1,      1,      1,      1,      1,
              1,      1,      1],
        [250005,      0,  69496,     14,   1119,   1340,  17354,    195,  95137,
           4086,      7,  66747,     71,  34202,   4939,    563,      5, 118963,
             25,      7,   1119,   5922,  22062,      4, 111767,    284,   1042,
            959,   2367,    935,  23295,    831,     54,    100,    398,      4,
           1284,   2367,    398,    831,     54,    100,    935,  23295,      4,
           4765,    136,  35839,     98,  69496,    164,     47,  33022,     10,
         171484,      4, 137633,     10,   8437,   5426,    136,  16916,   2367,
          31486,     70,    187,   1176,   5608,  11301,      5,    935, 250001,
            935,  23295,      2],
prompt_v2_5:
        [   581, 149357,    111,      0,  82514,   1314,    765,   2843,   5962,
          30388,      7,     23,     70, 144477,   9022, 162708,    111,   8455,
            202,      4,  70605,   2247,    136,  19591,   2783,      4, 216024,
             25,      7,   5368,  35461,      5, 250001,  96853, 216024, 250001,
         216024,     25,      7,   5368,  35461,      2, 250005,      1,      1,
              1,      1,      1,      1,      1,      1,      1,      1,      1,
              1,      1,      1,      1,      1,      1,      1,      1,      1,
              1,      1,      1,      1,      1,      1,      1,      1,      1,
              1,      1,      1,      1,      1,      1,      1,      1],
        [250005,    581, 149357,    111,      0,    601,  87747,  17759,  30110,
            581,  14098,  46684,  19295,     53,    136, 106294, 135440, 172040,
              7,    621, 101904,  11469,  48031,     23,     70,  57309,     66,
          16965,    420,  20271,     70,  69496,   1631,      5, 250001,  96853,
            581,  14098,  46684,  19295,     53,    136, 106294, 135440, 172040,
              7, 250001,    581,  14098,  46684,  19295,     53,    136, 106294,
         135440,      2,      1,      1,      1,      1,      1,      1,      1,
              1,      1,      1,      1,      1,      1,      1,      1,      1,
              1,      1,      1,      1,      1,      1,      1,      1],
prompt_v2_6:
        [   581, 149357,     12,     58,     44,      0, 116338,  51521,      7,
           5154,     10,  18025,   1340,   5962,   1919,   1631,    927,   1663,
            136,    764,    509, 168861,     47,     28,  75161,      5, 250001,
          96853,   1919, 250001,   1919,   1631,    927,   1663,      2, 250005,
              1,      1,      1,      1,      1,      1,      1,      1,      1,
              1,      1,      1,      1,      1,      1,      1,      1,      1,
              1,      1,      1,      1,      1,      1,      1,      1,      1,
              1,      1,      1,      1,      1,      1,      1,      1,      1,
              1,      1,      1,      1,      1,      1,      1,      1,      1,
              1],
        [250005,      0,    581,  88308,    111,    479,  61823,    449,    845,
          19725,    678,     10,    759,    824,  93905, 202711,    704,     42,
          47314,     70,  83572,      7,    136,     57,   4126,   2678,  12610,
            845,  19725,      5,      2,      1,      1,      1,      1,      1,
              1,      1,      1,      1,      1,      1,      1,      1,      1,
              1,      1,      1,      1,      1,      1,      1,      1,      1,
              1,      1,      1,      1,      1,      1,      1,      1,      1,
              1,      1,      1,      1,      1,      1,      1,      1],
prompt_v2_7:
        [   581, 149357,     12,     58,     44,      0,  82514,   1314,    765,
           2843,   5962,  30388,      7,     23,     70, 144477,   9022, 162708,
            111,   8455,    202,      4,  70605,   2247,    136,  19591,   2783,
              4, 216024,     25,      7,   5368,  35461,      5, 250001,  96853,
         216024, 250001, 216024,     25,      7,   5368,  35461,      2, 250005,
              1,      1,      1,      1,      1,      1,      1,      1,      1,
              1,      1,      1,      1,      1,      1,      1,      1,      1,
              1,      1,      1,      1,      1,      1,      1,      1,      1,
              1,      1,      1,      1,      1,      1,      1,      1,      1,
              1],
        [250005,  75533,  32399,  18908,  37379, 250001,  75533,      2,      1,
              1,      1,      1,      1,      1,      1,      1,      1,      1,
              1,      1,      1],
'''