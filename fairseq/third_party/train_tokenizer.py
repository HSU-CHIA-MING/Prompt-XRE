from pathlib import Path
import random
import os
from tokenizers import BertWordPieceTokenizer
import sentencepiece as spm
from tokenizers import ByteLevelBPETokenizer
from tokenizers import SentencePieceUnigramTokenizer

# data_path1 = "/export/c01/haoranxu/oscar/en"
data_path2 = "/export/c01/haoranxu/oscar/ru"
# data_path3 = "/export/c01/haoranxu/oscar/zh_seg"
model_path = "/export/c01/haoranxu/LMs/EnRu-large-64K-spm"

# paths1 = [str(x) for x in Path(data_path1).glob("**/*.txt")]
paths2 = [str(x) for x in Path(data_path2).glob("**/*.txt")]
# paths3 = [str(x) for x in Path(data_path3).glob("**/*.txt")]
# random.shuffle(paths1)
random.shuffle(paths2)
# random.shuffle(paths3)
NUM=10
# paths1 = random.sample(paths1, NUM)
paths2 = random.sample(paths2, NUM)
# paths3 = random.sample(paths3, NUM)
paths = []
for i in range(NUM):
    # paths.append(paths1[i])
    paths.append(paths2[i])
    # paths.append(paths3[i])
    # os.rename(paths1[i], "/export/c01/haoranxu/oscar/en-zh-brtx/"+paths1[i].split("/")[-1])
    os.rename(paths2[i], "/export/c01/haoranxu/oscar/en-zh-brtx/"+paths2[i].split("/")[-1])
    # os.rename(paths3[i], "/export/c01/haoranxu/oscar/en-zh-brtx/"+"seg"+paths3[i].split("/")[-1])

print(paths)


# Initialize a tokenizer
# tokenizer = BertWordPieceTokenizer(handle_chinese_chars=True)
# tokenizer = ByteLevelBPETokenizer(lowercase=True)
# tokenizer = SentencePieceUnigramTokenizer()

# Customize training
# tokenizer.train(files=paths, vocab_size=64_000, min_frequency=2)
# tokenizer.train(files=paths, vocab_size=64_000)

# Save files to disk
# tokenizer.save_model(model_path)

# spm.SentencePieceTrainer.train(input=",".join(paths), \
# input_format='text',\
# model_type='unigram',\
# vocab_size=63999, \
# accept_language='en,ru',\
# num_threads=24,\
# train_extremely_large_corpus=True,\
# model_prefix="en-ru",\
# max_sentence_length=1024,
# shuffle_input_sentence=True,
# bos_id=0,
# pad_id=1,
# eos_id=2,
# unk_id=3,
# normalization_rule_name="nmt_nfkc_cf")
