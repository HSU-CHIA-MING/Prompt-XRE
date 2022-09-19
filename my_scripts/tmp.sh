data_home=/workspace/data/datasets/NLG/translation/mBART_data/wmt17-enzh-conllu
home=/workspace/data/users/zanchangtong1
SPM=${home}/zanchangtong1/sentencepiece/build/src/spm_encode
MODEL=${home}/mbart.cc25/sentence.bpe.model

${SPM} --model=${MODEL} < /workspace/data/datasets/NLG/translation/mBART_data/wmt17-enzh-conllu/raw.XIE.zh_CN-en_XX.input0 > /workspace/data/datasets/NLG/translation/mBART_data/wmt17-enzh-conllu/raw.XIE.zh_CN-en_XX.spm.input0 

