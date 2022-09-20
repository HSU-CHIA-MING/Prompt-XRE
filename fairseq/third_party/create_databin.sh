START=${1}
END=${2}
PRE=${START}${END}

for i in `seq ${START} ${END}`
do
    echo part${i}
    cat /export/c01/haoranxu/oscar/en/en_part_${i}.txt \
    /export/c01/haoranxu/oscar/zh/zh_part_${i}.txt \
    /export/c01/haoranxu/oscar/zh_seg/zh_part_${i}.txt |shuf > /export/c01/haoranxu/oscar/cached/${PRE}train_raw.txt

    echo tokenizing ${i}
    spm_encode --input /export/c01/haoranxu/oscar/cached/${PRE}train_raw.txt \
    --output /export/c01/haoranxu/oscar/cached/${PRE}train.txt \
    --model /export/c01/haoranxu/LMs/EnZh-large-100K/en-zh.model

    echo fairseq process ${i}
    fairseq-preprocess \
    --only-source \
    --srcdict /export/c01/haoranxu/LMs/EnZh-large-100K/en-zh.dict \
    --trainpref /export/c01/haoranxu/oscar/cached/${PRE}train.txt \
    --destdir /export/c01/haoranxu/oscar/en-zh-databin/databin.${i} \
    --workers 24
    # --validpref /export/c01/haoranxu/oscar/cached/dev.txt \

    rm /export/c01/haoranxu/oscar/cached/${PRE}train*
done
