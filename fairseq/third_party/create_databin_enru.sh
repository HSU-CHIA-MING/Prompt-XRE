START=${1}
END=${2}
PRE=${START}${END}

for i in `seq ${START} ${END}`
do
    echo part${i}
    cat /export/c01/haoranxu/oscar/en/en_part_${i}.txt \
    /export/c01/haoranxu/oscar/ru/ru_part_${i}.txt | shuf > /export/c01/haoranxu/oscar/cached/${PRE}rutrain_raw.txt

    echo tokenizing ${i}
    spm_encode --input /export/c01/haoranxu/oscar/cached/${PRE}rutrain_raw.txt \
    --output /export/c01/haoranxu/oscar/cached/${PRE}rutrain.txt \
    --model /export/c01/haoranxu/LMs/EnRu-large-64K/en-ru.model

    echo fairseq process ${i}

    if [ ${i} == "1" ]; then
        fairseq-preprocess \
        --only-source \
        --srcdict /export/c01/haoranxu/LMs/EnRu-large-64K/en-ru.dict \
        --trainpref /export/c01/haoranxu/oscar/cached/${PRE}rutrain.txt \
        --validpref /export/c01/haoranxu/oscar/cached/dev-enru.txt \
        --destdir /export/c01/haoranxu/oscar/en-ru-databin/databin.${i} \
        --workers 24
    else 
        fairseq-preprocess \
        --only-source \
        --srcdict /export/c01/haoranxu/LMs/EnRu-large-64K/en-ru.dict \
        --trainpref /export/c01/haoranxu/oscar/cached/${PRE}rutrain.txt \
        --destdir /export/c01/haoranxu/oscar/en-ru-databin/databin.${i} \
        --workers 24

    fi

    rm /export/c01/haoranxu/oscar/cached/${PRE}rutrain*
done



