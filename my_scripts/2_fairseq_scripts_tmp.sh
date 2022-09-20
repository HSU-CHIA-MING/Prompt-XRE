home=/workspace/data/users/zanchangtong1




############3 V2 dataset 中构建prompt #################
prepare_split_ace05(){
    set -x
    prompt_id=split_raw

    SRC=input0
    TGT=label
    data_home=/workspace/data/users/zanchangtong1/3_XIE/data/ace05
    prompt_raw=${data_home}/${prompt_id}

    mkdir -p ${prompt_raw}

    for lang in En Zh Ar; do 
        for split in train dev test; do 
            data_dir=${data_home}/${lang}/${split}
            python make_prompt_data.py --input ${data_dir}/ace05.raw.${lang}.input0 \
            --output ${prompt_raw}/${split}.ace05.raw.${prompt_id}.${lang}.sent.input0 \
            --output2 ${prompt_raw}/${split}.ace05.raw.${prompt_id}.${lang}.entity1.input0 \
            --output3 ${prompt_raw}/${split}.ace05.raw.${prompt_id}.${lang}.entity2.input0 --prompt ${prompt_id}
        done
    done
}
preprocess_split_ace05(){
    set -x
    prompt_id=split_raw

    SRC=input0
    TGT=label
    data_home=/workspace/data/users/zanchangtong1/3_XIE/data/ace05
    prompt_raw=${data_home}/${prompt_id}

    spm_data=${data_home}/${prompt_id}/spm_data
    MODEL=${home}/mbart.cc25/sentence.bpe.model
    SPM=${home}/zanchangtong1/sentencepiece/build/src/spm_encode
    

    mkdir -p ${spm_data}

    for lang in En Zh Ar;do
        for split in train dev test;do
            ${SPM} --model=${MODEL} < ${prompt_raw}/${split}.ace05.raw.${prompt_id}.${lang}.sent.input0 > ${spm_data}/${split}.ace05.raw.${prompt_id}.${lang}.spm.sent.input0 &
            ${SPM} --model=${MODEL} < ${prompt_raw}/${split}.ace05.raw.${prompt_id}.${lang}.entity1.input0 > ${spm_data}/${split}.ace05.raw.${prompt_id}.${lang}.spm.entity1.input0 &
            ${SPM} --model=${MODEL} < ${prompt_raw}/${split}.ace05.raw.${prompt_id}.${lang}.entity2.input0 > ${spm_data}/${split}.ace05.raw.${prompt_id}.${lang}.spm.entity2.input0 &
        done
    done
    wait
    DICT=${home}/mbart.cc25/dict.txt
    destdir=/workspace/data/users/zanchangtong1/3_XIE/data_bin/ace05/mbart.${prompt_id}
    rm -r ${destdir}
    mkdir -p ${destdir}

    for lang in En Zh Ar; do
        fairseq-preprocess \
            --only-source \
            --trainpref ${spm_data}/train.ace05.raw.${prompt_id}.${lang}.spm.sent.input0 \
            --validpref ${spm_data}/dev.ace05.raw.${prompt_id}.${lang}.spm.sent.input0 \
            --testpref ${spm_data}/test.ace05.raw.${prompt_id}.${lang}.spm.sent.input0 \
            --destdir ${destdir}/${lang}/input0 \
            --srcdict ${DICT} 

        fairseq-preprocess \
            --only-source \
            --trainpref ${spm_data}/train.ace05.raw.${prompt_id}.${lang}.spm.entity1.input0 \
            --validpref ${spm_data}/dev.ace05.raw.${prompt_id}.${lang}.spm.entity1.input0 \
            --testpref ${spm_data}/test.ace05.raw.${prompt_id}.${lang}.spm.entity1.input0 \
            --destdir ${destdir}/${lang}/input1 \
            --srcdict ${DICT} 

        fairseq-preprocess \
            --only-source \
            --trainpref ${spm_data}/train.ace05.raw.${prompt_id}.${lang}.spm.entity2.input0 \
            --validpref ${spm_data}/dev.ace05.raw.${prompt_id}.${lang}.spm.entity2.input0 \
            --testpref ${spm_data}/test.ace05.raw.${prompt_id}.${lang}.spm.entity2.input0 \
            --destdir ${destdir}/${lang}/input2 \
            --srcdict ${DICT} 

        fairseq-preprocess \
            --only-source \
            --trainpref ${data_home}/${lang}/train/ace05.raw.${lang}.label \
            --validpref ${data_home}/${lang}/dev/ace05.raw.${lang}.label \
            --testpref ${data_home}/${lang}/test/ace05.raw.${lang}.label \
            --destdir ${destdir}/${lang}/label \
            --workers 60
        wait
    done
}
train_mbart_split_ace05(){
    prompt_id=$1
    gpu_id=$2
    src_lang=$3

    dropout=${4:-0.1}
    # 40G + fp16 
    if [ ${src_lang} = Ar ];then 
        TOTAL_NUM_UPDATES=11500     # 383*30
        WARMUP_UPDATES=1150         # 10 percent of the number of updates
        LR=3e-5                     # Peak LR for polynomial LR scheduler.
        NUM_CLASSES=18
        MAX_SENTENCES=12           # Batch size.
        max_epoch=30
        lang=ar_AR # 2607 
    elif [ ${src_lang} = Zh ];then
        TOTAL_NUM_UPDATES=17820    # 594*30
        WARMUP_UPDATES=1782        # 10 percent of the number of updates
        LR=3e-5                    # Peak LR for polynomial LR scheduler.
        NUM_CLASSES=18
        MAX_SENTENCES=12           # Batch size.
        max_epoch=30
        lang=zh_CN # 7124 
    else
        TOTAL_NUM_UPDATES=16650    # 555*30
        WARMUP_UPDATES=1665        # 10 percent of the number of updates
        LR=3e-5                    # Peak LR for polynomial LR scheduler.
        NUM_CLASSES=18
        MAX_SENTENCES=12           # Batch size.
        max_epoch=30
        lang=en_XX # 6653 
    fi
    exp_feature=ace05.mbart.dropout_${dropout}.split_raw.${prompt_id}.${lang}
    langs=ar_AR,cs_CZ,de_DE,en_XX,es_XX,et_EE,fi_FI,fr_XX,gu_IN,hi_IN,it_IT,ja_XX,kk_KZ,ko_KR,lt_LT,lv_LV,my_MM,ne_NP,nl_XX,ro_RO,ru_RU,si_LK,tr_TR,vi_VN,zh_CN
    PRETRAIN=$home/mbart.cc25/model.pt
    data_bin=${home}/3_XIE/data_bin/ace05/mbart.split_raw/${src_lang}
    checkpoint_dir=/workspace/data/users/zanchangtong1/3_XIE/checkpoints/${exp_feature}
    DICT=${home}/mbart.cc25/dict.txt

    mkdir -p ${checkpoint_dir}/input0
    mkdir -p ${checkpoint_dir}/label
    cp ${DICT} ${checkpoint_dir}/input0
    cp ${DICT} ${checkpoint_dir}/label

    CUDA_VISIBLE_DEVICES=${gpu_id} fairseq-train ${data_bin}/ \
        --restore-file ${PRETRAIN} \
        --save-dir ${checkpoint_dir} \
        --encoder-normalize-before --decoder-normalize-before \
        --batch-size $MAX_SENTENCES \
        --max-tokens 4400 \
        --src-language ${lang} --tgt-language ${lang} \
        --task mbart_sentence_prediction \
        --prompt_id $prompt_id \
        --add-prev-output-tokens \
        --layernorm-embedding \
        --share-all-embeddings \
        --share-decoder-input-output-embed \
        --reset-optimizer --reset-meters --reset-dataloader --reset-lr-scheduler \
        --required-batch-size-multiple 1 \
        --init-token 0 \
        --arch mbart_large \
        --criterion sentence_prediction \
        --num-classes $NUM_CLASSES \
        --dropout ${dropout} --attention-dropout 0.1 \
        --weight-decay 0.01 --optimizer adam --adam-betas "(0.9, 0.98)" --adam-eps 1e-08 \
        --clip-norm 0.0 \
        --lr-scheduler polynomial_decay --lr $LR --total-num-update $TOTAL_NUM_UPDATES --warmup-updates $WARMUP_UPDATES --max-update $TOTAL_NUM_UPDATES \
        --fp16 --fp16-init-scale 4 --threshold-loss-scale 1 --fp16-scale-window 128 \
        --max-epoch ${max_epoch} \
        --seed 222 --langs $langs \
        --find-unused-parameters \
        --ddp-backend no_c10d --baseline \
        --best-checkpoint-metric accuracy --maximize-best-checkpoint-metric --tensorboard-logdir ${checkpoint_dir}/tensorboard > ${checkpoint_dir}/${exp_feature}.log 2>&1 

    test_home=/workspace/data/users/zanchangtong1/3_XIE/data/ace05/split_raw/spm_data
    label_home=/workspace/data/users/zanchangtong1/3_XIE/data/ace05

    for lang in En Zh Ar;do
        if [ ${src_lang} = En ];then
            src_lid=250005
            tgt_lid=250005
        elif [ ${src_lang} = Zh ];then
            src_lid=250026
            tgt_lid=250026
        elif [ ${src_lang} = Ar ];then
            src_lid=250002
            tgt_lid=250002
        fi

        python -u test_bart_XIE_split_raw.py --sent ${test_home}/test.ace05.raw.split_raw.${lang}.spm.sent.input0 \
        --en1 ${test_home}/test.ace05.raw.split_raw.${lang}.spm.entity1.input0 \
        --en2 ${test_home}/test.ace05.raw.split_raw.${lang}.spm.entity2.input0 \
        --label ${label_home}/${lang}/test/ace05.raw.${lang}.label \
        --prompt_id ${prompt_id} \
        --model_path ${checkpoint_dir} \
        --checkpoint checkpoint_best.pt --gpu_id ${gpu_id} \
        --label_dict ${home}/3_XIE/data_bin/ace05/mbart.split_raw/${src_lang}/label/dict.txt \
        --src_lid ${src_lid} --tgt_lid ${tgt_lid} > ${checkpoint_dir}/best.${exp_feature}.${src_lang}-${lang} 2>&1 

        python -u test_bart_XIE_split_raw.py --sent ${test_home}/test.ace05.raw.split_raw.${lang}.spm.sent.input0 \
        --en1 ${test_home}/test.ace05.raw.split_raw.${lang}.spm.entity1.input0 \
        --en2 ${test_home}/test.ace05.raw.split_raw.${lang}.spm.entity2.input0 \
        --label ${label_home}/${lang}/test/ace05.raw.${lang}.label \
        --prompt_id ${prompt_id} \
        --model_path ${checkpoint_dir} \
        --checkpoint checkpoint_last.pt --gpu_id ${gpu_id} \
        --label_dict ${home}/3_XIE/data_bin/ace05/mbart.split_raw/${src_lang}/label/dict.txt \
        --src_lid ${src_lid} --tgt_lid ${tgt_lid} > ${checkpoint_dir}/last.${exp_feature}.${src_lang}-${lang} 2>&1 
    done
    
    rm ${checkpoint_dir}/checkpoint?.pt
    rm ${checkpoint_dir}/checkpoint??.pt
}

prepare(){
    prepare_split_ace05
    preprocess_split_ace05
}
# prepare

# exp_feature=hp1

func0(){
    gpu_id=7
    prompt_=prompt_v2_1
    train_mbart_split_ace05 $prompt_ $gpu_id Ar
    wait
    train_mbart_split_ace05 $prompt_ $gpu_id En 
    wait
    train_mbart_split_ace05 $prompt_ $gpu_id Zh 
}
func1(){
    gpu_id=0
    prompt_=prompt_v2_2
    train_mbart_split_ace05 $prompt_ $gpu_id Ar
    wait
    train_mbart_split_ace05 $prompt_ $gpu_id En 
    wait
    train_mbart_split_ace05 $prompt_ $gpu_id Zh 
}
func2(){
    gpu_id=1
    prompt_=prompt_v2_3
    train_mbart_split_ace05 $prompt_ $gpu_id Ar
    wait
    train_mbart_split_ace05 $prompt_ $gpu_id En 
    wait
    train_mbart_split_ace05 $prompt_ $gpu_id Zh 
}
func3(){
    gpu_id=2
    prompt_=prompt_v2_4
    train_mbart_split_ace05 $prompt_ $gpu_id Ar
    wait
    train_mbart_split_ace05 $prompt_ $gpu_id En 
    wait
    train_mbart_split_ace05 $prompt_ $gpu_id Zh 
}
func4(){
    gpu_id=3
    prompt_=prompt_v2_5
    train_mbart_split_ace05 $prompt_ $gpu_id Ar
    wait
    train_mbart_split_ace05 $prompt_ $gpu_id En 
    wait
    train_mbart_split_ace05 $prompt_ $gpu_id Zh 
}
func5(){
    gpu_id=4
    prompt_=prompt_v2_6
    train_mbart_split_ace05 $prompt_ $gpu_id Ar
    wait
    train_mbart_split_ace05 $prompt_ $gpu_id En 
    wait
    train_mbart_split_ace05 $prompt_ $gpu_id Zh 
}
func6(){
    gpu_id=5
    prompt_=prompt_v2_7
    train_mbart_split_ace05 $prompt_ $gpu_id Ar
    wait
    train_mbart_split_ace05 $prompt_ $gpu_id En 
    wait
    train_mbart_split_ace05 $prompt_ $gpu_id Zh 
}
func7(){
    gpu_id=6
    prompt_=prompt_v2_8
    train_mbart_split_ace05 $prompt_ $gpu_id Ar
    wait
    train_mbart_split_ace05 $prompt_ $gpu_id En
    wait
    train_mbart_split_ace05 $prompt_ $gpu_id Zh 
}
func8(){
    gpu_id=7
    prompt_=prompt_v2_9
    train_mbart_split_ace05 $prompt_ $gpu_id Ar
    wait
    train_mbart_split_ace05 $prompt_ $gpu_id En
    wait 
    train_mbart_split_ace05 $prompt_ $gpu_id Zh 
}

func0 &
func1 &
func2 &
func3 &
func4 &
func5 &
func6 &
func7 &
func8 &
wait

# cd /workspace/data/users/zanchangtong1/2_High_Resource_Translation/code/my_scripts

# bash setup_official.sh

# bash big.sh
