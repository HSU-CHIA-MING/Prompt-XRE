# BETTER Encoders (En/Fa)

## Prerequisites
```
conda create -n pretrain python=3.7
conda activate pretrain
```
* Install our fairseq repo
  ```
  cd fairseq-encoder
  pip install --editable ./
  ```
* [hydra](https://github.com/facebookresearch/hydra) = 1.0.3
  ```
  pip install hydra-core==1.0.3
  ```
 * sentencepiece
  ```
  pip install sentencepiece
  ```
* tensorboardX
  ```
  pip install tensorboardX
  ```
## Building Vocabulary

`cc100` and `sentencepiece` are used to build vocabulary. Due to memory limitation, only a small subset is used to build the vocabulary. We additionally manually enforce full coverage of Arabic unicode characters post-normalization.

```bash
vocab_name=spm.s4.2Bchar.unigram.128K.cov9999.arb_below_2k
# Sampling
shuf en.txt | head -c 4200000000 > en.sample.txt
shuf ar.txt | head -c 4200000000 > ar.sample.txt
# Training vocabulary
# * vocab_size = 127998 to exclude special tokens
spm_train \
  --input fa.sample.txt,en.sample.txt \
  --input_format text \
  --vocab_size 127998 \
  --model_type unigram \
  --character_coverage 0.9999 \
  --num_threads 24 \
  --model_prefix $vocab_name \
  --train_extremely_large_corpus true \
  --accept_language fa,en \
  --required_chars "؀؁؂؃؄؅؆؇؈؉؊،؍؎؏ؘؙؚؐؑؒؖؗ؛؞؟ؠءآأؤإئابةتثجحخدذرزسشصضطظعغػؼؽؾؿـفقكلمنهوىيًٌٍٟٓٔٗ٠١٢٣٤٥٦٧٨٩٪٫٬٭ٮٯٰٱٲٳٴٵٶٷٸٹٺٻټٽپٿڀځڂڃڄڅچڇڈډڊڋڌڍڎڏڐڑڒړڔڕږڗژڙښڛڜڝڞڟڠڡڢڣڤڥڦڧڨ
ڨکڪګڬڭڮگڰڱڲڳڴڵڶڷڸڹںڻڼڽھڿۀہۂۃۄۅۆۇۈۉۊۋیۍێۏېۑےۓ۔ەۖۗۘۥۦۧۨ۩۪ۭ۫ۮۯ۰۱۲۳۴۵۶۷۸۹ۺۻۼ۽۾ۿݐݑݒݓݔݕݖݗݘݙݚݛݜݝݞݟݠݡݢݣݤݥݦݧݨݩݪݫݬݭݮݯݰݱݲݳݴݵݶݷݸݹݺݻݼݽݾݿ"
# Building dictionary for Fairseq
cat $vocab_name.vocab| tail -n +4 | awk '{print $1, 1}' >$vocab_name.dict
```

## Preprocessing

Due to large corpus, we split the text into multiple shards.

```bash
# Split the corpus
mkdir splits tokenized
split -n l/10 -d -a 1 en.txt en.txt.
split -n l/10 -d -a 1 fa.txt fa.txt.
mv en.txt.* fa.txt.* splits

# Tokenize the corpus
# * This only need to be done once for the first corpus
spm_encode --model $vocab_name.model --input splits/fa.dev --output tokenized/fa.dev
spm_encode --model $vocab_name.model --input splits/en.dev --output tokenized/en.dev
# * This need to be repeated for every corpus (cc100, wiki, etc)
for i in $(seq 0 9); do
  spm_encode --model $vocab_name.model --input splits/fa.txt.$i --output tokenized/fa.txt.$i
  spm_encode --model $vocab_name.model --input splits/en.txt.$i --output tokenized/en.txt.$i
done

# Binarize the corpus
# * --validpref only need to be provided for the first corpus
for i in $(seq 0 9); do
fairseq-preprocess \
  --only-source \
  --srcdict $vocab_name.dict \
  --trainpref tokenized/fa.txt.$i \
  --validpref tokenized/fa.dev \
  --destdir EnFa-128K-bin/shard$i/fa \
  --workers 24
fairseq-preprocess \
  --only-source \
  --srcdict $vocab_name.dict \
  --trainpref tokenized/en.txt.$i \
  --validpref tokenized/en.dev \
  --destdir EnFa-128K-bin/shard$i/en \
  --workers 24
done

# Any additional corpus should be move to the same place
corpus_name=wiki
corpus_id=1
for i in $(seq 0 9); do
  for f in idx bin; do
    mv $corpus_name/EnFa-128K-bin/shard$i/fa/train.$f EnFa-128K-bin/shard$i/fa/train$corpus_id.$f
    mv $corpus_name/EnFa-128K-bin/shard$i/en/train.$f EnFa-128K-bin/shard$i/en/train$corpus_id.$f
  done
done
```


## Training

`train.sh` is used to trained the EnFa encoder.

## Conversion

`convert.py` convert the fairseq checkpoint to huggingface format. Note that it has additionally dependency on the source code of `transformers` tag `v2.11.0` since some of the module is not exposed.
