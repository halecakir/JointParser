TYPE={$1-None}
UDTYPE=$2

UDTRAIN=""
UDTEST=""

if [ $UDTYPE = "2.2" ]; then
    echo "UD_TYPE is 2.2"
    UDTRAIN="/home/huseyin/Data/ud-treebanks-v2.2/UD_Turkish-IMS/tr_imst-ud-train.conllu"
    UDTEST="/home/huseyin/Data/ud-treebanks-v2.2/UD_Turkish-IMS/tr_imst-ud-test.conllu"
else
    echo "UD_TYPE is 2.3"
    UDTRAIN="/home/huseyin/Data/ud-treebanks-v2.3/UD_Turkish-IMS/tr_imst-ud-train.conllu"
    UDTEST="/home/huseyin/Data/ud-treebanks-v2.3/UD_Turkish-IMS/tr_imst-ud-test.conllu"
fi

if [ $TYPE = "seg" ]; then
    echo "Only Segmentation"
    python jPTDP.py --dynet-seed  123456789 \
            --dynet-mem 1000 \
            --epochs 12 \
            --lstmdims 128 \
            --lstmlayers 2 \
            --hidden 100 \
            --wembedding 200 \
            --cembedding 50 \
            --membedding 50 \
            --tembedding 50 \
            --pembedding 100 \
            --model "$TYPE-trialmodel" \
            --params "$TYPE-trialmodel.params" \
            --outdir /home/huseyin/outdir \
            --train $UDTRAIN \
            --dev $UDTEST \
            --segmentation /home/huseyin/Data/metu.tr \
            --disablemorphtag \
            --disablepipeline \
            --prevectors /home/huseyin/Data/Turkish/tr.vectors.xz
elif [ $TYPE = "morphTag" ]; then
    echo "Only Morph Tagging"
    python jPTDP.py --dynet-seed  123456789 \
            --dynet-mem 1000 \
            --epochs 12 \
            --lstmdims 128 \
            --lstmlayers 2 \
            --hidden 100 \
            --wembedding 200 \
            --cembedding 50 \
            --membedding 50 \
            --tembedding 50 \
            --pembedding 100 \
            --model "$TYPE-trialmodel" \
            --params "$TYPE-trialmodel.params" \
            --outdir /home/huseyin/outdir \
            --train $UDTRAIN \
            --dev $UDTEST \
            --segmentation /home/huseyin/Data/metu.tr \
            --disablemorph \
            --disablepipeline \
            --prevectors /home/huseyin/Data/Turkish/tr.vectors.xz
elif [ $TYPE = "jointAll" ]; then
    echo "Joint All Tasks"
    python jPTDP.py --dynet-seed  123456789 \
                --dynet-mem 1000 \
                --epochs 12 \
                --lstmdims 128 \
                --lstmlayers 2 \
                --hidden 100 \
                --wembedding 200 \
                --cembedding 50 \
                --membedding 50 \
                --tembedding 50 \
                --pembedding 100 \
                --model "$TYPE-trialmodel" \
                --params "$TYPE-trialmodel.params" \
                --outdir /home/huseyin/outdir \
                --train $UDTRAIN \
                --dev $UDTEST \
                --segmentation /home/huseyin/Data/metu.tr \
                --disablepipeline \
                --prevectors /home/huseyin/Data/Turkish/tr.vectors.xz
else
    echo "Base Dependency Model"
    python jPTDP.py --dynet-seed  123456789 \
                --dynet-mem 1000 \
                --epochs 12 \
                --lstmdims 128 \
                --lstmlayers 2 \
                --hidden 100 \
                --wembedding 200 \
                --cembedding 50 \
                --membedding 50 \
                --tembedding 50 \
                --pembedding 100 \
                --model "$TYPE-trialmodel" \
                --params "$TYPE-trialmodel.params" \
                --outdir /home/huseyin/outdir \
                --train $UDTRAIN \
                --dev $UDTEST \
                --segmentation /home/huseyin/Data/metu.tr \
                --disablemorph \
                --disablepipeline \
                --disablemorphtag \
                --prevectors /home/huseyin/Data/Turkish/tr.vectors.xz
fi
