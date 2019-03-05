#!/bin/bash

if [ $# -ne 3 ]; then
    echo 'Usage : sh train.sh EXPERIMENT_TYPE UDTYPE PREDICT/TRAIN'
    echo 'Example: sh train.sh jointAll 2.2 predict'
    exit 0
fi

if [[ -z "${DATASET}" ]]; then
  echo "DATASET environment variable is undefined"
else
  DATASET_VARIABLE="${DATASET}"
  echo "DATASET located in ${DATASET}"
  
fi

#Create outdir if not exist
mkdir -p ../outdir

TYPE=$1
UDTYPE=$2
PREDICT=$3

UDTRAIN=""
UDTEST=""

if [ $UDTYPE = "2.2" ]; then
    echo "UD_TYPE is 2.2"
    UDTRAIN="$DATASET_VARIABLE/ud-treebanks-v2.2/UD_Turkish-IMST/tr_imst-ud-train.conllu"
    UDTEST="$DATASET_VARIABLE/ud-treebanks-v2.2/UD_Turkish-IMST/tr_imst-ud-test.conllu"
elif [ $UDTYPE = "2.3" ]; then
    echo "UD_TYPE is 2.3"
    UDTRAIN="$DATASET_VARIABLE/ud-treebanks-v2.3/UD_Turkish-IMST/tr_imst-ud-train.conllu"
    UDTEST="$DATASET_VARIABLE/ud-treebanks-v2.3/UD_Turkish-IMST/tr_imst-ud-test.conllu"
else
    echo "Invalid UD_TYPE"
    echo "Possible UD types: 2.2 and 2.3"
fi

if [ $PREDICT = "predict" ]; then
	echo "Predicting..."
    python jPTDP.py  	--dynet-seed  123456789 \
            			--dynet-mem 1000 \--predict \
				        --model "../outdir/$TYPE-trialmodel" \
				        --params "../outdir/$TYPE-trialmodel.params" \
				        --outdir ../outdir \
				        --train $UDTRAIN \
				        --test $UDTEST \
						--output "$TYPE-test.conllu.pred" \
					    --segmentation $DATASET_VARIABLE/metu.tr \
					    --prevectors $DATASET_VARIABLE/Turkish/tr.vectors.xz 
else
	echo "Training..."
    if [ $TYPE = "seg" ]; then
		echo "Only Segmentation"
		python jPTDP.py --dynet-seed  123456789 \
		        --dynet-mem 1000 \
		        --epochs 30 \
		        --lstmdims 128 \
		        --lstmlayers 2 \
		        --hidden 100 \
		        --wembedding 100 \
		        --cembedding 50 \
		        --membedding 50 \
		        --tembedding 50 \
		        --pembedding 100 \
		        --model "$TYPE-trialmodel" \
		        --params "$TYPE-trialmodel.params" \
		        --outdir ../outdir \
		        --train $UDTRAIN \
		        --dev $UDTEST \
		        --segmentation $DATASET_VARIABLE/metu.tr \
		        --disablemorphtag \
		        --disablepipeline \
		        --prevectors $DATASET_VARIABLE/Turkish/tr.vectors.xz
	elif [ $TYPE = "morphTag" ]; then
		echo "Only Morph Tagging"
		python jPTDP.py --dynet-seed  123456789 \
		        --dynet-mem 1000 \
		        --epochs 30 \
		        --lstmdims 128 \
		        --lstmlayers 2 \
		        --hidden 100 \
		        --wembedding 100 \
		        --cembedding 50 \
		        --membedding 50 \
		        --tembedding 50 \
		        --pembedding 100 \
		        --model "$TYPE-trialmodel" \
		        --params "$TYPE-trialmodel.params" \
		        --outdir ../outdir \
		        --train $UDTRAIN \
		        --dev $UDTEST \
		        --segmentation $DATASET_VARIABLE/metu.tr \
		        --disablemorph \
		        --disablepipeline \
		        --prevectors $DATASET_VARIABLE/Turkish/tr.vectors.xz
	elif [ $TYPE = "jointAll" ]; then
		echo "Joint All Tasks"
		python jPTDP.py --dynet-seed  123456789 \
		            --dynet-mem 1000 \
		            --epochs 30 \
		            --lstmdims 128 \
		            --lstmlayers 2 \
		            --hidden 100 \
		            --wembedding 100 \
		            --cembedding 50 \
		            --membedding 50 \
		            --tembedding 50 \
		            --pembedding 100 \
		            --model "$TYPE-trialmodel" \
		            --params "$TYPE-trialmodel.params" \
		            --outdir ../outdir \
		            --train $UDTRAIN \
		            --dev $UDTEST \
		            --segmentation $DATASET_VARIABLE/metu.tr \
		            --disablepipeline \
		            --prevectors $DATASET_VARIABLE/Turkish/tr.vectors.xz
	elif [ $TYPE = "base" ]; then
		echo "Base Dependency Model"
		python jPTDP.py --dynet-seed  123456789 \
		            --dynet-mem 1000 \
		            --epochs 30 \
		            --lstmdims 128 \
		            --lstmlayers 2 \
		            --hidden 100 \
		            --wembedding 100 \
		            --cembedding 50 \
		            --membedding 50 \
		            --tembedding 50 \
		            --pembedding 100 \
		            --model "$TYPE-trialmodel" \
		            --params "$TYPE-trialmodel.params" \
		            --outdir ../outdir \
		            --train $UDTRAIN \
		            --dev $UDTEST \
		            --segmentation $DATASET_VARIABLE/metu.tr \
		            --disablemorph \
		            --disablepipeline \
		            --disablemorphtag \
		            --prevectors $DATASET_VARIABLE/Turkish/tr.vectors.xz
	else
	   echo "Invalid Experiment Type"
	   echo "Valid types: seg, morphTag, jointAll, and base" 
	fi
fi




