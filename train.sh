#!/bin/bash

if [ $# -ne 4 ]; then
    echo 'Usage : sh train.sh EXPERIMENT_TYPE UDTYPE PREDICT/TRAIN LANGUAGE'
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
LANG=$4

UDTRAIN=""
UDTEST=""
PREVECTORS=""


echo "UD_TYPE is $UDTYPE"

if [ $LANG = "Turkish" ]; then
	UDTRAIN="$DATASET_VARIABLE/ud-treebanks-v$UDTYPE/UD_Turkish-IMST/tr_imst-ud-train.conllu"
	UDTEST="$DATASET_VARIABLE/ud-treebanks-v$UDTYPE/UD_Turkish-IMST/tr_imst-ud-test.conllu"
	PREVECTORS="$DATASET_VARIABLE/$LANG/tr.vectors.xz"
elif [ $LANG = "Finnish" ]; then
	UDTRAIN="$DATASET_VARIABLE/ud-treebanks-v$UDTYPE/UD_Finnish-TDT/fi_tdt-ud-train.conllu"
	UDTEST="$DATASET_VARIABLE/ud-treebanks-v$UDTYPE/UD_Finnish-TDT/fi_tdt-ud-test.conllu"
	PREVECTORS="$DATASET_VARIABLE/$LANG/fi.vectors.xz"
elif [ $LANG = "Hungarian" ]; then
	UDTRAIN="$DATASET_VARIABLE/ud-treebanks-v$UDTYPE/UD_Hungarian-Szeged/hu_szeged-ud-train.conllu"
	UDTEST="$DATASET_VARIABLE/ud-treebanks-v$UDTYPE/UD_Hungarian-Szeged/hu_szeged-ud-test.conllu"
	PREVECTORS="$DATASET_VARIABLE/$LANG/hu.vectors.xz"
fi

if [ $PREDICT = "predict" ]; then
	echo "Predicting..."
    python jPTDP.py  	--dynet-seed  123456789 \
            			--dynet-mem 1000 \--predict \
				        --model "../outdir/$LANG-$TYPE-trialmodel" \
				        --params "../outdir/$LANG-$TYPE-trialmodel.params" \
				        --outdir ../outdir \
				        --train $UDTRAIN \
				        --test $UDTEST \
						--output "$LANG-$TYPE-test.conllu.pred" \
					    --segmentation $DATASET_VARIABLE/metu.tr \
					    --prevectors $PREVECTORS 
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
						--model "$LANG-$TYPE-trialmodel" \
						--params "$LANG-$TYPE-trialmodel.params" \
						--outdir ../outdir \
						--train $UDTRAIN \
						--dev $UDTEST \
						--segmentation $DATASET_VARIABLE/metu.tr \
						--disablemorphtag \
						--disablepipeline \
						--prevectors $PREVECTORS
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
						--model "$LANG-$TYPE-trialmodel" \
						--params "$LANG-$TYPE-trialmodel.params" \
						--outdir ../outdir \
						--train $UDTRAIN \
						--dev $UDTEST \
						--segmentation $DATASET_VARIABLE/metu.tr \
						--disablemorph \
						--disablepipeline \
						--prevectors $PREVECTORS
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
						--model "$LANG-$TYPE-trialmodel" \
						--params "$TYPE-trialmodel.params" \
						--outdir ../outdir \
						--train $UDTRAIN \
						--dev $UDTEST \
						--segmentation $DATASET_VARIABLE/metu.tr \
						--disablepipeline \
						--prevectors $PREVECTORS
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
						--model "$LANG-$TYPE-trialmodel" \
						--params "$LANG-$TYPE-trialmodel.params" \
						--outdir ../outdir \
						--train $UDTRAIN \
						--dev $UDTEST \
						--segmentation $DATASET_VARIABLE/metu.tr \
						--disablemorph \
						--disablepipeline \
						--disablemorphtag \
						--prevectors $PREVECTORS
	else
	   echo "Invalid Experiment Type"
	   echo "Valid types: seg, morphTag, jointAll, and base" 
	fi
fi




