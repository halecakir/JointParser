#!/bin/bash

if [ $# -ne 8 ]; then
    echo 'Usage : sh train.sh EXPERIMENT_TYPE UDTYPE PREDICT/TRAIN LANGUAGE EPOCHNUM MORPH_COMP MTAG_COMP POS_COMP'
    echo 'Example: sh train.sh jointAll 2.2 train Turkish 30 morph_attention mtag_attention pos_attention'
    exit 0
fi

if [[ -z "${DATASET_VARIABLE}" ]]; then
  echo "DATASET environment variable is undefined"
else
  DATASET_VARIABLE="${DATASET_VARIABLE}"
  echo "DATASET located in ${DATASET_VARIABLE}"
  
fi

#Create outdir if not exist
mkdir -p ../outdir

TYPE=$1
UDTYPE=$2
PREDICT=$3
LANG=$4
EPOCH=$5
MORPH_COMP=$6
MTAG_COMP=$7
POS_COMP=$8

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
elif [ $LANG = "Czech" ]; then
	UDTRAIN="$DATASET_VARIABLE/ud-treebanks-v$UDTYPE/UD_Czech-PDT/cs_pdt-ud-train.conllu"
	UDTEST="$DATASET_VARIABLE/ud-treebanks-v$UDTYPE/UD_Czech-PDT/cs_pdt-ud-test.conllu"
	PREVECTORS="$DATASET_VARIABLE/$LANG/cs.vectors.xz"
fi

if [ $PREDICT = "predict" ]; then
	echo "Predicting..."
    python jPTDP.py  	--dynet-seed  123456789 \
            			--dynet-mem 1000 \--predict \
				        --model "$LANG-$TYPE-$MORPH_COMP-$MTAG_COMP-$POS_COMP-model" \
				        --params "$LANG-$TYPE-$MORPH_COMP-$MTAG_COMP-$POS_COMP-model.params" \
				        --outdir ../outdir \
				        --train $UDTRAIN \
				        --test $UDTEST \
						--output "$LANG-$TYPE-$MORPH_COMP-$MTAG_COMP-$POS_COMP-test.conllu.pred" \
						--mtag-encoding-composition-type $MTAG_COMP \
						--morph-encoding-composition-type $MORPH_COMP \
						--pos-encoding-composition-type $POS_COMP \
					    --segmentation $DATASET_VARIABLE/metu.tr \
					    --prevectors $PREVECTORS 
else
	echo "Training..."
    if [ $TYPE = "seg" ]; then
		echo "Only Segmentation"
		python jPTDP.py --dynet-seed  123456789 \
						--dynet-mem 1000 \
						--epochs $EPOCH \
						--lstmdims 128 \
						--lstmlayers 2 \
						--hidden 100 \
						--wembedding 100 \
						--cembedding 50 \
						--membedding 50 \
						--tembedding 50 \
						--pembedding 100 \
						--model "$LANG-$TYPE-$MORPH_COMP-$MTAG_COMP-$POS_COMP-model" \
						--params "$LANG-$TYPE-$MORPH_COMP-$MTAG_COMP-$POS_COMP-model.params" \
						--output "$LANG-$TYPE-$MORPH_COMP-$MTAG_COMP-$POS_COMP-test.conllu.pred" \
						--outdir ../outdir \
						--train $UDTRAIN \
						--dev $UDTEST \
						--segmentation $DATASET_VARIABLE/metu.tr \
						--disablemorphtag \
						--disablepipeline \
						--prevectors $PREVECTORS
    elif [ $TYPE = "segAblation" ]; then
		echo "Only Gold Segmentation"
		python jPTDP.py --dynet-seed  123456789 \
						--dynet-mem 1000 \
						--epochs $EPOCH \
						--lstmdims 128 \
						--lstmlayers 2 \
						--hidden 100 \
						--wembedding 100 \
						--cembedding 50 \
						--membedding 50 \
						--tembedding 50 \
						--pembedding 100 \
						--model "$LANG-$TYPE-$MORPH_COMP-$MTAG_COMP-$POS_COMP-model" \
						--params "$LANG-$TYPE-$MORPH_COMP-$MTAG_COMP-$POS_COMP-model.params" \
						--output "$LANG-$TYPE-$MORPH_COMP-$MTAG_COMP-$POS_COMP-test.conllu.pred" \
						--outdir ../outdir \
						--train $UDTRAIN \
						--dev $UDTEST \
						--segmentation $DATASET_VARIABLE/metu.tr \
						--enable-gold-morph \
						--disablemorphtag \
						--disablepipeline \
						--prevectors $PREVECTORS						
	elif [ $TYPE = "morphTag" ]; then
		echo "Only Morph Tagging"
		python jPTDP.py --dynet-seed  123456789 \
						--dynet-mem 1000 \
						--epochs $EPOCH \
						--lstmdims 128 \
						--lstmlayers 2 \
						--hidden 100 \
						--wembedding 100 \
						--cembedding 50 \
						--membedding 50 \
						--tembedding 50 \
						--pembedding 100 \
						--model "$LANG-$TYPE-$MORPH_COMP-$MTAG_COMP-$POS_COMP-model" \
						--params "$LANG-$TYPE-$MORPH_COMP-$MTAG_COMP-$POS_COMP-model.params" \
						--output "$LANG-$TYPE-$MORPH_COMP-$MTAG_COMP-$POS_COMP-test.conllu.pred" \
						--outdir ../outdir \
						--train $UDTRAIN \
						--dev $UDTEST \
						--segmentation $DATASET_VARIABLE/metu.tr \
						--disablemorph \
						--disablepipeline \
						--mtag-encoding-composition-type $MTAG_COMP \
						--morph-encoding-composition-type $MORPH_COMP \
						--pos-encoding-composition-type $POS_COMP \
						--prevectors $PREVECTORS
	elif [ $TYPE = "morphTagAblation" ]; then
		echo "Only Gold Morph Tagging"
		python jPTDP.py --dynet-seed  123456789 \
						--dynet-mem 1000 \
						--epochs $EPOCH \
						--lstmdims 128 \
						--lstmlayers 2 \
						--hidden 100 \
						--wembedding 100 \
						--cembedding 50 \
						--membedding 50 \
						--tembedding 50 \
						--pembedding 100 \
						--model "$LANG-$TYPE-$MORPH_COMP-$MTAG_COMP-$POS_COMP-model" \
						--params "$LANG-$TYPE-$MORPH_COMP-$MTAG_COMP-$POS_COMP-model.params" \
						--output "$LANG-$TYPE-$MORPH_COMP-$MTAG_COMP-$POS_COMP-test.conllu.pred" \
						--outdir ../outdir \
						--train $UDTRAIN \
						--dev $UDTEST \
						--segmentation $DATASET_VARIABLE/metu.tr \
						--disablemorph \
						--enable-gold-morphtag \
						--disablepipeline \
						--mtag-encoding-composition-type $MTAG_COMP \
						--morph-encoding-composition-type $MORPH_COMP \
						--pos-encoding-composition-type $POS_COMP \
						--prevectors $PREVECTORS
	elif [ $TYPE = "jointAll" ]; then
		echo "Joint All Tasks"
		python jPTDP.py --dynet-seed  123456789 \
						--dynet-mem 1000 \
						--epochs $EPOCH \
						--lstmdims 128 \
						--lstmlayers 2 \
						--hidden 100 \
						--wembedding 100 \
						--cembedding 50 \
						--membedding 50 \
						--tembedding 50 \
						--pembedding 100 \
						--model "$LANG-$TYPE-$MORPH_COMP-$MTAG_COMP-$POS_COMP-model" \
						--params "$LANG-$TYPE-$MORPH_COMP-$MTAG_COMP-$POS_COMP-model.params" \
						--output "$LANG-$TYPE-$MORPH_COMP-$MTAG_COMP-$POS_COMP-test.conllu.pred" \
						--outdir ../outdir \
						--train $UDTRAIN \
						--dev $UDTEST \
						--segmentation $DATASET_VARIABLE/metu.tr \
						--disablepipeline \
						--mtag-encoding-composition-type $MTAG_COMP \
						--morph-encoding-composition-type $MORPH_COMP \
						--pos-encoding-composition-type $POS_COMP \
						--prevectors $PREVECTORS
	elif [ $TYPE = "jointAllAblationSeg" ]; then
		echo "Joint All Tasks with gold morph segs"
		python jPTDP.py --dynet-seed  123456789 \
						--dynet-mem 1000 \
						--epochs $EPOCH \
						--lstmdims 128 \
						--lstmlayers 2 \
						--hidden 100 \
						--wembedding 100 \
						--cembedding 50 \
						--membedding 50 \
						--tembedding 50 \
						--pembedding 100 \
						--model "$LANG-$TYPE-$MORPH_COMP-$MTAG_COMP-$POS_COMP-model" \
						--params "$LANG-$TYPE-$MORPH_COMP-$MTAG_COMP-$POS_COMP-model.params" \
						--output "$LANG-$TYPE-$MORPH_COMP-$MTAG_COMP-$POS_COMP-test.conllu.pred" \
						--outdir ../outdir \
						--train $UDTRAIN \
						--dev $UDTEST \
						--segmentation $DATASET_VARIABLE/metu.tr \
						--disablepipeline \
						--enable-gold-morph \
						--mtag-encoding-composition-type $MTAG_COMP \
						--morph-encoding-composition-type $MORPH_COMP \
						--pos-encoding-composition-type $POS_COMP \					
						--prevectors $PREVECTORS
	elif [ $TYPE = "jointAllAblationMorphTag" ]; then
		echo "Joint All Tasks with gold morph tags"
		python jPTDP.py --dynet-seed  123456789 \
						--dynet-mem 1000 \
						--epochs $EPOCH \
						--lstmdims 128 \
						--lstmlayers 2 \
						--hidden 100 \
						--wembedding 100 \
						--cembedding 50 \
						--membedding 50 \
						--tembedding 50 \
						--pembedding 100 \
						--model "$LANG-$TYPE-$MORPH_COMP-$MTAG_COMP-$POS_COMP-model" \
						--params "$LANG-$TYPE-$MORPH_COMP-$MTAG_COMP-$POS_COMP-model.params" \
						--output "$LANG-$TYPE-$MORPH_COMP-$MTAG_COMP-$POS_COMP-test.conllu.pred" \
						--outdir ../outdir \
						--train $UDTRAIN \
						--dev $UDTEST \
						--segmentation $DATASET_VARIABLE/metu.tr \
						--disablepipeline \
						--enable-gold-morphtag \
						--mtag-encoding-composition-type $MTAG_COMP \
						--morph-encoding-composition-type $MORPH_COMP \
						--pos-encoding-composition-type $POS_COMP \					
						--prevectors $PREVECTORS
	elif [ $TYPE = "jointAllAblationBoth" ]; then
		echo "Joint All Tasks with gold morphs and tags"
		python jPTDP.py --dynet-seed  123456789 \
						--dynet-mem 1000 \
						--epochs $EPOCH \
						--lstmdims 128 \
						--lstmlayers 2 \
						--hidden 100 \
						--wembedding 100 \
						--cembedding 50 \
						--membedding 50 \
						--tembedding 50 \
						--pembedding 100 \
						--model "$LANG-$TYPE-$MORPH_COMP-$MTAG_COMP-$POS_COMP-model" \
						--params "$LANG-$TYPE-$MORPH_COMP-$MTAG_COMP-$POS_COMP-model.params" \
						--output "$LANG-$TYPE-$MORPH_COMP-$MTAG_COMP-$POS_COMP-test.conllu.pred" \
						--outdir ../outdir \
						--train $UDTRAIN \
						--dev $UDTEST \
						--segmentation $DATASET_VARIABLE/metu.tr \
						--disablepipeline \
						--enable-gold-morphtag \
						--enable-gold-morph \
						--mtag-encoding-composition-type $MTAG_COMP \
						--morph-encoding-composition-type $MORPH_COMP \
						--pos-encoding-composition-type $POS_COMP \					
						--prevectors $PREVECTORS						
	elif [ $TYPE = "base" ]; then
		echo "Base Dependency Model"
		python jPTDP.py --dynet-seed  123456789 \
						--dynet-mem 1000 \
						--epochs $EPOCH \
						--lstmdims 128 \
						--lstmlayers 2 \
						--hidden 100 \
						--wembedding 100 \
						--cembedding 50 \
						--membedding 50 \
						--tembedding 50 \
						--pembedding 100 \
						--model "$LANG-$TYPE-$MORPH_COMP-$MTAG_COMP-$POS_COMP-model" \
						--params "$LANG-$TYPE-$MORPH_COMP-$MTAG_COMP-$POS_COMP-model.params" \
						--output "$LANG-$TYPE-$MORPH_COMP-$MTAG_COMP-$POS_COMP-test.conllu.pred" \
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




