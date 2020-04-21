#!/bin/bash

if [ $# -ne 9 ]; then
    echo 'Usage : sh train.sh EXPERIMENT_TYPE UDTYPE PREDICT/TRAIN LANGUAGE EPOCHNUM MTAG_COMP MORPH_COMP POS_COMP COMP_ALPHA'
    echo 'Example: sh train.sh jointAll 2.2 train Turkish 15 w_sum w_sum w_sum 0.3'
    exit 0
fi

if [[ -z "${DATASET}" ]]; then
  echo "DATASET environment variable is undefined"
else
  DATASET="${DATASET}"
  echo "DATASET located in ${DATASET}"
  
fi


#Create outdir if not exist
COMMIT_ID=`git rev-parse HEAD`
OUTDIR="../outdir/$COMMIT_ID"
mkdir -p $OUTDIR

TYPE=$1
UDTYPE=$2
PREDICT=$3
LANG=$4
EPOCH=$5
MTAG_COMP=$6
MORPH_COMP=$7
POS_COMP=$8
COMP_ALPHA=$9

UDTRAIN=""
UDTEST=""
PREVECTORS=""

#Set Python seed
export PYTHONHASHSEED=0

echo "UD_TYPE is $UDTYPE"

if [ $LANG = "Turkish" ]; then
	UDTRAIN="$DATASET/ud-treebanks-v$UDTYPE/UD_Turkish-IMST/tr_imst-ud-train.conllu"
	UDTEST="$DATASET/ud-treebanks-v$UDTYPE/UD_Turkish-IMST/tr_imst-ud-test.conllu"
	PREVECTORS="$DATASET/$LANG/tr.vectors.xz"
elif [ $LANG = "Finnish" ]; then
	UDTRAIN="$DATASET/ud-treebanks-v$UDTYPE/UD_Finnish-TDT/fi_tdt-ud-train.conllu"
	UDTEST="$DATASET/ud-treebanks-v$UDTYPE/UD_Finnish-TDT/fi_tdt-ud-test.conllu"
	PREVECTORS="$DATASET/$LANG/fi.vectors.xz"
elif [ $LANG = "Hungarian" ]; then
	UDTRAIN="$DATASET/ud-treebanks-v$UDTYPE/UD_Hungarian-Szeged/hu_szeged-ud-train.conllu"
	UDTEST="$DATASET/ud-treebanks-v$UDTYPE/UD_Hungarian-Szeged/hu_szeged-ud-test.conllu"
	PREVECTORS="$DATASET/$LANG/hu.vectors.xz"
elif [ $LANG = "Czech" ]; then
	UDTRAIN="$DATASET/ud-treebanks-v$UDTYPE/UD_Czech-PDT/cs_pdt-ud-train.conllu"
	UDTEST="$DATASET/ud-treebanks-v$UDTYPE/UD_Czech-PDT/cs_pdt-ud-test.conllu"
	PREVECTORS="$DATASET/$LANG/cs.vectors.xz"
fi

if [ $PREDICT = "predict" ]; then
	echo "Predicting..."
    python jPTDP.py  	--type $TYPE \
						--dynet-seed  123456789 \
            			--dynet-mem 1000 \--predict \
				        --model "$LANG-$TYPE-MTAG_COMP=$MTAG_COMP-MORPH_COMP=$MORPH_COMP-POS_COMP=$POS_COMP-COMP_ALPHA=$COMP_ALPHA-trialmodel" \
				        --params "$LANG-$TYPE-MTAG_COMP=$MTAG_COMP-MORPH_COMP=$MORPH_COMP-POS_COMP=$POS_COMP-COMP_ALPHA=$COMP_ALPHA-trialmodel.params" \
				        --outdir $OUTDIR \
				        --train $UDTRAIN \
				        --test $UDTEST \
						--output "$LANG-$TYPE-MTAG_COMP=$MTAG_COMP-MORPH_COMP=$MORPH_COMP-POS_COMP=$POS_COMP-COMP_ALPHA=$COMP_ALPHA-test.conllu.pred" \
						--mtag-encoding-composition-type $MTAG_COMP \
						--encoding-composition-alpha $COMP_ALPHA \
					    --segmentation $DATASET/metu.tr \
						--morph-encoding-composition-type $MORPH_COMP \
						--mtag-encoding-composition-type $MTAG_COMP \
						--pos-encoding-composition-type $POS_COMP \
					    --prevectors $PREVECTORS 
else
	echo "Training..."
    if [ $TYPE = "seg" ]; then
		echo "Only Segmentation"
		python jPTDP.py --type $TYPE \
						--dynet-seed  123456789 \
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
						--model "$LANG-$TYPE-MTAG_COMP=$MTAG_COMP-MORPH_COMP=$MORPH_COMP-POS_COMP=$POS_COMP-COMP_ALPHA=$COMP_ALPHA-trialmodel" \
						--params "$LANG-$TYPE-MTAG_COMP=$MTAG_COMP-MORPH_COMP=$MORPH_COMP-POS_COMP=$POS_COMP-COMP_ALPHA=$COMP_ALPHA-trialmodel.params" \
						--output "$LANG-$TYPE-MTAG_COMP=$MTAG_COMP-MORPH_COMP=$MORPH_COMP-POS_COMP=$POS_COMP-COMP_ALPHA=$COMP_ALPHA-test.conllu.pred" \
						--outdir $OUTDIR \
						--train $UDTRAIN \
						--dev $UDTEST \
						--segmentation $DATASET/metu.tr \
						--disablemorphtag \
						--disablepipeline \
						--morph-encoding-composition-type $MORPH_COMP \
						--mtag-encoding-composition-type $MTAG_COMP \
						--pos-encoding-composition-type $POS_COMP \						
						--prevectors $PREVECTORS
    elif [ $TYPE = "segAblation" ]; then
		echo "Only Gold Segmentation"
		python jPTDP.py --type $TYPE \
						--dynet-seed  123456789 \
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
						--model "$LANG-$TYPE-MTAG_COMP=$MTAG_COMP-MORPH_COMP=$MORPH_COMP-POS_COMP=$POS_COMP-COMP_ALPHA=$COMP_ALPHA-trialmodel" \
						--params "$LANG-$TYPE-MTAG_COMP=$MTAG_COMP-MORPH_COMP=$MORPH_COMP-POS_COMP=$POS_COMP-COMP_ALPHA=$COMP_ALPHA-trialmodel.params" \
						--output "$LANG-$TYPE-MTAG_COMP=$MTAG_COMP-MORPH_COMP=$MORPH_COMP-POS_COMP=$POS_COMP-COMP_ALPHA=$COMP_ALPHA-test.conllu.pred" \
						--outdir $OUTDIR \
						--train $UDTRAIN \
						--dev $UDTEST \
						--segmentation $DATASET/metu.tr \
						--enable-gold-morph \
						--disablemorphtag \
						--disablepipeline \
						--morph-encoding-composition-type $MORPH_COMP \
						--mtag-encoding-composition-type $MTAG_COMP \
						--pos-encoding-composition-type $POS_COMP \						
						--prevectors $PREVECTORS						
	elif [ $TYPE = "morphTag" ]; then
		echo "Only Morph Tagging"
		python jPTDP.py --type $TYPE \
						--dynet-seed  123456789 \
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
						--model "$LANG-$TYPE-MTAG_COMP=$MTAG_COMP-MORPH_COMP=$MORPH_COMP-POS_COMP=$POS_COMP-COMP_ALPHA=$COMP_ALPHA-trialmodel" \
						--params "$LANG-$TYPE-MTAG_COMP=$MTAG_COMP-MORPH_COMP=$MORPH_COMP-POS_COMP=$POS_COMP-COMP_ALPHA=$COMP_ALPHA-trialmodel.params" \
						--output "$LANG-$TYPE-MTAG_COMP=$MTAG_COMP-MORPH_COMP=$MORPH_COMP-POS_COMP=$POS_COMP-COMP_ALPHA=$COMP_ALPHA-test.conllu.pred" \
						--outdir $OUTDIR \
						--train $UDTRAIN \
						--dev $UDTEST \
						--segmentation $DATASET/metu.tr \
						--disablemorph \
						--disablepipeline \
						--morph-encoding-composition-type $MORPH_COMP \
						--mtag-encoding-composition-type $MTAG_COMP \
						--pos-encoding-composition-type $POS_COMP \
						--encoding-composition-alpha $COMP_ALPHA \
						--prevectors $PREVECTORS
	elif [ $TYPE = "morphTagAblation" ]; then
		echo "Only Gold Morph Tagging"
		python jPTDP.py --type $TYPE \
						--dynet-seed  123456789 \
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
						--model "$LANG-$TYPE-MTAG_COMP=$MTAG_COMP-MORPH_COMP=$MORPH_COMP-POS_COMP=$POS_COMP-COMP_ALPHA=$COMP_ALPHA-trialmodel" \
						--params "$LANG-$TYPE-MTAG_COMP=$MTAG_COMP-MORPH_COMP=$MORPH_COMP-POS_COMP=$POS_COMP-COMP_ALPHA=$COMP_ALPHA-trialmodel.params" \
						--output "$LANG-$TYPE-MTAG_COMP=$MTAG_COMP-MORPH_COMP=$MORPH_COMP-POS_COMP=$POS_COMP-COMP_ALPHA=$COMP_ALPHA-test.conllu.pred" \
						--outdir $OUTDIR \
						--train $UDTRAIN \
						--dev $UDTEST \
						--segmentation $DATASET/metu.tr \
						--disablemorph \
						--enable-gold-morphtag \
						--disablepipeline \
						--morph-encoding-composition-type $MORPH_COMP \
						--mtag-encoding-composition-type $MTAG_COMP \
						--pos-encoding-composition-type $POS_COMP \
						--encoding-composition-alpha $COMP_ALPHA \
						--prevectors $PREVECTORS
	elif [ $TYPE = "jointAll" ]; then
		echo "Joint All Tasks"
		python jPTDP.py --type $TYPE \
						--dynet-seed  123456789 \
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
						--model "$LANG-$TYPE-MTAG_COMP=$MTAG_COMP-MORPH_COMP=$MORPH_COMP-POS_COMP=$POS_COMP-COMP_ALPHA=$COMP_ALPHA-trialmodel" \
						--params "$LANG-$TYPE-MTAG_COMP=$MTAG_COMP-MORPH_COMP=$MORPH_COMP-POS_COMP=$POS_COMP-COMP_ALPHA=$COMP_ALPHA-trialmodel.params" \
						--output "$LANG-$TYPE-MTAG_COMP=$MTAG_COMP-MORPH_COMP=$MORPH_COMP-POS_COMP=$POS_COMP-COMP_ALPHA=$COMP_ALPHA-test.conllu.pred" \
						--outdir $OUTDIR \
						--train $UDTRAIN \
						--dev $UDTEST \
						--segmentation $DATASET/metu.tr \
						--disablepipeline \
						--morph-encoding-composition-type $MORPH_COMP \
						--mtag-encoding-composition-type $MTAG_COMP \
						--pos-encoding-composition-type $POS_COMP \
						--encoding-composition-alpha $COMP_ALPHA \
						--prevectors $PREVECTORS
	elif [ $TYPE = "jointAllAblationSeg" ]; then
		echo "Joint All Tasks with gold morph segs"
		python jPTDP.py --type $TYPE \
						--dynet-seed  123456789 \
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
						--model "$LANG-$TYPE-MTAG_COMP=$MTAG_COMP-MORPH_COMP=$MORPH_COMP-POS_COMP=$POS_COMP-COMP_ALPHA=$COMP_ALPHA-trialmodel" \
						--params "$LANG-$TYPE-MTAG_COMP=$MTAG_COMP-MORPH_COMP=$MORPH_COMP-POS_COMP=$POS_COMP-COMP_ALPHA=$COMP_ALPHA-trialmodel.params" \
						--output "$LANG-$TYPE-MTAG_COMP=$MTAG_COMP-MORPH_COMP=$MORPH_COMP-POS_COMP=$POS_COMP-COMP_ALPHA=$COMP_ALPHA-test.conllu.pred" \
						--outdir $OUTDIR \
						--train $UDTRAIN \
						--dev $UDTEST \
						--segmentation $DATASET/metu.tr \
						--disablepipeline \
						--enable-gold-morph \
						--morph-encoding-composition-type $MORPH_COMP \
						--mtag-encoding-composition-type $MTAG_COMP \
						--pos-encoding-composition-type $POS_COMP \
						--encoding-composition-alpha $COMP_ALPHA \						
						--prevectors $PREVECTORS
	elif [ $TYPE = "jointAllAblationMorphTag" ]; then
		echo "Joint All Tasks with gold morph tags"
		python jPTDP.py --type $TYPE \
						--dynet-seed  123456789 \
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
						--model "$LANG-$TYPE-MTAG_COMP=$MTAG_COMP-MORPH_COMP=$MORPH_COMP-POS_COMP=$POS_COMP-COMP_ALPHA=$COMP_ALPHA-trialmodel" \
						--params "$LANG-$TYPE-MTAG_COMP=$MTAG_COMP-MORPH_COMP=$MORPH_COMP-POS_COMP=$POS_COMP-COMP_ALPHA=$COMP_ALPHA-trialmodel.params" \
						--output "$LANG-$TYPE-MTAG_COMP=$MTAG_COMP-MORPH_COMP=$MORPH_COMP-POS_COMP=$POS_COMP-COMP_ALPHA=$COMP_ALPHA-test.conllu.pred" \
						--outdir $OUTDIR \
						--train $UDTRAIN \
						--dev $UDTEST \
						--segmentation $DATASET/metu.tr \
						--disablepipeline \
						--enable-gold-morphtag \
						--morph-encoding-composition-type $MORPH_COMP \
						--mtag-encoding-composition-type $MTAG_COMP \
						--pos-encoding-composition-type $POS_COMP \
						--encoding-composition-alpha $COMP_ALPHA \						
						--prevectors $PREVECTORS
	elif [ $TYPE = "jointAllAblationBoth" ]; then
		echo "Joint All Tasks with gold morphs and tags"
		python jPTDP.py --type $TYPE \
						--dynet-seed  123456789 \
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
						--model "$LANG-$TYPE-MTAG_COMP=$MTAG_COMP-MORPH_COMP=$MORPH_COMP-POS_COMP=$POS_COMP-COMP_ALPHA=$COMP_ALPHA-trialmodel" \
						--params "$LANG-$TYPE-MTAG_COMP=$MTAG_COMP-MORPH_COMP=$MORPH_COMP-POS_COMP=$POS_COMP-COMP_ALPHA=$COMP_ALPHA-trialmodel.params" \
						--output "$LANG-$TYPE-MTAG_COMP=$MTAG_COMP-MORPH_COMP=$MORPH_COMP-POS_COMP=$POS_COMP-COMP_ALPHA=$COMP_ALPHA-test.conllu.pred" \
						--outdir $OUTDIR \
						--train $UDTRAIN \
						--dev $UDTEST \
						--segmentation $DATASET/metu.tr \
						--disablepipeline \
						--enable-gold-morphtag \
						--enable-gold-morph \
						--morph-encoding-composition-type $MORPH_COMP \
						--mtag-encoding-composition-type $MTAG_COMP \
						--pos-encoding-composition-type $POS_COMP \
						--encoding-composition-alpha $COMP_ALPHA \						
						--prevectors $PREVECTORS						
	elif [ $TYPE = "base" ]; then
		echo "Base Dependency Model"
		python jPTDP.py --type $TYPE \
						--dynet-seed  123456789 \
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
						--model "$LANG-$TYPE-MTAG_COMP=$MTAG_COMP-MORPH_COMP=$MORPH_COMP-POS_COMP=$POS_COMP-COMP_ALPHA=$COMP_ALPHA-trialmodel" \
						--params "$LANG-$TYPE-MTAG_COMP=$MTAG_COMP-MORPH_COMP=$MORPH_COMP-POS_COMP=$POS_COMP-COMP_ALPHA=$COMP_ALPHA-trialmodel.params" \
						--output "$LANG-$TYPE-MTAG_COMP=$MTAG_COMP-MORPH_COMP=$MORPH_COMP-POS_COMP=$POS_COMP-COMP_ALPHA=$COMP_ALPHA-test.conllu.pred" \
						--outdir $OUTDIR \
						--train $UDTRAIN \
						--dev $UDTEST \
						--segmentation $DATASET/metu.tr \
						--disablemorph \
						--disablepipeline \
						--disablemorphtag \
						--morph-encoding-composition-type $MORPH_COMP \
						--mtag-encoding-composition-type $MTAG_COMP \
						--pos-encoding-composition-type $POS_COMP \
						--prevectors $PREVECTORS
	else
	   echo "Invalid Experiment Type"
	   echo "Valid types: seg, morphTag, jointAll, and base" 
	fi
fi




