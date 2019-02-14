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
                --model trialmodel \
                --params trialmodel.params \
                --outdir /home/huseyin/Desktop/NLP/outdir \
                --train /home/huseyin/Desktop/NLP/data/UD_Turkish-IMST/tr_imst-ud-train.conllu \
                --dev /home/huseyin/Desktop/NLP/data/UD_Turkish-IMST/tr_imst-ud-dev.conllu \
                --segmentation /home/huseyin/Desktop/NLP/data/metu.tr \
                --disablemorph \
                --disablepipeline \
                --disablemorphtag \
                --prevectors /home/huseyin/Desktop/NLP/data/Turkish/tr.vectors.xz
                #--prevectors /home/huseyin/Desktop/NLP/data/boun_full_vector 


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
                --model trialmodel \
                --params trialmodel.params \
                --outdir /home/huseyin/outdir \
                --train /home/huseyin/Data/ud-treebanks-v2.2/UD_Turkish-IMST/tr_imst-ud-train.conllu \
                --dev /home/huseyin/Data/ud-treebanks-v2.2/UD_Turkish-IMST/tr_imst-ud-test.conllu \
                --segmentation /home/huseyin/Data/metu.tr \
                --disablemorph \
                --disablepipeline \
                --disablemorphtag \
                --prevectors /home/huseyin/Data/Turkish/tr.vectors.xz