# coding=utf-8
from optparse import OptionParser
import pickle, utils, learner, os, os.path, time
import numpy as np
import evaluation.conll18_ud_eval as ud_eval


def save_dependencies(dependencies, path):
    with open(path, 'w') as fh:
        for sentence in dependencies:
            for entry in sentence[1:]:
                fh.write(str(entry) + '\n')
            fh.write('\n')

def score(gold_dev, gold_pred):
    gold_ud = ud_eval.load_conllu_file(gold_dev)
    system_ud = ud_eval.load_conllu_file(gold_pred)
    evaluation = ud_eval.evaluate(gold_ud, system_ud)
    return evaluation

if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option("--train", dest="conll_train", help="Path to annotated CONLL train file", metavar="FILE", default="N/A")
    parser.add_option("--dev", dest="conll_dev", help="Path to annotated CONLL dev file", metavar="FILE", default="N/A")
    parser.add_option("--test", dest="conll_test", help="Path to CONLL test file", metavar="FILE", default="N/A")
    parser.add_option("--segmentation", dest="segmentation_path", help="Path to Morhp seqmentation file", metavar="FILE", default="N/A")
    parser.add_option("--output", dest="conll_test_output", help="File name for predicted output", metavar="FILE", default="N/A")
    parser.add_option("--prevectors", dest="external_embedding", help="Pre-trained vector embeddings", metavar="FILE")
    parser.add_option("--prevectype", dest="external_embedding_type", help="Pre-trained vector embeddings type", default=None)
    parser.add_option("--params", dest="params", help="Parameters file", metavar="FILE", default="model.params")
    parser.add_option("--model", dest="model", help="Load/Save model file", metavar="FILE", default="model")
    parser.add_option("--wembedding", type="int", dest="wembedding_dims", default=100)
    parser.add_option("--cembedding", type="int", dest="cembedding_dims", default=50)
    parser.add_option("--membedding", type="int", dest="membedding_dims", default=50)
    parser.add_option("--tembedding", type="int", dest="tembedding_dims", default=50)
    parser.add_option("--pembedding", type="int", dest="pembedding_dims", default=100)
    parser.add_option("--tagging-att-size", type="int", dest="tagging_att_size", default=20)
    parser.add_option("--epochs", type="int", dest="epochs", default=30)
    parser.add_option("--hidden", type="int", dest="hidden_units", default=100)
    #parser.add_option("--lr", type="float", dest="learning_rate", default=None)
    parser.add_option("--outdir", type="string", dest="output", default="results")
    parser.add_option("--activation", type="string", dest="activation", default="tanh")
    parser.add_option("--lstmlayers", type="int", dest="lstm_layers", default=2)
    parser.add_option("--lstmdims", type="int", dest="lstm_dims", default=128)
    parser.add_option("--disablemorph", action="store_false", dest="morphFlag", default=True)
    parser.add_option("--disablemorphtag", action="store_false", dest="morphTagFlag", default=True)
    parser.add_option("--disablepipeline", action="store_false", dest="pipeline", default=True)
    parser.add_option("--disableblstm", action="store_false", dest="blstmFlag", default=True)
    parser.add_option("--disablelabels", action="store_false", dest="labelsFlag", default=True)
    parser.add_option("--predict", action="store_true", dest="predictFlag", default=False)
    parser.add_option("--bibi-lstm", action="store_false", dest="bibiFlag", default=True)
    parser.add_option("--disablecostaug", action="store_false", dest="costaugFlag", default=True)
    parser.add_option("--disablelower", action="store_false", dest="lowerCase", default=True)
    parser.add_option("--dynet-seed", type="int", dest="seed", default=0)
    parser.add_option("--dynet-mem", type="int", dest="mem", default=0)

    (options, args) = parser.parse_args()

    #print 'Using external embedding:', options.external_embedding
    pretrained_flag = False

    if options.predictFlag:
        print("PREDICT...")
        with open(options.params, 'rb') as paramsfp:
            words, w2i, c2i, m2i, t2i, morph_dict, pos, rels, stored_opt = pickle.load(paramsfp)
        stored_opt.external_embedding = None
        print('Loading pre-trained model')
        parser = learner.jPosDepLearner(words, pos, rels, w2i, c2i, m2i, t2i, morph_dict, stored_opt)
        parser.Load(options.model)

        testoutpath = os.path.join(options.output, options.conll_test_output)
        print('Predicting POS tags and parsing dependencies')
        with open(testoutpath, 'w') as fh:
            for sentence in parser.Predict(options.conll_test):
                for entry in sentence[1:]:
                    fh.write(str(entry) + '\n')
                fh.write('\n')

    else:
        print("TRAIN...")
        print("Training file: " + options.conll_train)

        highestScore = 0.0
        eId = 0

        if os.path.isfile(os.path.join(options.output, options.params)) and \
                os.path.isfile(os.path.join(options.output, os.path.basename(options.model))) :

            print('Found a previous saved model => Loading this model')
            pretrained_flag = True
            with open(os.path.join(options.output, options.params), 'rb') as paramsfp:
                words, w2i, c2i, m2i, t2i, morph_dict, pos, rels, stored_opt = pickle.load(paramsfp)
            stored_opt.external_embedding = None
            parser = learner.jPosDepLearner(words, pos, rels, w2i, c2i, m2i, t2i, morph_dict, stored_opt)
            parser.Load(os.path.join(options.output, os.path.basename(options.model)))
            parser.trainer.restart()
            if options.conll_dev != "N/A":
                devPredSents = parser.Predict(options.conll_dev)

                count = 0
                seg_count = 0
                segAcc = 0

                for idSent, devSent in enumerate(devPredSents):
                    conll_devSent = [entry for entry in devSent if isinstance(entry, utils.ConllEntry)]

                    for entry in conll_devSent:
                        if entry.id <= 0:
                            continue
                        if options.morphFlag and len(entry.seg) != 0 and len(entry.pred_seg) == len(entry.seg):
                            segAcc += np.average([(gold and pred > 0.6) or (not gold and pred < 0.6) for pred, gold in zip(entry.pred_seg, entry.seg)])
                            seg_count += 1
                        count += 1
                #save predicted sentences
                save_dependencies(devPredSents, "temp/pred.conllu")
                evaluation = score(options.conll_dev, "temp/pred.conllu")

                las = evaluation["LAS"].f1 * 100 
                uas = evaluation["UAS"].f1 * 100 
                upos = evaluation["UPOS"].f1 * 100
                feats = evaluation["FEATS"].f1 * 100
                f1_uas_las = 2 * ((las * uas)/(uas + las))
                print("---\nLAS accuracy:\t%.2f" % (las))
                print("UAS accuracy:\t%.2f" % (uas))
                print("POS accuracy:\t%.2f" % (upos))
                if options.morphFlag:
                    print("SEG accuracy:\t%.2f" % (float(segAcc) * 100 / seg_count))
                if options.morphTagFlag:
                    print("TAG accuracy:\t%.2f" % (feats))

                if f1_uas_las >= highestScore:
                    parser.Save(os.path.join(options.output, os.path.basename(options.model)))
                    highestScore = f1_uas_las
                    eId = epoch + 1
                    print("Saved Highest POS&LAS: %.2f at epoch %d" % (highestScore, eId))

                print("Highest POS&LAS: %.2f at epoch %d" % (highestScore, eId))

        else:
            print('Extracting vocabulary')
            morph_dict = utils.get_morph_dict(options.segmentation_path, options.lowerCase)
            words, w2i, c2i, m2i, t2i, pos, rels = utils.vocab(options.conll_train,morph_dict)

            with open(os.path.join(options.output, options.params), 'wb') as paramsfp:
                pickle.dump((words, w2i, c2i, m2i, t2i, morph_dict, pos, rels, options), paramsfp)

            #print 'Initializing joint model'
            parser = learner.jPosDepLearner(words, pos, rels, w2i, c2i, m2i, t2i, morph_dict, options)

        if options.pipeline and options.morphFlag and not pretrained_flag:
            for epoch in range(5):
                print('\n-----------------\nStarting Morph2Vec epoch', epoch + 1)
                parser.Train_Morph()

            parser.initialize()

        for epoch in range(options.epochs):
            print('\n-----------------\nStarting epoch', epoch + 1)

            if epoch % 10 == 0:
                if epoch == 0:
                    parser.trainer.restart(learning_rate=0.001)
                elif epoch == 10:
                    parser.trainer.restart(learning_rate=0.0005)
                else:
                    parser.trainer.restart(learning_rate=0.00025)

            parser.Train(options.conll_train)

            if options.conll_dev == "N/A":
                parser.Save(os.path.join(options.output, os.path.basename(options.model)))

            else:
                devPredSents = parser.Predict(options.conll_dev)

                count = 0
                seg_count = 0
                segAcc = 0

                for idSent, devSent in enumerate(devPredSents):
                    conll_devSent = [entry for entry in devSent if isinstance(entry, utils.ConllEntry)]

                    for entry in conll_devSent:
                        if entry.id <= 0:
                            continue
                        if options.morphFlag and len(entry.seg) != 0 and len(entry.pred_seg) == len(entry.seg):
                            segAcc += np.average([(gold and pred > 0.6) or (not gold and pred < 0.6) for pred, gold in zip(entry.pred_seg, entry.seg)])
                            seg_count += 1
                        count += 1
                #save predicted sentences
                save_dependencies(devPredSents, "temp/pred.conllu")
                evaluation = score(options.conll_dev, "temp/pred.conllu")

                las = evaluation["LAS"].f1 * 100 
                uas = evaluation["UAS"].f1 * 100 
                upos = evaluation["UPOS"].f1 * 100
                feats = evaluation["FEATS"].f1 * 100
                f1_uas_las = 2 * ((las * uas)/(uas + las))
                print("---\nLAS accuracy:\t%.2f" % (las))
                print("UAS accuracy:\t%.2f" % (uas))
                print("POS accuracy:\t%.2f" % (upos))
                if options.morphFlag:
                    print("SEG accuracy:\t%.2f" % (float(segAcc) * 100 / seg_count))
                if options.morphTagFlag:
                    print("TAG accuracy:\t%.2f" % (feats))

                if f1_uas_las >= highestScore:
                    parser.Save(os.path.join(options.output, os.path.basename(options.model)))
                    highestScore = f1_uas_las
                    eId = epoch + 1
                    print("Saved Highest POS&LAS: %.2f at epoch %d" % (highestScore, eId))

                print("Highest POS&LAS: %.2f at epoch %d" % (highestScore, eId))
