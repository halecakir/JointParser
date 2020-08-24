# coding=utf-8
from optparse import OptionParser
import pickle, utils, learner, os, os.path, time
import numpy as np
import evaluation.conll18_ud_eval as ud_eval

from comet_ml import Experiment

# check
if os.environ.get("PYTHONHASHSEED") != "0":
    raise Exception("You must set PYTHONHASHSEED=0 when starting the Jupyter server to get reproducible results.")
    

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
    parser.add_option("--type", dest="type", help="Experiment Type")
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
    parser.add_option("--mtag-encoding-composition-type", type="string", dest="mtag_encoding_composition_type", default="None")
    parser.add_option("--morph-encoding-composition-type", type="string", dest="morph_encoding_composition_type", default="None")
    parser.add_option("--pos-encoding-composition-type", type="string", dest="pos_encoding_composition_type", default="None")
    parser.add_option("--encoding-composition-alpha", type="float", dest="encoding_composition_alpha", default=0.5)
    parser.add_option("--outdir", type="string", dest="output", default="results")
    parser.add_option("--activation", type="string", dest="activation", default="tanh")
    parser.add_option("--lstmlayers", type="int", dest="lstm_layers", default=2)
    parser.add_option("--lstmdims", type="int", dest="lstm_dims", default=128)
    parser.add_option("--disablemorph", action="store_false", dest="morphFlag", default=True)
    parser.add_option("--enable-gold-morph", action="store_true", dest="goldMorphFlag", default=False)
    parser.add_option("--disablemorphtag", action="store_false", dest="morphTagFlag", default=True)
    parser.add_option("--enable-gold-morphtag", action="store_true", dest="goldMorphTagFlag", default=False)
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

    # Create an experiment
    experiment = Experiment(api_key="nLqFerDLnwvCiAptbL4u0FZIj",
                        project_name="thesis", workspace="halecakir")

    #experiment.log_parameters(vars(options))
    #print 'Using external embedding:', options.external_embedding
    pretrained_flag = False

    if options.predictFlag:
        print("PREDICT...")
        with open(os.path.join(options.output, options.params), 'rb') as paramsfp:
            words, w2i, c2i, m2i, t2i, morph_dict, pos, rels, stored_opt = pickle.load(paramsfp)
        stored_opt.external_embedding = None

        print('Loading pre-trained model')
        parser = learner.jPosDepLearner(words, pos, rels, w2i, c2i, m2i, t2i, morph_dict, stored_opt)

        parser.Load(os.path.join(options.output, options.model))
        
        testoutpath = os.path.join(options.output, options.conll_test_output)
        print('Predicting POS tags and parsing dependencies')
        prediction = list(parser.Predict(options.conll_test))
        with open(testoutpath, 'w') as fh:
            for sentence in prediction:
                for entry in sentence[1:]:
                    fh.write(str(entry) + '\n')
                fh.write('\n')
        if options.conll_test != "N/A":                
            count = 0
            seg_count = 0
            segAcc = 0

            for idSent, sentence in enumerate(prediction):
                conll_sent = [entry for entry in sentence if isinstance(entry, utils.ConllEntry)]
                for entry in conll_sent:
                    if entry.id <= 0:
                        continue
                    if options.morphFlag and len(entry.seg) != 0 and len(entry.pred_seg) == len(entry.seg):
                        segAcc += np.average([(gold and pred > 0.6) or (not gold and pred < 0.6) for pred, gold in zip(entry.pred_seg, entry.seg)])
                        seg_count += 1
                    count += 1               
            evaluation = score(options.conll_test, testoutpath)
            outStr = "{},{},{},{},{},".format(options.type,
                                    stored_opt.morph_encoding_composition_type,
                                    stored_opt.mtag_encoding_composition_type,
                                    stored_opt.pos_encoding_composition_type,
                                    stored_opt.encoding_composition_alpha)
            las = evaluation["LAS"].f1 * 100 
            uas = evaluation["UAS"].f1 * 100 
            upos = evaluation["UPOS"].f1 * 100
            feats = evaluation["UFeats"].f1 * 100
            outStr += "{:.2f},{:.2f},{:.2f}".format(las, uas, upos)

            experiment.log_metric("LAS", las, step=0)
            experiment.log_metric("UAS", uas, step=0)
            experiment.log_metric("UPOS", upos, step=0)
            if stored_opt.morphTagFlag:
                outStr += ",{:.2f}".format(feats)
                experiment.log_metric("UFeats", feats, step=0)
            if stored_opt.morphFlag:
                seg = float(segAcc) * 100 / seg_count
                outStr += ",{:.2f}".format(seg)
                experiment.log_metric("SEG", seg, step=0)
            print(outStr)

    else:
        print("TRAIN...")
        print("Training file: " + options.conll_train)

        highestScore = 0.0
        eId = 0

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
            experiment.set_epoch(epoch+1)
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
                print("Predict step is finished")
                count = 0
                seg_count = 0
                segAcc = 0
                dev_sentences = []
                for idSent, devSent in enumerate(devPredSents):
                    conll_devSent = [entry for entry in devSent if isinstance(entry, utils.ConllEntry)]
                    dev_sentences.append(devSent)
                    for entry in conll_devSent:
                        if entry.id <= 0:
                            continue
                        if options.morphFlag and len(entry.seg) != 0 and len(entry.pred_seg) == len(entry.seg):
                            segAcc += np.average([(gold and pred > 0.6) or (not gold and pred < 0.6) for pred, gold in zip(entry.pred_seg, entry.seg)])
                            seg_count += 1
                        count += 1
                #save predicted sentences
                save_dependencies(dev_sentences, os.path.join("temp", options.conll_test_output))
                evaluation = score(options.conll_dev, os.path.join("temp", options.conll_test_output))

                las = evaluation["LAS"].f1 * 100 
                uas = evaluation["UAS"].f1 * 100 
                upos = evaluation["UPOS"].f1 * 100
                feats = evaluation["UFeats"].f1 * 100
                f1_uas_las = 2 * ((las * uas)/(uas + las))
                print("---\nLAS accuracy:\t%.2f" % (las))
                print("UAS accuracy:\t%.2f" % (uas))
                print("POS accuracy:\t%.2f" % (upos))
                # Log metrics to Comet.ml
                experiment.log_metric("LAS", las, step=epoch+1)
                experiment.log_metric("UAS", uas, step=epoch+1)
                experiment.log_metric("UPOS", upos, step=epoch+1)
                experiment.log_metric("F1_UAS_LAS", f1_uas_las, step=epoch+1)

                if options.morphFlag:
                    seg = (float(segAcc) * 100 / seg_count)
                    print("SEG accuracy:\t%.2f" % seg)
                    experiment.log_metric("SEG", seg, step=epoch+1)
                if options.morphTagFlag:
                    print("TAG accuracy:\t%.2f" % (feats))
                    experiment.log_metric("UFeats", feats, step=epoch+1)


                if f1_uas_las >= highestScore:
                    parser.Save(os.path.join(options.output, os.path.basename(options.model)))
                    highestScore = f1_uas_las
                    eId = epoch + 1
                    print("Saved Highest POS&LAS: %.2f at epoch %d" % (highestScore, eId))

                print("Highest POS&LAS: %.2f at epoch %d" % (highestScore, eId))
                