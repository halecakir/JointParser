# -*- encoding: utf-8 -*-
from optparse import OptionParser
import pickle, utils, learner, os, os.path, time
import numpy as np

if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option("--train", dest="conll_train", help="Path to annotated CONLL train file", metavar="FILE", default="N/A")
    parser.add_option("--dev", dest="conll_dev", help="Path to annotated CONLL dev file", metavar="FILE", default="N/A")
    parser.add_option("--test", dest="conll_test", help="Path to CONLL test file", metavar="FILE", default="N/A")
    parser.add_option("--goldmorph", dest="gold_morph_path", help="Path to Morph segmentation file", metavar="FILE", default="N/A")
    parser.add_option("--output", dest="conll_test_output", help="File name for predicted output", metavar="FILE", default="N/A")
    parser.add_option("--prevectors", dest="external_embedding", help="Pre-trained vector embeddings", metavar="FILE")
    parser.add_option("--prevectype", dest="external_embedding_type", help="Pre-trained vector embeddings type", default=None)
    parser.add_option("--morphvectors", dest="external_morph_embedding", help="Pre-trained morph vector embeddings", metavar="FILE")
    parser.add_option("--morphvectype", dest="external_morph_embedding_type", help="Pre-trained morph vector embeddings type", default=None)
    parser.add_option("--params", dest="params", help="Parameters file", metavar="FILE", default="model.params")
    parser.add_option("--model", dest="model", help="Load/Save model file", metavar="FILE", default="model")
    parser.add_option("--wembedding", type="int", dest="wembedding_dims", default=100)
    parser.add_option("--cembedding", type="int", dest="cembedding_dims", default=50)
    parser.add_option("--membedding", type="int", dest="membedding_dims", default=50)
    parser.add_option("--tembedding", type="int", dest="tembedding_dims", default=50)
    parser.add_option("--pembedding", type="int", dest="pembedding_dims", default=100)
    parser.add_option("--epochs", type="int", dest="epochs", default=30)
    parser.add_option("--hidden", type="int", dest="hidden_units", default=100)
    #parser.add_option("--lr", type="float", dest="learning_rate", default=None)
    parser.add_option("--outdir", type="string", dest="output", default="results")
    parser.add_option("--activation", type="string", dest="activation", default="tanh")
    parser.add_option("--lstmlayers", type="int", dest="lstm_layers", default=2)
    parser.add_option("--lstmdims", type="int", dest="lstm_dims", default=128)
    parser.add_option("--morphdims", type="int", dest="morph_dims", default=300)
    parser.add_option("--disablemorph", action="store_false", dest="morphFlag", default=True)
    parser.add_option("--disablemorphtag", action="store_false", dest="morphTagFlag", default=True)
    parser.add_option("--disableblstm", action="store_false", dest="blstmFlag", default=True)
    parser.add_option("--disablelabels", action="store_false", dest="labelsFlag", default=True)
    parser.add_option("--predict", action="store_true", dest="predictFlag", default=False)
    parser.add_option("--bibi-lstm", action="store_false", dest="bibiFlag", default=True)
    parser.add_option("--disablecostaug", action="store_false", dest="costaugFlag", default=True)
    parser.add_option("--disablelower", action="store_false", dest="lowerCase", default=True)
    parser.add_option("--dynet-seed", type="int", dest="seed", default=0)
    parser.add_option("--dynet-mem", type="int", dest="mem", default=0)
    parser.add_option("--dynet-weight-decay", type="float", dest="weight_decay", default=0)

    (options, args) = parser.parse_args()

    #print 'Using external embedding:', options.external_embedding

    if options.predictFlag:
        with open(options.params, 'r') as paramsfp:
            words, w2i, c2i, m2i, t2i, morph_dict, pos, rels, stored_opt = pickle.load(paramsfp)
        stored_opt.external_embedding = None
        print 'Loading pre-trained model'
        parser = learner.jPosDepLearner(words, pos, rels, w2i, c2i, m2i, t2i, morph_dict, stored_opt)
        parser.Load(options.model)

        testoutpath = os.path.join(options.output, options.conll_test_output)
        print 'Predicting POS tags and parsing dependencies'
        #ts = time.time()
        #test_pred = list(parser.Predict(options.conll_test))
        #te = time.time()
        #print 'Finished in', te-ts, 'seconds.'
        #utils.write_conll(testoutpath, test_pred)

        with open(testoutpath, 'w') as fh:
            for sentence in parser.Predict(options.conll_test):
                for entry in sentence[1:]:
                    fh.write(str(entry) + '\n')
                fh.write('\n')

    else:
        print("Training file: " + options.conll_train)
        if options.conll_dev != "N/A":
            print("Development file: " + options.conll_dev)

        highestScore = 0.0
        eId = 0

        if os.path.isfile(os.path.join(options.output, options.params)) and \
                os.path.isfile(os.path.join(options.output, os.path.basename(options.model))) :

            print 'Found a previous saved model => Loading this model'
            with open(os.path.join(options.output, options.params), 'r') as paramsfp:
                words, w2i, c2i, m2i, t2i, morph_dict, pos, rels, stored_opt = pickle.load(paramsfp)
            stored_opt.external_embedding = None
            parser = learner.jPosDepLearner(words, pos, rels, w2i, c2i, m2i, t2i, morph_dict, stored_opt)
            parser.Load(os.path.join(options.output, os.path.basename(options.model)))
            parser.trainer.restart()
            if options.conll_dev != "N/A":
                utils.save_embeddings("word_emb.p",parser.morph2word(utils.get_morph_dict("turkish_new_data_gold_segmented.txt",True)))
                utils.save_embeddings("morph_emb.p",parser.morph())
                devPredSents = parser.Predict(options.conll_dev)

                count = 0
                memb_count = 0
                seg_count = 0
                lasCount = 0
                uasCount = 0
                posCount = 0
                segAcc = 0
                membAcc = 0
                poslasCount = 0
                for idSent, devSent in enumerate(devPredSents):
                    conll_devSent = [entry for entry in devSent if isinstance(entry, utils.ConllEntry)]

                    for entry in conll_devSent:
                        if entry.id <= 0:
                            continue
                        if entry.pos == entry.pred_pos and entry.parent_id == entry.pred_parent_id and entry.pred_relation == entry.relation:
                            poslasCount += 1
                        if entry.pos == entry.pred_pos:
                            posCount += 1
                        if entry.parent_id == entry.pred_parent_id and entry.pred_relation == entry.relation:
                            lasCount += 1
                        if entry.parent_id == entry.pred_parent_id:
                            uasCount += 1
                        if len(entry.seg) != 0 and len(entry.pred_seg) == len(entry.seg):
                            segAcc += np.average([(gold and pred > 0.6) or (not gold and pred < 0.6) for pred, gold in zip(entry.pred_seg, entry.seg)])
                            seg_count += 1
                        if entry.morph is not None:
                            memb_count += 1
                            membAcc += utils.percentage_arccosine_similarity(entry.morph, entry.pred_morph)
                        count += 1

                print "---\nLAS accuracy:\t%.2f" % (float(lasCount) * 100 / count)
                print "UAS accuracy:\t%.2f" % (float(uasCount) * 100 / count)
                print "POS accuracy:\t%.2f" % (float(posCount) * 100 / count)
                print "SEG accuracy:\t%.2f" % (float(segAcc) * 100 / seg_count)
                print "MEMB accuracy:\t%.2f" % (float(membAcc) / memb_count)
                print "POS&LAS:\t%.2f" % (float(poslasCount) * 100 / count)

                score = float(poslasCount) * 100 / count
                if score >= highestScore:
                    parser.Save(os.path.join(options.output, os.path.basename(options.model)))
                    highestScore = score

                print "POS&LAS of the previous saved model: %.2f" % (highestScore)

        else:
            print 'Extracting vocabulary'
            morph_dict = utils.get_morph_dict(options.gold_morph_path, options.lowerCase)
            morph_dict_2 = utils.get_morph_dict("special/90K_gold.tr", options.lowerCase)
            words, w2i, c2i, m2i, t2i, pos, rels = utils.vocab(options.conll_train,morph_dict_2)

            with open(os.path.join(options.output, options.params), 'w') as paramsfp:
                pickle.dump((words, w2i, c2i, m2i, t2i, morph_dict, pos, rels, options), paramsfp)

            #print 'Initializing joint model'
            parser = learner.jPosDepLearner(words, pos, rels, w2i, c2i, m2i, t2i, morph_dict, options)


        for epoch in xrange(5):
            print '\n-----------------\nStarting Morph2Vec epoch', epoch + 1
            parser.Train_Morph(morph_dict_2)

        for epoch in xrange(options.epochs):
            print '\n-----------------\nStarting epoch', epoch + 1

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
                utils.save_embeddings("word_emb.p",parser.morph2word(utils.get_morph_dict("turkish_new_data_gold_segmented.txt",True)))
                utils.save_embeddings("morph_emb.p",parser.morph())
                devPredSents = parser.Predict(options.conll_dev)

                count = 0
                memb_count = 0
                seg_count = 0
                lasCount = 0
                uasCount = 0
                posCount = 0
                segAcc = 0
                membAcc = 0
                poslasCount = 0
                for idSent, devSent in enumerate(devPredSents):
                    conll_devSent = [entry for entry in devSent if isinstance(entry, utils.ConllEntry)]

                    for entry in conll_devSent:
                        if entry.id <= 0:
                            continue
                        if entry.pos == entry.pred_pos and entry.parent_id == entry.pred_parent_id and entry.pred_relation == entry.relation:
                            poslasCount += 1
                        if entry.pos == entry.pred_pos:
                            posCount += 1
                        if entry.parent_id == entry.pred_parent_id and entry.pred_relation == entry.relation:
                            lasCount += 1
                        if entry.parent_id == entry.pred_parent_id:
                            uasCount += 1
                        if len(entry.seg) != 0 and len(entry.pred_seg) == len(entry.seg):
                            segAcc += np.average([(gold and pred > 0.6) or (not gold and pred < 0.6) for pred, gold in zip(entry.pred_seg, entry.seg)])
                            seg_count += 1
                        if entry.morph is not None:
                            memb_count += 1
                            membAcc += utils.percentage_arccosine_similarity(entry.morph, entry.pred_morph)
                        count += 1
                        
                print "---\nLAS accuracy:\t%.2f" % (float(lasCount) * 100 / count)
                print "UAS accuracy:\t%.2f" % (float(uasCount) * 100 / count)
                print "POS accuracy:\t%.2f" % (float(posCount) * 100 / count)
                print "SEG accuracy:\t%.2f" % (float(segAcc) * 100 / seg_count)
                print "MEMB accuracy:\t%.2f" % (float(membAcc) / memb_count)
                print "POS&LAS:\t%.2f" % (float(poslasCount) * 100 / count)
                
                score = float(poslasCount) * 100 / count
                if score >= highestScore:
                    parser.Save(os.path.join(options.output, os.path.basename(options.model)))
                    highestScore = score
                    eId = epoch + 1
                
                print "Highest POS&LAS: %.2f at epoch %d" % (highestScore, eId)

