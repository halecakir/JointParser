import learner, pickle, utils

MODEL_PATH="/home/huseyin/saved-models/Turkish-jointAll-MTAG_COMP=w_sum-MORPH_COMP=w_sum-POS_COMP=w_sum-COMP_ALPHA=0.1-trialmodel"
MODEL_OPT_PATH="/home/huseyin/saved-models/Turkish-jointAll-MTAG_COMP=w_sum-MORPH_COMP=w_sum-POS_COMP=w_sum-COMP_ALPHA=0.1-trialmodel.params"

with open(MODEL_OPT_PATH, 'rb') as paramsfp:
    words, w2i, c2i, m2i, t2i, morph_dict, pos, rels, stored_opt = pickle.load(paramsfp)
    stored_opt.external_embedding = None

print('Loading pre-trained model')
parser = learner.jPosDepLearner(words, pos, rels, w2i, c2i, m2i, t2i, morph_dict, stored_opt)
parser.Load(MODEL_PATH)