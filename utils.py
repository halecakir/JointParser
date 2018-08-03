# coding=utf-8
from collections import Counter
import os, re, codecs
from gensim.models import KeyedVectors
from gensim.models.wrappers import FastText
import pickle
import numpy as np

class ConllEntry:
    def __init__(self, id, form, lemma, pos, xpos, feats=None, parent_id=None, relation=None, deps=None, misc=None):
        self.id = id
        self.form = form
        self.norm = normalize(form)
        self.xpos = xpos
        self.pos = pos
        self.parent_id = parent_id
        self.relation = relation

        self.lemma = lemma
        self.feats = feats
        self.deps = deps
        self.misc = misc

        self.pred_parent_id = None
        self.pred_relation = None
        self.pred_pos = None

        self.idChars = []
        self.idMorphs = []

        self.seq_vec = []
        self.seq_emb = []

    def __str__(self):
        values = [str(self.id), self.form, self.lemma, self.pred_pos, self.xpos, self.feats,
                  str(self.pred_parent_id) if self.pred_parent_id is not None else None, self.pred_relation, self.deps,
                  self.misc]
        return '\t'.join(['_' if v is None else v for v in values])


def vocab(conll_path, morph_dict_array):
    wordsCount = Counter()
    posCount = Counter()
    relCount = Counter()

    # Character vocabulary
    c2i = {}
    c2i["_UNK"] = 0  # unk char
    c2i["<w>"] = 1  # word start
    c2i["</w>"] = 2  # word end index
    c2i["NUM"] = 3
    c2i["EMAIL"] = 4
    c2i["URL"] = 5

    m2i = {}
    m2i["UNK"] = 0
    m2i["<w>"] = 1
    m2i["</w>"] = 2

    root = ConllEntry(0, '*root*', '*root*', 'ROOT-POS', 'ROOT-CPOS', '_', -1, 'rroot', '_', '_')
    root.idChars = [1, 2]
    root.idMorphs = [[1, 2]]
    tokens = [root]

    #create morpheme indexes out of morpheme dictionary
    all_morphs = []
    for word in morph_dict_array.keys():
        for morphs in morph_dict_array[word]:
            all_morphs += morphs
    all_morphs = list(set(all_morphs))
    for idx in xrange(len(all_morphs)):
        m2i[all_morphs[idx]] = idx+3

    for line in open(conll_path, 'r'):
        tok = line.strip().split('\t')
        if not tok or line.strip() == '':
            if len(tokens) > 1:
                wordsCount.update([node.norm for node in tokens if isinstance(node, ConllEntry)])
                posCount.update([node.pos for node in tokens if isinstance(node, ConllEntry)])
                relCount.update([node.relation for node in tokens if isinstance(node, ConllEntry)])
            tokens = [root]
        else:
            if line[0] == '#' or '-' in tok[0] or '.' in tok[0]:
                tokens.append(line.strip())
            else:
                entry = ConllEntry(int(tok[0]), tok[1], tok[2], tok[3], tok[4], tok[5],
                                   int(tok[6]) if tok[6] != '_' else -1, tok[7], tok[8], tok[9])

                if entry.norm == 'NUM':
                    entry.idChars = [1, 3, 2]
                elif entry.norm == 'EMAIL':
                    entry.idChars = [1, 4, 2]
                elif entry.norm == 'URL':
                    entry.idChars = [1, 5, 2]
                else:
                    chars_of_word = [1]
                    for char in tok[1]:
                        if char not in c2i:
                            c2i[char] = len(c2i)
                        chars_of_word.append(c2i[char])
                    chars_of_word.append(2)
                    entry.idChars = chars_of_word

                morphs_of_word = []
                if entry.norm in morph_dict_array:
                    for idx in xrange(len(morph_dict_array[entry.norm])):
                        morphs_of_word_instance = []
                        morphs_of_word_instance.append(m2i["<w>"])
                        morph_seq = morph_dict_array[entry.norm][idx]
                        for morph in morph_seq:
                            if morph not in m2i:
                                morphs_of_word_instance.append(m2i["UNK"])
                            else:
                                morphs_of_word_instance.append(m2i[morph])
                        morphs_of_word_instance.append(m2i["<w>"])
                        morphs_of_word.append(morphs_of_word_instance)
                elif entry.norm in m2i:
                    morphs_of_word = [[m2i["<w>"],m2i[entry.norm],m2i["</w>"]]]
                else:
                    morphs_of_word = [[m2i["<w>"],m2i["UNK"],m2i["</w>"]]]
                entry.idMorphs = morphs_of_word

                tokens.append(entry)

    if len(tokens) > 1:
        wordsCount.update([node.norm for node in tokens if isinstance(node, ConllEntry)])
        posCount.update([node.pos for node in tokens if isinstance(node, ConllEntry)])
        relCount.update([node.relation for node in tokens if isinstance(node, ConllEntry)])

    return (wordsCount, {w: i for i, w in enumerate(wordsCount.keys())}, c2i, m2i, posCount.keys(), relCount.keys())


def read_conll(fh, c2i, morph_dict_array, m2i):
    # Character vocabulary
    root = ConllEntry(0, '*root*', '*root*', 'ROOT-POS', 'ROOT-CPOS', '_', -1, 'rroot', '_', '_')
    root.idChars = [1, 2]
    root.idMorphs = [[1, 2]]
    tokens = [root]

    for line in fh:
        tok = line.strip().split('\t')
        if not tok or line.strip() == '':
            if len(tokens) > 1: yield tokens
            tokens = [root]
        else:
            if line[0] == '#' or '-' in tok[0] or '.' in tok[0]:
                tokens.append(line.strip())
            else:
                entry = ConllEntry(int(tok[0]), tok[1], tok[2], tok[3], tok[4], tok[5],
                                   int(tok[6]) if tok[6] != '_' else -1, tok[7], tok[8], tok[9])

                if entry.norm == 'NUM':
                    entry.idChars = [1, 3, 2]
                elif entry.norm == 'EMAIL':
                    entry.idChars = [1, 4, 2]
                elif entry.norm == 'URL':
                    entry.idChars = [1, 5, 2]
                else:
                    if entry.norm == "”" or entry.norm == "’":
                        tok[1] = "''"
                        entry.norm = '"'
                    if entry.norm == "“" or entry.norm == "‘":
                        tok[1] = "``"
                        entry.norm = '"'
                    if "’" in entry.norm:
                        entry.norm = re.sub(r"’", "'", entry.norm)
                        tok[1] = entry.norm
                    if entry.norm == "—":
                        entry.norm = "-"
                        tok[1] = "-"
                        
                    chars_of_word = [1]
                    for char in tok[1]:
                        if char in c2i:
                            chars_of_word.append(c2i[char])
                        else:
                            chars_of_word.append(0)
                    chars_of_word.append(2)
                    entry.idChars = chars_of_word

                morphs_of_word = []
                if entry.norm in morph_dict_array:
                    for idx in xrange(len(morph_dict_array[entry.norm])):
                        morphs_of_word_instance = []
                        morphs_of_word_instance.append(m2i["<w>"])
                        morph_seq = morph_dict_array[entry.norm][idx]
                        for morph in morph_seq:
                            if morph not in m2i:
                                morphs_of_word_instance.append(m2i["UNK"])
                            else:
                                morphs_of_word_instance.append(m2i[morph])
                        morphs_of_word_instance.append(m2i["<w>"])
                        morphs_of_word.append(morphs_of_word_instance)
                elif entry.norm in m2i:
                    morphs_of_word = [[m2i["<w>"],m2i[entry.norm],m2i["</w>"]]]
                else:
                    morphs_of_word = [[m2i["<w>"],m2i["UNK"],m2i["</w>"]]]
                entry.idMorphs = morphs_of_word

                tokens.append(entry)

    if len(tokens) > 1:
        yield tokens


def write_conll(fn, conll_gen):
    with open(fn, 'w') as fh:
        for sentence in conll_gen:
            for entry in sentence[1:]:
                fh.write(str(entry) + '\n')
            fh.write('\n')

numberRegex = re.compile("[0-9]+|[0-9]+\\.[0-9]+|[0-9]+[0-9,]+");
def normalize(word):
    if numberRegex.match(word):
        return 'NUM'
    else:
        w = word.lower()
        w = re.sub(r".+@.+", "EMAIL", w)
        w = re.sub(r"(https?://|www\.).*", "URL", w)
        w = re.sub(r"``", '"', w)
        w = re.sub(r"''", '"', w)
        return w

#try:
#    import lzma
#except ImportError:
#    from backports import lzma

def load_embeddings_file(file_name, lower=False, type=None):
    if type == None:
        file_type = file_name.rsplit(".",1)[1] if '.' in file_name else None
        if file_type == "p":
            type = "pickle"
        elif file_type == "bin":
            type = "word2vec"
        elif file_type == "vec":
            type = "fasttext"
        else:
            type = "word2vec"

    if type == "word2vec":
        model = KeyedVectors.load_word2vec_format(file_name, binary=True, unicode_errors="ignore")
        words = model.index2entity
    elif type == "fasttext":
        model = FastText.load_fasttext_format(file_name)
        words = [w for w in model.wv.vocab]
    elif type == "pickle":
        with open(file_name,'rb') as fp:
            model = pickle.load(fp)
        words = model.keys()

    if lower:
        vectors = {word.lower(): model[word] for word in words}
    else:
        vectors = {word: model[word] for word in words}

    if "UNK" not in vectors:
        unk = np.mean([vectors[word] for word in vectors.keys()], axis=0)
        vectors["UNK"] = unk

    return vectors, len(vectors["UNK"])

def get_morph_dict(seqment_file, lowerCase):
    if seqment_file == "N/A":
        return {}

    morph_dict = {}
    with open(seqment_file) as text:
        for line in text:
            line = line.strip()
            index = line.split(":")[0].lower() if lowerCase else line.split(":")[0]
            data = line.split(":")[1].split("+")[0]
            if '-' in data:
                morph_dict[index] = data.split("-")
            else:
                morph_dict[index] = [data]
    return morph_dict

def get_morph_dict_array(seqment_file, lowerCase):
    if seqment_file == "N/A":
        return {}

    morph_dict = {}
    with open(seqment_file) as text:
        for line in text:
            line = line.strip()
            index = line.split(":")[0].lower() if lowerCase else line.split(":")[0]
            datas = line.split(":")[1].split("+")
            word_seq = []
            for data in datas:
                if data != "###":
                    if '-' in data:
                        word_seq.append(data.split("-"))
                    else:
                        word_seq.append([data])
            morph_dict[index] = word_seq
    return morph_dict

def get_morph_gold(gold_morph_dict, unsupervised_morph_dict):
    gold_data = {}

    for index in unsupervised_morph_dict.keys():
        if index in gold_morph_dict:
            gold_seq = gold_morph_dict[index]
            idx = 0
            for un_seq in unsupervised_morph_dict[index]:
                if len(un_seq) == len(gold_seq):
                    FLAG = True
                    for un, gold in zip(un_seq, gold_seq):
                        if un != gold:
                            FLAG = False
                    if FLAG:
                        gold_data[index] = idx
                idx += 1
            if index not in gold_data:
                gold_data[index] = 0
    return gold_data

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100