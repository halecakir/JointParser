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

    def __str__(self):
        values = [str(self.id), self.form, self.lemma, self.pred_pos, self.xpos, self.feats,
                  str(self.pred_parent_id) if self.pred_parent_id is not None else None, self.pred_relation, self.deps,
                  self.misc]
        return '\t'.join(['_' if v is None else v for v in values])


def vocab(conll_path,morph_dict):
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
    root.idMorphs = [1, 2]
    tokens = [root]

    #create morpheme indexes out of morpheme dictionary
    all_morphs = []
    for word in morph_dict.keys():
        all_morphs += morph_dict[word]
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
                morphs_of_word.append(m2i["<w>"])
                if entry.norm in morph_dict:
                    for morph in morph_dict[entry.norm]:
                        if morph not in m2i:
                            morphs_of_word.append(m2i["UNK"])
                        else:
                            morphs_of_word.append(m2i[morph])
                elif entry.norm in m2i:
                    morphs_of_word.append(m2i[entry.norm])
                else:
                    morphs_of_word.append(m2i["UNK"])
                morphs_of_word.append(m2i["</w>"])
                entry.idMorphs = morphs_of_word

                tokens.append(entry)

    if len(tokens) > 1:
        wordsCount.update([node.norm for node in tokens if isinstance(node, ConllEntry)])
        posCount.update([node.pos for node in tokens if isinstance(node, ConllEntry)])
        relCount.update([node.relation for node in tokens if isinstance(node, ConllEntry)])

    return (wordsCount, {w: i for i, w in enumerate(wordsCount.keys())}, c2i, m2i, posCount.keys(), relCount.keys())


def read_conll(fh, c2i, morphemes):
    # Character vocabulary
    root = ConllEntry(0, '*root*', '*root*', 'ROOT-POS', 'ROOT-CPOS', '_', -1, 'rroot', '_', '_')
    root.idChars = [1, 2]
    root.idMorphs = [1, 2]
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
                morphs_of_word.append(morphemes[1]["<w>"])
                if entry.norm in morphemes[0]:
                    for morph in morphemes[0][entry.norm]:
                        if morph not in morphemes[1]:
                            morphs_of_word.append(morphemes[1]["UNK"])
                        else:
                            morphs_of_word.append(morphemes[1][morph])
                elif entry.norm in morphemes[1]:
                    morphs_of_word.append(morphemes[1][entry.norm])
                else:
                    morphs_of_word.append(morphemes[1]["UNK"])
                morphs_of_word.append(morphemes[1]["</w>"])
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
