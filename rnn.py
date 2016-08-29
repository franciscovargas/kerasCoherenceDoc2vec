from keras.models import Sequential
from keras.engine.training import slice_X
from keras.layers import Activation, TimeDistributed, Dense, RepeatVector, recurrent, Dropout
from keras.layers.recurrent import SimpleRNN, LSTM
from keras.utils.np_utils import to_categorical
import numpy as np
from gensim.models.word2vec import Word2Vec
from itertools import chain


class MySentences(object):
    def __init__(self, docstr):
        print(docstr, "£££")
        self.docstr = docstr

    def __iter__(self):
        ator = self.docstr.split("\n")
        for line in ator:
            yield line.split()


def doc2mat(txt):
    print("".join(x for x in txt))
    sentences = MySentences("".join(x for x in txt))
    print("DONE")
    model = Word2Vec(iter=4, min_count=1)
    model.build_vocab(sentences)
    X = list()
    doc = list()
    for i in range(10):
        model.train(sentences)
    for tx in txt:
        sentences = MySentences(tx)
        for sent in sentences:
            for word in sent:
                doc.append(model[word])
        X.append(doc)
        doc = list()
    return np.array(X)


def makeYlabl(txt, subs):
    n = len(subs.split())
    target_txt = list(chain(*map(lambda x: x.split(),
                                 txt.split("\n"))))
    print(target_txt)
    target = target_txt.index(subs[0])
    target_seq = np.zeros(len(target_txt))
    target_seq[target: target + n] = 1
    print(target_seq)
    return target_seq


class FuzzySearch(Sequential):

    HIDDEN = 80

    def __init__(self,
                 word_dim=100,
                 instances=32,
                 targets=2,
                 hidden_l=2):

        self.word_dim = word_dim
        self.instances = instances
        self.targets = targets
        self.hidden_l = hidden_l
        super(FuzzySearch, self).__init__()

        self.add(LSTM(self.HIDDEN,
                      input_shape=(None, word_dim),
                      return_sequences=True))
        # self.add(Dropout(0.2))
        while(hidden_l > 1):
            self.add(LSTM(self.HIDDEN, return_sequences=True))
            # self.add(Dropout(0.2))
            hidden_l -= 1

        self.add(TimeDistributed(Dense(1)))
        self.add(Activation("sigmoid"))

        self.compile(loss="binary_crossentropy", optimizer="rmsprop", metrics=['accuracy'])


class WordTable(object):
    '''
    Given a set of characters:
    + Encode them to a one hot integer representation
    + Decode the one hot integer representation to their character output
    + Decode a vector of probabilities to their character output
    '''
    def __init__(self, chars, maxlen):
        self.chars = sorted(set(chars))
        self.char_indices = dict((c, i) for i, c in enumerate(self.chars))
        self.indices_char = dict((i, c) for i, c in enumerate(self.chars))
        self.maxlen = maxlen

    def encode(self, C=None, maxlen=None):
        C = C if C is not None else self.chars
        maxlen = len(C)
        if maxlen != len(self.chars):
            X = np.zeros((maxlen, len(self.chars)))
            for i, c in enumerate(C):
                X[i, self.char_indices[c]] = 1
        else:
            X = np.identity(maxlen, int)
        return X

    def decode(self, X, calc_argmax=True):
        if calc_argmax:
            X = X.argmax(axis=-1)
        return ''.join(self.indices_char[x] for x in X)


def exp1():
    txt = ["hello my name is francisco.\n I work in production operations at BLK. \n I am single and just graduated from uni"]
    subs = "I work in production operations at BLK"
    y = (makeYlabl(txt[0], subs)).reshape(1, 20, 1)
    X = doc2mat(txt).reshape(1, 20, 100)
    print(X.shape)
    print(X.shape, len(y))

    sv = FuzzySearch()
    wt = WordTable(["hello", "its", "me"], 3)
    print(wt.encode(["hello", "its"]))
    print(sv.layers)
    sv.fit(X, y, nb_epoch=100)
    pred = np.argmax(sv.predict(X), axis=-1)
    print(pred, y, sv.predict(X))


if __name__ == '__main__':
    exp1()
