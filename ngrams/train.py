from collections import Counter
import numpy as np


def tokenize(corpus,n_grams=1):
    if n_grams == 1:
        return corpus
    else:
        corpus_len = len(corpus)
        ret = [corpus[i:i+n_grams] for i in range(corpus_len-n_grams+1)]
        return ret


def load_corpus(files,delta,vocab_size,n_grams=1,ignore_punctuation=True,to_lower=True):
    char_to_remove = ' .,"\n[]()-;0123456789?*_!&$:<>\t«»'
    corpus = ''.join([open(file=file).read() for file in files])
    print(files)
    if ignore_punctuation:
        for c in char_to_remove:
            corpus=corpus.replace(c,'')
    if to_lower:
        corpus=corpus.lower()
    grams_1=Counter(tokenize(corpus,n_grams-1))
    grams=Counter(tokenize(corpus,n_grams))
    # print(''.join(grams.keys()))
    # print(grams.keys())
    #calculate smoothed probabilities with
    for g in grams.keys():
        g_1=g[:-1]
        smoothed_log_prob=-np.log10((grams.get(g) + delta)/(grams_1.get(g_1) + vocab_size * delta))
        grams[g]=smoothed_log_prob
    default_smoothed_log_prob = -np.log10(delta / (grams_1.get(g_1) + vocab_size * delta))
    grams['<unk>'] = default_smoothed_log_prob
    return grams

def train_unigram(n_grams,delta):
    en_unigram = load_corpus(['../datasets/en-moby-dick.txt','../datasets/en-the-little-prince.txt'],delta,26,n_grams=n_grams)
    fr_unigram = load_corpus(['../datasets/fr-vingt-mille-lieues-sous-les-mers.txt','../datasets/fr-le-petit-prince.txt'],delta,26,n_grams=n_grams)
    it_unigram = load_corpus(['../datasets/it-una-donna.txt'],delta,26,n_grams=n_grams)

    return [en_unigram, fr_unigram, it_unigram]

def train_bigram():
    en_bigram = load_corpus(['../datasets/en-moby-dick.txt', '../datasets/en-the-little-prince.txt'], 0.5, 26,
                             n_grams=2)
    fr_bigram = load_corpus(
        ['../datasets/fr-vingt-mille-lieues-sous-les-mers.txt', '../datasets/fr-le-petit-prince.txt'], 0.5, 26,
        n_grams=2)
    it_bigram = load_corpus(['../datasets/it-una-donna.txt'], 0.5, 25, n_grams=2)

    return [en_bigram, fr_bigram, it_bigram]

def train_ngram(n_grams,delta):
    en_bigram = load_corpus(['../datasets/en-moby-dick.txt', '../datasets/en-the-little-prince.txt'], delta, 26,
                             n_grams=n_grams)
    fr_bigram = load_corpus(
        ['../datasets/fr-vingt-mille-lieues-sous-les-mers.txt', '../datasets/fr-le-petit-prince.txt'], delta, 26,
        n_grams=n_grams)
    it_bigram = load_corpus(['../datasets/it-una-donna.txt'], delta, 25, n_grams=n_grams)

    return [en_bigram, fr_bigram, it_bigram]

def load_input(file):
    return open(file=file).read().split('\n')

def dump(model,out_file):
    out_string = ''
    for g,prob in model.items():
        if g =='<unk>':
            continue
        n_grams = len(g)
        if n_grams == 1:
            line = '(%s) = %s' % (g,prob)
        else:
            line = '(%s|%s) = %s' % (g[-1:],g[:-1],prob)
        out_string = out_string + line + '\n'
    open(out_file, 'w').write(out_string)

