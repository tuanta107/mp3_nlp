from collections import Counter
import numpy as np
import re


def tokenize(corpus,n_grams=1):
    if n_grams == 1:
        return corpus
    else:
        corpus_len = len(corpus)
        ret = []
        for i in range (corpus_len-n_grams+1):
            token_candidate = corpus[i:i+n_grams]
            if not '#' in token_candidate[1:-1]:
                ret.append(token_candidate)
        return ret


def load_corpus(files,delta,vocab_size,excluding_chars,n_grams=1,ignore_punctuation=True,to_lower=True):
    corpus = ''.join([open(file=file).read() for file in files])
    if to_lower:
        corpus=corpus.lower()
    if ignore_punctuation:
        for c in excluding_chars:
            corpus=corpus.replace(c,'')
    else:
        for c in excluding_chars:
            corpus=corpus.replace(c,'#')
        # print(corpus)
        corpus = re.sub('##+', '#',corpus)
        # print(corpus)
    grams_1=Counter(tokenize(corpus,n_grams-1))
    print('gram-1',grams_1)
    grams=Counter(tokenize(corpus,n_grams))
    print(len(corpus))
    print(grams)
    print(len(grams.keys()))
    #calculate smoothed probabilities with
    for g in grams.keys():
        g_1=g[:-1]
        smoothed_log_prob=-np.log10((grams.get(g) + delta)/(grams_1.get(g_1) + vocab_size * delta))
        grams[g]=smoothed_log_prob
    default_smoothed_log_prob = -np.log10(delta / (grams_1.get(g_1) + vocab_size * delta))
    grams['<unk>'] = default_smoothed_log_prob
    return grams
def train_ngram_2(n_grams,delta,excluding_chars):
    en_unigram = load_corpus(['../datasets/en-moby-dick.txt','../datasets/en-the-little-prince.txt'],
                             delta,26,n_grams=n_grams,
                             excluding_chars=excluding_chars,
                             ignore_punctuation=False)
    fr_unigram = load_corpus(['../datasets/fr-vingt-mille-lieues-sous-les-mers.txt','../datasets/fr-le-petit-prince.txt'],
    # fr_unigram = load_corpus(['../datasets/fr-vingt-mille-lieues-sous-les-mers.txt','../datasets/fr-le-petit-prince.txt','../datasets/belle-rose.txt'],
                             delta,26,n_grams=n_grams,
                             excluding_chars=excluding_chars,
                             ignore_punctuation = False)
    it_unigram = load_corpus(['../datasets/it-una-donna.txt','../datasets/vacanza_in_tenda.txt'],
    # it_unigram = load_corpus(['../datasets/it-una-donna.txt','../datasets/vacanza_in_tenda.txt','../datasets/l-argentina-vista-come-e'],
                             delta,25,n_grams=n_grams,excluding_chars=excluding_chars,
                             ignore_punctuation=False)
    return [en_unigram, fr_unigram, it_unigram]


# def train_ngram(n_grams,delta,excluding_chars):
#     en_unigram = load_corpus(['../datasets/en-moby-dick.txt','../datasets/en-the-little-prince.txt'],delta,26,n_grams=n_grams,excluding_chars=excluding_chars)
#     fr_unigram = load_corpus(['../datasets/fr-vingt-mille-lieues-sous-les-mers.txt','../datasets/fr-le-petit-prince.txt'],delta,26,n_grams=n_grams,excluding_chars=excluding_chars)
#     it_unigram = load_corpus(['../datasets/it-una-donna.txt','../datasets/vacanza_in_tenda.txt','../datasets/l-argentina-vista-come-e'],delta,25,n_grams=n_grams,excluding_chars=excluding_chars)
#     return [en_unigram, fr_unigram, it_unigram]

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

