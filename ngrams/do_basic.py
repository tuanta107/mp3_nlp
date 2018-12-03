import numpy as np
import ngrams.train as train

unigram_dump_files = ['../output/unigramEN.txt','../output/unigramFR.txt','../output/unigramIT.txt']
bigram_dump_files = ['../output/bigramEN.txt','../output/bigramFR.txt','../output/bigramIT.txt']
excluding_chars = ' .,"\n\'[]()-;0123456789?*_!&$:<>\t«»'

unigram_models = train.train_ngram_2(n_grams=1,delta=0.5,excluding_chars=excluding_chars)
bigram_models = train.train_ngram_2(n_grams=2,delta=0.5,excluding_chars=excluding_chars)

for model,out_file in zip(unigram_models,unigram_dump_files):
    train.dump(model=model,out_file=out_file)

for model,out_file in zip(bigram_models,bigram_dump_files):
    train.dump(model=model,out_file=out_file)

#test input file
input_file='../datasets/first10sentences.txt'
result_file='../datasets/first10sentences_result.txt'
lang=train.load_input(result_file)
u_correct_count=0
b_correct_count=0
for sentence_ind,sentence in enumerate(train.load_input(input_file)):
    languages=['EN','FR','IT']
    output_file='../output/out%d.txt' % (sentence_ind)
    sentence = sentence.lower()
    output_string = sentence
    for c in excluding_chars:
        sentence = sentence.replace(c,'#')
    print(sentence)
    n_grams=2
    log_probs=[0,0,0]
    sentence_log_probs=[0,0,0]
    output_string += '\nUNIGRAM MODEL:'
    for token_ind in range (len(sentence)):
        c = sentence[token_ind:token_ind + 1]
        if c != '#':
            output_string += '\n\nUNIGRAM %s:' % (c)
            for model_ind,ngram in enumerate(unigram_models):
                log_probs[model_ind]=ngram.get(c,ngram.get('<unk>'))
                sentence_log_probs[model_ind] += log_probs[model_ind]
                output_string += '\n%s: P(%s) = %s ==> log prob of sentence so far: %s' \
                                 % (languages[model_ind],c,log_probs[model_ind],sentence_log_probs[model_ind])
    #conclusion
    most_probable_lang = languages[np.argmin(sentence_log_probs)]
    output_string += '\nAccording to the UNIGRAM model, the sentence is in %s' % most_probable_lang
    print('UNIGRAM:',most_probable_lang)
    if most_probable_lang == lang[sentence_ind]:
        u_correct_count += 1
    log_probs=[0,0,0]
    sentence_log_probs=[0,0,0]
    output_string += '\nBIGRAM MODEL:'
    for token_ind in range (len(sentence)-1):
        c = sentence[token_ind:token_ind + 2]
        output_string += '\n\nBIGRAM %s:' % (c)
        for model_ind,ngram in enumerate(bigram_models):
            log_probs[model_ind]=ngram.get(c,ngram.get('<unk>'))
            sentence_log_probs[model_ind] += log_probs[model_ind]
            output_string += '\n%s: P(%s|%s) = %s ==> log prob of sentence so far: %s' \
                             % (languages[model_ind],c[-1:],c[:-1],log_probs[model_ind],sentence_log_probs[model_ind])
    #conclusion
    most_probable_lang = languages[np.argmin(sentence_log_probs)]
    output_string += '\nAccording to the BIGRAM model, the sentence is in %s' % most_probable_lang
    print('BIGRAM:',most_probable_lang)
    if most_probable_lang == lang[sentence_ind]:
        b_correct_count += 1
    open(output_file, 'w').write(output_string)
print('Unigram accuracy:%d'%(u_correct_count))
print('Bigram accuracy:%d' %(b_correct_count))