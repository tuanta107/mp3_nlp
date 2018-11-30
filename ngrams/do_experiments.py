import numpy as np
import ngrams.train as train

trigram_dump_files = ['../output/trigramEN.txt','../output/trigramFR.txt','../output/trigramIT.txt']

trigram_models = train.train_ngram(n_grams=3,delta=0.5)
# bigram_models = train_bigram(n_grams=2,delta=0.5)
# bigram_models = train.train_bigram()

for model,out_file in zip(trigram_models,trigram_dump_files):
    train.dump(model=model,out_file=out_file)

# for model,out_file in zip(bigram_models,bigram_dump_files):
#     train.dump(model=model,out_file=out_file)

#test input file
input_file='../datasets/first10sentences.txt'

for sentence_ind,sentence in enumerate(train.load_input(input_file)):
    languages=['ENGLISH','FRENCH','ITALIAN']
    output_file='../output/out%d.txt' % (sentence_ind)
    sentence=sentence.lower().replace('.','').replace('?','')
    print(sentence)
    output_string = sentence
    n_grams=3
    log_probs=[0,0,0]
    sentence_log_probs=[0,0,0]
    output_string += '\nTRIGRAM MODEL:'
    for token_ind in range (len(sentence)-n_grams+1):
        c = sentence[token_ind:token_ind + n_grams]
        output_string += '\n\nTRIGRAM %s:' % (c)
        for model_ind,ngram in enumerate(trigram_models):
            log_probs[model_ind]=ngram.get(c,ngram.get('<unk>'))
            sentence_log_probs[model_ind] += log_probs[model_ind]
            output_string += '\n%s: P(%s) = %s ==> log prob of sentence so far: %s' \
                             % (languages[model_ind],c,log_probs[model_ind],sentence_log_probs[model_ind])
    #conclusion
    most_probable_lang = languages[np.argmin(sentence_log_probs)]
    output_string += '\nAccording to the TRIGRAM model, the sentence is in %s' % most_probable_lang
    print('TRIGRAM:',most_probable_lang)
    print(output_string)
    # open(output_file, 'w').write(output_string)
