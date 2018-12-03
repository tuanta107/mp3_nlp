import numpy as np
import ngrams.train as train
trigram_dump_files = ['../output/trigramEN.txt','../output/trigramFR.txt','../output/trigramIT.txt']
excluding_chars = ' .,"\n[]();0123456789?*_!&$:<>\t«»'

n_grams=2
models = train.train_ngram_2(n_grams=n_grams,delta=0.5,excluding_chars=excluding_chars,)

for model,out_file in zip(models,trigram_dump_files):
    train.dump(model=model,out_file=out_file)

# for model,out_file in zip(bigram_models,bigram_dump_files):
#     train.dump(model=model,out_file=out_file)

#test input file
# input_file='../datasets/first10sentences.txt'
# result_file='../datasets/first10sentences_result.txt'
input_file='../datasets/friends_simplified.txt'
result_file='../datasets/friends_lang.txt'
lang=train.load_input(result_file)
u_correct_count=0
for sentence_ind,sentence in enumerate(train.load_input(input_file)):
    languages=['EN','FR','IT']
    output_file='../output/out%d.txt' % (sentence_ind)
    sentence = '#' + sentence.lower()
    for c in excluding_chars:
        sentence = sentence.replace(c,'#')
    print(sentence)
    output_string = sentence
    log_probs=[0,0,0]
    sentence_log_probs=[0,0,0]
    tokens = train.tokenize(sentence,n_grams)
    output_string += '\nNGRAM MODEL:'
    for token in tokens:
        output_string += '\n\nNGRAM %s:' % (token)
        for model_ind,ngram in enumerate(models):
            log_probs[model_ind]=ngram.get(token,ngram.get('<unk>'))
            sentence_log_probs[model_ind] += log_probs[model_ind]
            output_string += '\n%s: P(%s) = %s ==> log prob of sentence so far: %s' \
                             % (languages[model_ind],token,log_probs[model_ind],sentence_log_probs[model_ind])
    #conclusion
    most_probable_lang = languages[np.argmin(sentence_log_probs)]
    output_string += '\nAccording to the NGRAM model, the sentence is in %s' % most_probable_lang
    if most_probable_lang == lang[sentence_ind]:
        u_correct_count += 1
    else:
        print('Failed, sentence in %s but model predicted %s' % (lang[sentence_ind],most_probable_lang))
    print(output_string)
    # open(output_file, 'w').write(output_string)
print('Model accuracy:%d' %(u_correct_count))