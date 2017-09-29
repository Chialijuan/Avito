
# coding: utf-8

from __future__ import unicode_literals
import multiprocessing as mp
from contextlib import closing
import pandas as pd
from multiprocessing.dummy import Pool as ThreadPool
import timeit
from gensim import models, similarities

pd.options.display.max_colwidth = -1
pd.options.display.max_columns = None


filepath = '~/Documents/Avito/'
df = pd.read_csv('ItemInfo_train5.csv',encoding='utf-8'
                 ,usecols=['description_x_clean','description_y_clean']
                 ,converters={'description_x_clean': lambda x: x.strip(u'[]').split(', ')
                 ,'description_y_clean': lambda x: x.strip('[]').split(', ')})
                             

##### Word2Vec 
# - compute similarity between 2 documents (`description`)

#  Loading Word2Vec model
print('Loading model...')
MODEL_WORD = models.word2vec.Word2Vec.load('description_2vec.model')

# model_new.wv.similarity('продам','размер')
#model_doc = models.doc2vec.Doc2Vec.load('description_doc')
#  Converting vocab list to dictionary for faster query time 
print(type(MODEL_WORD.wv.index2word))
vocab = set(MODEL_WORD.wv.index2word)


#  Filtering out words not in vocab
def remove_unknown_words(row):
    if len(row) != 0:
        sent = []
        for word in row:
            if word in vocab:
                sent.append(word)
        return sent

#  Compute cosine similarity score between 2 description 
def doc_sim(df):
    try:
        print('Normal')
        return MODEL_WORD.n_similarity(df['description_x_clean']
                                  ,df['description_y_clean'])
    except KeyError:
        print('UNK present')
        w1 = remove_unknown_words(df['description_x_clean'])
        w2 = remove_unknown_words(df['description_y_clean'])
        if len(w1) != 0 and len(w2) != 0:
            return MODEL_WORD.n_similarity(w1,w2)
        else:
            return 0
               
    except:
        return None


## Start Timing
start_time = timeit.default_timer()

## Pandas apply function
df['desc_sim'] = df[['description_x_clean','description_y_clean']].apply(doc_sim,axis=1)                                                                           

# MultiThreading
#pool = ThreadPool(4)
#df['description_similarity'] = pool.map(doc_sim,df)


## MultiPooling
###### WIP ######
# print('MultiPooling!')
# with closing(mp.Pool(4)) as p:
#    df['description_similarity'] = p.map(doc_sim,df)
#    p.terminate()

print('Deleting model and df..')
del MODEL_WORD
df.drop(['description_x_clean','description_y_clean'], axis=1, inplace=True)

print('No. of null:{}'.format(df.desc_sim.isnull().sum()))
print(df.head())
elapsed = timeit.default_timer() - start_time
print('Time elapsed: {}'.format(elapsed))

df.to_csv('description_similarity.csv', encoding='utf-8', index=False)
print('Saved to description_similarity.csv')

