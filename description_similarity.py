
# coding: utf-8

# In[1]:

from __future__ import unicode_literals
import pandas as pd


pd.options.display.max_colwidth = -1
pd.options.display.max_columns = None


filepath = '~/Documents/Avito/'
df = pd.read_csv('ItemInfo_train5.csv',encoding='utf-8'
                 ,usecols=['description_x_clean','description_y_clean']
                 ,converters={'description_x_clean': lambda x: x.strip(u'[]').split(', ')
                 ,'description_y_clean': lambda x: x.strip('[]').split(', ')}
                ,nrows=10)
                             

# ### Word2Vec 
# - compute similarity between 2 documents (`description`)

from gensim import models, similarities


# model_new.wv.similarity('продам','размер')


#model_doc = models.doc2vec.Doc2Vec.load('description_doc')

# Vocabulary


# Filtering out words not in vocab
def remove_unknown_words(row):
    sent = []
    for word in row:
        if word in set(model_word.wv.index2word):
            sent.append(word)
    return sent

        
# Loading Word2Vec model
model_word = models.word2vec.Word2Vec.load('description_2vec.model')


def doc_sim(df):
    try:
        print('Normal')
        return model_word.n_similarity(df['description_x_clean']
                                  ,df['description_y_clean'])
    except:
        print('UNK present')
        #return model_word.n_similarity(remove_unknown_words(df['description_x_clean']),
        #       remove_unknown_words(df['description_y_clean']))


# In[17]:

df['description_similarity'] = df[['description_x_clean','description_y_clean']].apply(doc_sim,axis=1)                                                                           


# In[18]:

print(df.description_similarity.isnull().sum())
print(df.head())

# In[19]:

len(df)

