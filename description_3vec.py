# coding: utf-8

from __future__ import unicode_literals

import multiprocessing as mp
import dask.dataframe as dd
import string
from stop_words import get_stop_words
import regex as re
from contextlib import closing


#pd.options.display.max_colwidth = -1
#pd.options.display.max_columns = None

filepath = "~/Documents/Avito/"
item_info = dd.read_csv(filepath+"ItemInfo_train.csv")

print('Length of df: {}'.format(len(item_info)))

# Russian Stopwords
ru_stopwords = get_stop_words('russian')

def remove_punctuation(text):
    return re.sub(ur"\p{P}+", "", text)

def clean_text(doc):
    try:
        docs = remove_punctuation(doc)
        docs = docs.lower().split(' ')
        return [tok for tok in docs if tok not in ru_stopwords]
    except Exception as e :
        print(doc)
        print(e)
        pass

item_info['description_clean'] = item_info['description'].apply(clean_text)


item_info.to_csv('ItemInfo_train2.csv', encoding='utf-8')

