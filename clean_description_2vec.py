from __future__ import unicode_literals
from multiprocessing.dummy import Pool as ThreadPool
import pandas as pd
import string
from stop_words import get_stop_words
import regex as re
import timeit

#filepath = '~/Documents/Avito/Avito/'

# Read in as string not unicode 
item_info = pd.read_csv('ItemInfo_test_mapped.csv', usecols=['itemID','itemID_2','description_x','description_y'], encoding='utf-8')

print('Length of df: {}'.format(len(item_info)))

# Russian Stopwords
#ru_stopwords = get_stop_words('russian')

# Remove Punctuations and multiple spaces
def remove_punctuation(text):
    result = re.sub(ur"\p{P}+", "", text)
    result = re.sub(ur'(\s)\1{1,}', r'\1', result)
    return result

def clean_text(doc):
    try:
        #print(doc)
        docs = remove_punctuation(doc).lower().split()
        result =  [tok for tok in docs if tok not in get_stop_words('russian')]
        #print(type(result))
        return result
    except Exception as e :
        print(doc)
        print(e)
        pass

item_info = item_info.fillna('None')

#description_x = item_info['description_x']
#description_y = item_info['description_y']

#item_info.drop(['description_x','description_y'], inplace=True, axis=1)
print(item_info.columns)
start_time = timeit.default_timer()

# pandas .apply
#item_info['description_clean'] = item_info['description'].apply(clean_text)

# MultiThreading
pool = ThreadPool(4)
item_info['description_x_clean'] = pool.map(clean_text,item_info['description_x'])
print(type(item_info['description_x_clean'][0]))
item_info['description_y_clean'] = pool.map(clean_text,item_info['description_y'])

# MultiPooling
#with closing(mp.Pool(3)) as p:
#    item_info['description_clean'] = p.imap(clean_text,description,10)
#    p.terminate()

#print(item_info.description_x_clean[398])
elapsed = timeit.default_timer()-start_time
print('Time elapsed: {}'.format(elapsed))
# Time Elapsed for Thread: 1118.923s = 19 mins
# Time Elapsed for .apply: Killed

item_info.to_csv('ItemInfo_test5.csv', columns = ['itemID','itemID_2','description_x_clean','description_y_clean'], encoding='utf-8', index=False)
print('Saved to ItemInfo_test5.csv')
