from __future__ import unicode_literals
from fuzzywuzzy import fuzz
import jellyfish
import pandas as pd 
import numpy as np

# Substitute for train/test
item_info = pd.read_csv('../ItemInfo_test.csv', encoding='utf-8',usecols =['itemID','title'], converters={'title':unicode})

item_pairs = pd.read_csv('../ItemPairs_test.csv', encoding='utf-8',usecols=['itemID_1','itemID_2'])
item_pairs.rename(columns={'itemID_1':'itemID'}, inplace=True)

df = pd.merge(item_info,item_pairs, on='itemID')


item_info.rename(columns={'itemID':'itemID_2'}, inplace=True)

df = pd.merge(df,item_info, on='itemID_2')
print(df.head())

# Fuzzy Wuzzy 
print('Computing token_set_ratio...')
df.loc[:,('fuzz_ratio')] = df.apply(lambda x: fuzz.token_set_ratio(x['title_x'],x['title_y'])/100.0, axis=1)


# Jellyfish
# Omit Hamming; undefined for strings with different length
print('Computing jellyfish ratio...')
df.loc[:,('lev_dist')] = df.apply(lambda x: jellyfish.levenshtein_distance(x['title_x'],x['title_y']), axis=1)                                             

df.loc[:,('jaro_dist')] = df.apply(lambda x: jellyfish.jaro_distance(x['title_x'], x['title_y']), axis=1)

df.loc[:,('jarow_dist')] = df.apply(lambda x: jellyfish.jaro_winkler(x['title_x'], x['title_y']), axis=1)

print('Deleting unwanted columns in df...')
df.drop(['title_x','title_y'], inplace=True,axis=1)

print(df.head())
print('Saving to title_sim.csv...')
df.to_csv('title_sim_test.csv', encoding='utf-8', index=False)
