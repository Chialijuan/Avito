"""
This module creates features for intersecting BOW and non-intersecting BOW in description.
"""
from __future__ import unicode_literals
import pandas as pd


df = pd.read_csv('./ItemInfo_test5.csv', encoding='utf-8',usecols=['itemID','itemID_2','description_x_clean','description_y_clean'], converters={'description_x_clean':lambda x: x.strip(u'[]').split(', '),'description_y_clean':lambda x: x.strip(u'[]').split(', ')})


# Feature engineering: intersecting bag of words
def intersection_BOW(df):
    x = df['description_x_clean']
    y = df['description_y_clean']

    return ' '.join(set(x).intersection(set(y))).split(' ')


# Feature engineering: Bag of words in either x or y but not both
def sym_diff_BOW(df):
    x = df['description_x_clean']
    y = df['description_y_clean']
    
    return ' '.join(set(x).symmetric_difference(set(y))).split(' ')


df['intersect_BOW'] = df[['description_x_clean','description_y_clean']].apply(intersection_BOW,axis=1)
df['sym_diff_BOW'] = df[['description_x_clean','description_y_clean']].apply(sym_diff_BOW,axis=1)


df.head()

print('Saving to intersectingBOW_test.csv...')
df.to_csv('intersectingBOW_test.csv', encoding='utf-8', index=False)

