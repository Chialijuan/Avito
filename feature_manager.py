"""
This module combine all new features into a single file
"""

from __future__ import unicode_literals
import pandas as pd 

def load_data(data):
     #### Reading in all data
    item_info = pd.read_csv('ItemInfo_{}_mapped.csv'.format(data), encoding='utf-8')
    title_sim = pd.read_csv('title_sim_{}.csv'.format(data), encoding='utf-8')
    desc_sim = pd.read_csv('description_similarity_{}.csv'.format(data), encoding='utf-8', usecols=['desc_sim'])
    intersecting_BOW = pd.read_csv('intersectingBOW_{}.csv'.format(data), encoding='utf-8')
    clustering_pairs = pd.read_csv('clustering_pairs_{}.csv'.format(data), encoding='utf-8')
    #reverse_geo = pd.read_csv('reverse_geo_{}.csv'.format(data), encoding='utf-8')
    #reverse_geo.drop(['itemID', 'itemID_2'], axis=1, inplace=True)
    #### Merging on itemID and itemID_2
    
    clustering_pairs.rename(columns={'itemID_1': 'itemID'}, inplace=True)
    #clustering_pairs.drop('generationMethod',axis=1, inplace=True)
    lst = [item_info, title_sim, intersecting_BOW, clustering_pairs]
    
    df = reduce(lambda left, right:  pd.merge(left, right, on=['itemID', 'itemID_2'], how='left'),lst)
    
    df = pd.concat([df,desc_sim], axis=1)
    #df = pd.merge(df, reverse_geo, on=['lat_x','lon_x'], how='left')
    print(df.head())
    print('Saving to ItemInfo_{}full.csv...'.format(data))
    df.to_csv('ItemInfo_{}full.csv'.format(data), encoding='utf-8', index=False)
    
load_data('test')
