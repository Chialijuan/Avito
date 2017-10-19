
# coding: utf-8
from __future__ import unicode_literals
import pandas as pd
#get_ipython().magic(u'matplotlib inline')
from matplotlib import pyplot as plt
import requests as re
from bs4 import BeautifulSoup
import math


print('Loading in data..')
filepath = '../'
item_info = pd.read_csv(filepath+'ItemInfo_train_map.csv',encoding = 'utf-8')


#### Mapping for parentCategoryID, regionID
def pair_features(items):
    cat = pd.read_csv(filepath+'Category.csv')
    location = pd.read_csv(filepath+'Location.csv')

    item_pairs =pd.read_csv(filepath+'ItemPairs_train.csv')
    item_pairs.rename(columns={'itemID_1':'itemID'}, inplace=True)

    
    df1 = pd.merge(pd.merge(pd.merge(items,cat, on='categoryID'),
              location, on='locationID'),
              item_pairs, on='itemID')

    cat2 = cat.rename(columns={'categoryID':'categoryID_y'})
    location2 = location.rename(columns={'locationID':'locationID_y'})

    items.rename(columns={'itemID':'itemID_2'}, inplace=True)

   
    df2 = pd.merge(pd.merge(pd.merge(df1,items, on='itemID_2'),
               cat2, on='categoryID_y'),
               location2, on='locationID_y')
    
    return  df2.drop(['attrsJSON_x','metroID_x','attrsJSON_y','metroID_y'],axis=1)



df3 = pair_features(item_info)
#print('Mapping lat lon..')
#df1 = df_latlon(df.loc[200:300,:], 'lat_x', 'lon_x', '_x')
#df1.to_csv('df1.csv', encoding='utf-8', index=False)
#df3 = df_latlon(df1, 'lat_y', 'lon_y', '_y')
#

pd.options.display.max_colwidth = -1
pd.options.display.max_columns = None

# ### Feature: Number of images identifiers that overlaps

def n_similar_images(df):
    try:
        arr1 = df.images_array_x.split(',')
        try:
            arr2 = df.images_array_y.split(',')
            return sum(a in b for a, b in zip(arr1,arr2))
        except: pass
    except: pass

#num_similarimages = df3[['images_array_x','images_array_y']].apply(n_similar_images, axis=1)

#num_similarimages.sum()

# ### Feature: Difference in length of image array


def diff_len_image_array(df2):
    try:
        arr1 = df2.images_array_x.split(',')
        try:
            arr2 = df2.images_array_y.split(',')
            return abs(len(arr1)-len(arr2))
        except: pass
    except: pass


df3['lendiff_imagearray'] = df3[['images_array_x','images_array_y']].apply(diff_len_image_array, axis=1)


# ### Feature: sameCat
df3['sameCat'] = df3.apply(lambda x: 1 if x.categoryID_x == x.categoryID_y else 0, axis=1)

# ### Feature: sameParentCat
df3['sameParentCat'] = df3.apply(lambda x: 1 if x.parentCategoryID_x == x.parentCategoryID_y else 0 , axis=1)

# - Feature engineering done on the categories showed that `categoryID` and `parentCategoryID` is not very useful in predicting if an ad is a duplicate of the other.

# ### Feature: priceDifference
df3['priceDifference'] = df3.apply(lambda x: abs(x.price_x - x.price_y), axis=1)


# ### Feature: latlonDifference
# - Compute the differences between the lat and lon values for item pairs
# using **Haversine** distance

def haversine(lat1, lon1, lat2, lon2):
    r = 6371
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    
    dlon = (lon2-lon1)/2
    dlat = (lat2-lat1)/2
    ax = math.sin(dlat)**2 + math.cos(lat1) *math.cos(lat2) * math.sin(dlon)**2
    d = 2 * r * math.asin(math.sqrt(ax))
    return d

df3['latlonDifference'] = df3.apply(lambda x: haversine(x['lat_x'], x['lon_x'], x['lat_y'], x['lon_y']),axis=1)


df3.head()
print('Saving to ItemInfo_train_full.csv')
df3.to_csv(filepath+'ItemInfo_train_full.csv', encoding='utf-8', index=False)

