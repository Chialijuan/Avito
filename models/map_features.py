"""
This module maps features of duplicate ads (ItemID and ItemID_2), which includes the differences in price, latlon and length of image array.
"""

from __future__ import unicode_literals
import pandas as pd
import math

# Substitute for train/test 
item_info = pd.read_csv('../ItemInfo_test.csv', encoding='utf-8')

item_pairs = pd.read_csv('../ItemPairs_test.csv', encoding='utf-8')
item_pairs.rename(columns={'itemID_1':'itemID'}, inplace=True)

df = pd.merge(item_info, item_pairs, on='itemID', how='right')
item_info.rename(columns={'itemID': 'itemID_2'}, inplace=True)
df = pd.merge(df, item_info, on='itemID_2', how='left')

cat = pd.read_csv('../Category.csv')
cat_y = cat.rename(columns={'categoryID': 'categoryID_y'})
cat_x = cat.rename(columns={'categoryID': 'categoryID_x'})

loc = pd.read_csv('../Location.csv')
loc_y = loc.rename(columns={'locationID': 'locationID_y'})
loc_x = loc.rename(columns={'locationID': 'locationID_x'})

df = pd.merge(df, cat_x, on='categoryID_x', how='left')
df = pd.merge(df, cat_y, on='categoryID_y', how='left')
df = pd.merge(df, loc_x, on='locationID_x', how='left')
df = pd.merge(df, loc_y, on='locationID_y', how='left')

del cat, cat_x, cat_y, loc, loc_x, loc_y

print('Len of item_pairs: {}'.format(len(item_pairs)))
print('Len of df after merge:{}'.format(len(df)))

#### Difference in length of image array
def diff_len_imagearray(df2):
    try:
        arr1 = df2.images_array_x.split(',')
        try:
            arr2 = df2.images_array_y.split(',')
            return abs(len(arr1)-len(arr2))
        except: pass
    except: pass

df['lendiff_imagearray'] = df[['images_array_x','images_array_y']].apply(diff_len_imagearray, axis=1)
#### priceDifference
df['priceDifference'] = df.apply(lambda x: abs(x.price_x - x.price_y), axis=1)

#### latlonDifference
def haversine(lat1, lon1, lat2, lon2):
    r = 6371
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
                
    dlon = (lon2-lon1)/2
    dlat = (lat2-lat1)/2
    ax = math.sin(dlat)**2 + math.cos(lat1) *math.cos(lat2) * math.sin(dlon)**2
    d = 2 * r * math.asin(math.sqrt(ax))
    return d

df['latlonDifference'] = df.apply(lambda x: haversine(x['lat_x'], x['lon_x'], x['lat_y'], x['lon_y']) ,axis=1)
print(df.head())
#print('Saving to ItemInfo_test_mapped.csv...')
#df.to_csv('ItemInfo_test_mapped.csv', encoding='utf-8', index=False)

