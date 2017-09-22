
# coding: utf-8

# In[1]:

import pandas as pd
#get_ipython().magic(u'matplotlib inline')
from matplotlib import pyplot as plt


# In[2]:

filepath = '~/Documents/Avito/'


# ## Initial testing on 400 rows

# In[3]:

item_info = pd.read_csv(filepath+'ItemInfo_train.csv')
item_info.tail(3)


# ### Mapping for parentCategoryID, regionID

# In[4]:

cat = pd.read_csv(filepath+'Category.csv')
location = pd.read_csv(filepath+'Location.csv')

item_pairs =pd.read_csv(filepath+'ItemPairs_train.csv')
item_pairs.rename(columns={'itemID_1':'itemID'}, inplace=True)


# In[5]:

df = pd.merge(pd.merge(pd.merge(item_info,cat, on='categoryID'),
              location, on='locationID'),
              item_pairs, on='itemID')

df.tail()


# ###  Mapping `lat`, `lon` using MapQuest

# In[6]:

import requests as re
from bs4 import BeautifulSoup


# In[27]:

# Obtain api key from MapQuest

# Remove newline character when reading from file
key = open('/home/juan/Documents/Avito/MapQuest_Key.txt','r').read().strip()
osm = ('http://www.mapquestapi.com/geocoding/v1/reverse?key='+key+'&location=')


# ### Feature: Street, Neighborhood, City, State, PostalCode, GeocodeQuality

# In[28]:

def latlon(lat,lon, code):
    srch = osm + str(lat)+','+ str(lon)+'&outFormat=xml'
#     print(srch)
    r = re.get(srch)
    soup = BeautifulSoup(r.content,'lxml') 
    return get_add(soup, code)


# In[29]:

def get_add(soup,code):
    return {'Street'+code: get_str(soup.street),
        'Neighborhood'+code: get_str(soup.find(type="Neighborhood")),
        'City'+code: get_str(soup.find(type="City")),
        'State'+code:get_str(soup.find(type="State")),
           'postalCode'+code: get_str(soup.postalcode),
           'geocodeQuality'+code: get_str(soup.geocodequality),}


# In[30]:

def get_str(tag):
    if tag is None:
        return 'Nil'
    else:
        return tag.string


# In[31]:

# lat lon should be in str
def df_latlon(df,lat,lon,code):
    df_ll = df.apply(lambda x: pd.Series(latlon(x[lat], x[lon], code)), axis=1)
    return pd.concat([df,df_ll],axis=1)


# In[32]:

df1 = df_latlon(df, 'lat', 'lon', '_x')
df1.head()


# ### Merging ItemID_2 info with ItemID_1

# In[33]:

cat2 = cat.rename(columns={'categoryID':'categoryID_y'})
location2 = location.rename(columns={'locationID':'locationID_y'})


# In[34]:

item_infofull = pd.read_csv(filepath+'ItemInfo_train.csv')


# In[35]:

len(item_infofull)


# In[36]:

item_infofull.rename(columns={'itemID':'itemID_2'}, inplace=True)

df2 = pd.merge(pd.merge(pd.merge(df1,item_infofull, on='itemID_2'),
               cat2, on='categoryID_y'),
               location2, on='locationID_y')


# In[37]:

df2 = df2.drop(['attrsJSON_x','metroID_x','attrsJSON_y','metroID_y'],axis=1)


# In[38]:

df3 = df_latlon(df2, 'lat_y', 'lon_y', '_y')
df3.tail()


# In[60]:

pd.options.display.max_colwidth = -1
pd.options.display.max_columns = None
df3.head()


# In[40]:

df3.isDuplicate.sum()


# ### Feature: Number of images identifiers that overlaps

# In[41]:

def n_similar_images(df):
    try:
        arr1 = df.images_array_x.split(',')
        try:
            arr2 = df.images_array_y.split(',')
            return sum(a in b for a, b in zip(arr1,arr2))
        except: pass
    except: pass


# In[42]:

num_similarimages = df3[['images_array_x','images_array_y']].apply(n_similar_images, axis=1)


# In[43]:

num_similarimages.sum()


# ### Feature: Difference in length of image array

# In[44]:

def diff_len_image_array(df2):
    try:
        arr1 = df2.images_array_x.split(',')
        try:
            arr2 = df2.images_array_y.split(',')
            return abs(len(arr1)-len(arr2))
        except: pass
    except: pass


# In[45]:

df3['lendiff_imagearray'] = df3[['images_array_x','images_array_y']].apply(diff_len_image_array, axis=1)


# ### Feature: sameCat

# In[46]:

df3['sameCat'] = df3.apply(
    lambda x: 1 if x.categoryID_x == x.categoryID_y else 0, axis=1)


# In[47]:

df3['sameCat'].sum()


# ### Feature: sameParentCat

# In[48]:

df3['sameParentCat'] = df3.apply(
    lambda x: 1 if x.parentCategoryID_x == x.parentCategoryID_y else 0 , axis=1)


# In[49]:

df3['sameParentCat'].sum()


# - Feature engineering done on the categories showed that `categoryID` and `parentCategoryID` is not very useful in predicting if an ad is a duplicate of the other.

# ### Feature: priceDifference

# In[50]:

df3['priceDifference'] = df3.apply(
    lambda x: abs(x.price_x - x.price_y), axis=1)


# ### Feature: samelatlon
# - Checks if item pairs have the same lat and lon values

# In[51]:

df3.apply(
    lambda x: 1 if x.lon_x == x.lon_y else 0, axis=1).sum()


# ### Feature: latlonDifference
# - Compute the differences between the lat and lon values for item pairs
# using **Haversine** distance
# 

# In[52]:

import math


# In[53]:

def haversine(lat1, lon1, lat2, lon2):
    r = 6371
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    
    dlon = (lon2-lon1)/2
    dlat = (lat2-lat1)/2
    ax = math.sin(dlat)**2 + math.cos(lat1) *math.cos(lat2) * math.sin(dlon)**2
    d = 2 * r * math.asin(math.sqrt(ax))
    return d


# In[54]:

df3['latlonDifference'] = df3.apply(
    lambda x: haversine(x['lat_x'], x['lon_x'], x['lat_y'], x['lon_y'])
    ,axis=1)


# In[56]:


# In[58]:

df3.head()


# In[59]:

df3.to_csv(filepath+'ItemInfo_train3.csv', encoding='utf-8', index=False)

