""" Mapping `lat`, `lon` using MapQuest API key

"""
from __future__ import unicode_literals
import pandas as pd
import requests as re
from bs4 import BeautifulSoup
import time 
# Substitute for train/test 
# Load in data
item_info = pd.read_csv('ItemInfo_test_mapped.csv', encoding='utf-8', usecols =['itemID','itemID_2','lat_x', 'lon_x'])
print(item_info.head())

# Remove duplicated latlon
unique_ll = item_info.sort_values('itemID').drop_duplicates(subset=['lat_x', 'lon_x'], keep='last').reset_index()
print('No. of unique lat,lon: {}'.format(unique_ll.count()))

chunks = [unique_ll[i:i + 15000] for i in xrange(0, len(unique_ll), 15000)]
print(len(chunks))
#import pdb; pdb.set_trace()
#unique_ll2 = unique_ll.loc[552:14448,:]   #reverse_geo1
#unique_ll2 = unique_ll.loc[14448:29448,:] # reverse_geo2 (621 rows)
#unique_ll2 = unique_ll.loc[24140:30069,:]  # reverse_geo3 + reverse_geo5
#unique_ll2 = unique_ll.loc[30069:,:]  # reverse_geo4
#unique_ll4 = unique_ll.loc[29448:,:]
#lst = [unique_ll2,unique_ll3,unique_ll4]
#print('printing first element in list:')
#print(lst[0])
# Remove newline character when reading from file
#key = open('/home/juan/Documents/Avito/MapQuest_Key.txt','r').read().strip()
#osm = ('http://www.mapquestapi.com/geocoding/v1/reverse?key='+key+'&location=')

def get_osm(n):
    file = '/home/juan/Documents/Avito/MapQuest_Key{}.txt'.format(n)
    key = open(file,'r').read().strip()
    
    return ('http://www.mapquestapi.com/geocoding/v1/reverse?key='+key+'&location=')

### Feature: Street, Neighborhood, City, State, PostalCode, GeocodeQuality

# lat lon should be in str

def latlon(lat,lon, code, osm):
    srch = osm + str(lat)+','+ str(lon)+'&outFormat=xml'
    print(srch)
    #print('Scraping from MapQuest..')
    time.sleep(0.01)
    r = re.get(srch)
    soup = BeautifulSoup(r.content,'lxml')
    #return None
    return {'Street'+code: get_str(soup.street),
        'Neighborhood'+code: get_str(soup.find(type="Neighborhood")),
        'City'+code: get_str(soup.find(type="City")),
        'State'+code:get_str(soup.find(type="State")),
           'postalCode'+code: get_str(soup.postalcode),
           'geocodeQuality'+code: get_str(soup.geocodequality),}

def get_str(tag):
    if tag is None:
        return 'Nil'
    else:
        return tag.string

def df_latlon(df,lat,lon,code, osm):
        df_ll = df.apply(lambda x: pd.Series(latlon(x[lat], x[lon], code, osm)), axis=1)
        final =  pd.concat([df,df_ll],axis=1)
        print(final)
        return final

#df_lst = []
#for i,chunk in enumerate(chunks):
#    print(get_osm(i))
#    df_lst.append(df_latlon(chunk, 'lat_x','lon_x','_x', get_osm(i)))
#
#result = pd.concat(df_lst,axis=0)
#print(result.head())

result0 = df_latlon(chunks[0], 'lat_x', 'lon_x', '_x', get_osm(0))
result1 = df_latlon(chunks[1], 'lat_x', 'lon_x', '_x', get_osm(1))
result = pd.concat([result0, result1], axis=0)

#MultiPooling
#pool = mp.Pool(processes=4)
#df_ll = [pool.apply(df_latlon, args=(r1,r2)) for r1,r2 in df[['lat_x','lon_x']]]
#df_ll = pool.map(df_latlon, (df['lat_x'],df['lon_x']))

#df_ll = pool.map(df_latlon, df)

#print(df_ll)
#print(type(df_ll))
#result = pd.concat([df,df_ll],axis=1)

print('Saving to reverse_geo6_test.csv...')
result.to_csv('reverse_geo6_test.csv', encoding='utf-8', index=False)

