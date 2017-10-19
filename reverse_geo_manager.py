"""Merge all of reverse_geo data into single file
"""
from __future__ import unicode_literals
import pandas as pd 

folder = []
for i in range(1,7):
    filename = 'reverse_geo{}.csv'.format(i)
    file = pd.read_csv(filename,encoding='utf-8')
    folder.append(file)
df  = pd.concat(folder)
print(df.head())
print('Length of df:{}'.format(len(df)))

print('Saving to reverse_geo.csv...')
df.to_csv('reverse_geo.csv',index=False,columns=['City_x','Neighborhood_x','State_x','Street_x','geocodeQuality_x', 'itemID', 'itemID_2', 'postalCode_x','lat_x','lon_x'],encoding='utf-8')
