"""Neural Network Model
"""
from __future__ import unicode_literals
import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split 
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from sklearn.preprocessing import LabelEncoder
import sys
model = Sequential()

from keras.layers import Dense, Activation

model.add(BatchNormalization(input_shape=(30,),axis=1))
model.add(Dense(units=300, input_dim=30))
model.add(Activation('relu'))
model.add(Dense(units=200))
model.add(Activation('relu'))
model.add(Dense(units=100))
model.add(Activation('relu'))
model.add(Dense(units=64))
model.add(Activation('relu'))
model.add(Dense(units=64))
model.add(Activation('relu'))
model.add(Dense(units=1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',optimizer='sgd',metrics=['accuracy'])
               
usecols_train = ['itemID', 'categoryID_x', 'title_x', 'price_x', 'locationID_x', 'itemID_2', 'lat_x', 'lon_y', 'isDuplicate', 'categoryID_y', 'title_y', 'price_y', 'locationID_y', 'lat_y', 'lon_y', 'parentCategoryID_y', 'parentCategoryID_x', 'regionID_x', 'regionID_y', 'lendiff_imagearray', 'priceDifference', 'latlonDifference', 'fuzz_ratio', 'lev_dist', 'jaro_dist', 'jarow_dist', 'description_x_clean', 'description_y_clean', 'intersect_BOW','sym_diff_BOW','clusters','desc_sim'] 
usecols_test = ['id','itemID', 'categoryID_x', 'title_x', 'price_x', 'locationID_x', 'itemID_2', 'lat_x', 'lon_y', 'categoryID_y', 'title_y', 'price_y', 'locationID_y', 'lat_y', 'lon_y', 'parentCategoryID_y', 'parentCategoryID_x', 'regionID_x', 'regionID_y', 'lendiff_imagearray', 'priceDifference', 'latlonDifference', 'fuzz_ratio', 'lev_dist', 'jaro_dist', 'jarow_dist', 'description_x_clean', 'description_y_clean', 'intersect_BOW','sym_diff_BOW','clusters','desc_sim']            

train = pd.read_csv('ItemInfo_trainfull.csv', encoding='utf-8', usecols=usecols_train)
test = pd.read_csv('ItemInfo_testfull.csv', encoding='utf-8', usecols=usecols_test)
print(train.head())                                                                            
print(test.head())                                                                        

# Null values are numerical, impute with zero                                           
train = train.fillna(value=0,axis=1)                                               
test = test.fillna(value=0, axis=1)                                                       

nonnum_columns_train = [key for key in dict(train.dtypes) if dict(train.dtypes)[key] in ['object']] 
nonnum_columns_test = [key for key in dict(test.dtypes) if dict(test.dtypes)[key] in ['object']]     
                                                                                        
def labelEncoder(data, columns):                                                  
    le = LabelEncoder()                                                          
    for feature in columns:    
        #print(feature)                                                                    
        #print(type(feature))                                                          
        #print(train[feature])                                                             
        data[feature] = le.fit_transform(data[feature])                   
    return data                                                                           
                                                                                     
train = labelEncoder(train, nonnum_columns_train) 
test = labelEncoder(test, nonnum_columns_test)   
      
y = train['isDuplicate']
x= train.drop('isDuplicate', axis=1)

##Evaluate model
seed = 42
test_size = 0.3
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = test_size, random_state = seed)                      
print (np.array(x_train[:10]))
model.fit(np.array(x_train), np.array(y_train), epochs=10, batch_size=32)
print (model.evaluate(np.array(x_test), np.array(y_test),batch_size=32))

##Saving NN model
#from sklearn.externals import joblib
#joblib.dump(model, 'nn_model.pkl', compress=True)

#model = joblib.load('nn_model.pkl')
id = test['id']
test.drop('id', axis=1, inplace=True)

pred = model.predict(np.array(test), batch_size=128)
pred = [item for sublist in pred for item in sublist]

result = pd.DataFrame({'id': id, 'probability': pred})
print(result.head)
result.to_csv('Submission_2.csv', encoding='utf-8', index=False)

