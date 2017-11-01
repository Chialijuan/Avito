"""Ensemble models 
"""

from __future__ import unicode_literals
import numpy as np
import pandas as pd 
from mlxtend.classifier import EnsembleVoteClassifier, StackingClassifier
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn import model_selection
import xgboost as xgb
import sys

### Load data
usecols_train = ['itemID', 'categoryID_x', 'title_x', 'price_x', 'locationID_x', 'itemID_2', 'lat_x', 'lon_y', 'isDuplicate', 'categoryID_y', 'title_y', 'price_y', 'locationID_y', 'lat_y', 'lon_y', 'parentCategoryID_y', 'parentCategoryID_x', 'regionID_x', 'regionID_y', 'lendiff_imagearray', 'priceDifference', 'latlonDifference', 'fuzz_ratio', 'lev_dist', 'jaro_dist', 'jarow_dist', 'description_x_clean', 'description_y_clean', 'intersect_BOW','sym_diff_BOW','clusters','desc_sim'] 
usecols_test = ['id','itemID', 'categoryID_x', 'title_x', 'price_x', 'locationID_x', 'itemID_2', 'lat_x', 'lon_y', 'categoryID_y', 'title_y', 'price_y', 'locationID_y', 'lat_y', 'lon_y', 'parentCategoryID_y', 'parentCategoryID_x', 'regionID_x', 'regionID_y', 'lendiff_imagearray', 'priceDifference', 'latlonDifference', 'fuzz_ratio', 'lev_dist', 'jaro_dist', 'jarow_dist', 'description_x_clean', 'description_y_clean', 'intersect_BOW','sym_diff_BOW','clusters','desc_sim']            

train = pd.read_csv('ItemInfo_trainfull.csv', encoding='utf-8', usecols=usecols_train)
#test = pd.read_csv('ItemInfo_testfull.csv', encoding='utf-8', usecols=usecols_test)

# Null values are numerical, impute with zero                                           
train = train.fillna(value=0,axis=1)                                               
#test = test.fillna(value=0, axis=1)                                                       

nonnum_columns_train = [key for key in dict(train.dtypes) if dict(train.dtypes)[key] in ['object']] 
#nonnum_columns_test = [key for key in dict(test.dtypes) if dict(test.dtypes)[key] in ['object']]     
                                                                                        
def labelEncoder(data, columns):                                                  
    le = LabelEncoder()                                                          
    for feature in columns:    
        #print(feature)                                                                    
        #print(type(feature))                                                          
        #print(train[feature])                                                             
        data[feature] = le.fit_transform(data[feature])                   
    return data                                                                           
                                                                                     
train = labelEncoder(train, nonnum_columns_train) 
#test = labelEncoder(test, nonnum_columns_test)   
      
y = train['isDuplicate']
x = train.drop('isDuplicate', axis=1)

               
##### Neural Network
#model = Sequential()
#
#from keras.layers import Dense, Activation
#
#model.add(BatchNormalization(input_shape=(30,),axis=1))
#model.add(Dense(units=300, input_dim=30))
#model.add(Activation('relu'))
#model.add(Dense(units=200))
#model.add(Activation('relu'))
#model.add(Dense(units=100))
#model.add(Activation('relu'))
#model.add(Dense(units=64))
#model.add(Activation('relu'))
#model.add(Dense(units=64))
#model.add(Activation('relu'))
#model.add(Dense(units=1))
#model.add(Activation('sigmoid'))
#
#model.compile(loss='binary_crossentropy',optimizer='sgd',metrics=['accuracy'])
               
# Fitting of NN model
#model.fit(np.array(x_train), np.array(y_train), epochs=10, batch_size=32)
#print (model.evaluate(np.array(x_test), np.array(y_test),batch_size=32))

##### xgBoost model
model_xg = xgb.XGBClassifier(learning_rate=0.05, max_depth=6, min_child_weight=4, n_estimators=1000)

model_xg1 = xgb.XGBClassifier(learning_rate=0.1,gamma=0,max_depth=6,min_child_weight=1,max_delta_step=0,subsample=1,colsample_bytree=1,silent=1,seed=0,reg_lambda=1,reg_alpha=0)

##### Random Forest
model_rf = RandomForestClassifier(n_estimators=200)

##### EnsembleVoteClassifier
eclf = EnsembleVoteClassifier(clfs=[model_xg, model_xg1, model_rf], voting='soft')

#labels = ['XG', 'XG1', 'RF']
#for clf, label in zip([model_xg, model_xg1, model_rf], labels):
#    scores = model_selection.cross_val_score(clf, x, y, cv=5, scoring='roc_auc')
#    print("Accuracy: {} (+/- {}) [{}]".format(scores.mean(), scores.std(), label))
#
scores = model_selection.cross_val_score(eclf, x, y, cv=5, scoring='roc_auc')
print("Accuracy: {} (+/- {}) [{}]".format(scores.mean(), scores.std(), ['Ensemble']))

#### Test
#id = test['id']
#test.drop('id', axis=1, inplace=True)

#pred = model.predict(np.array(test), batch_size=128)
#pred = [item for sublist in pred for item in sublist]

#result = pd.DataFrame({'id': id, 'probability': pred})
#print(result.head)
#result.to_csv('Submission_4.csv', encoding='utf-8', index=False)

