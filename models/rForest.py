"""Random Forest Model
"""
from __future__ import unicode_literals
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split, GridSearchCV

#usecols_train = ['itemID', 'categoryID_x', 'title_x', 'price_x', 'locationID_x', 'itemID_2', 'lat_x', 'lon_y', 'isDuplicate', 'categoryID_y', 'title_y', 'price_y', 'locationID_y', 'lat_y', 'lon_y', 'parentCategoryID_y', 'parentCategoryID_x', 'regionID_x', 'regionID_y', 'lendiff_imagearray', 'priceDifference', 'latlonDifference', 'fuzz_ratio', 'lev_dist', 'jaro_dist', 'jarow_dist', 'description_x_clean', 'description_y_clean', 'intersect_BOW','sym_diff_BOW','clusters','desc_sim']  
                                             
usecols_test = ['id','itemID', 'categoryID_x', 'title_x', 'price_x', 'locationID_x', 'itemID_2', 'lat_x', 'lon_y', 'categoryID_y', 'title_y', 'price_y', 'locationID_y', 'lat_y', 'lon_y', 'parentCategoryID_y', 'parentCategoryID_x', 'regionID_x', 'regionID_y', 'lendiff_imagearray', 'priceDifference', 'latlonDifference', 'fuzz_ratio', 'lev_dist', 'jaro_dist', 'jarow_dist', 'description_x_clean', 'description_y_clean', 'intersect_BOW','sym_diff_BOW','clusters','desc_sim']        
                                                                                                                    
#train = pd.read_csv('ItemInfo_trainfull.csv', encoding='utf-8', usecols=usecols_train)
test = pd.read_csv('ItemInfo_testfull.csv', encoding='utf-8', usecols=usecols_test) 
#print(train.head())                                                                    
#print(test.head())                                                                              
# Null values are numerical, impute with zero                                                    
#train = train.fillna(value=0,axis=1)                                                      
test = test.fillna(value=0, axis=1)    
                                                        
#nonnum_columns_train = [key for key in dict(train.dtypes) if dict(train.dtypes)[key] in ['object']]   
nonnum_columns_test = [key for key in dict(test.dtypes) if dict(test.dtypes)[key] in ['object']]      
def labelEncoder(data,columns):                                                                     
    le = LabelEncoder()                                                      
                                                       
    for feature in columns:                                                              
        #print(feature)                                                                         
        #print(type(feature))                                                                  
        #print(train[feature])                                                                  
        data[feature] = le.fit_transform(data[feature])                                        
    return data                                                                            
                                                                                            
#train = labelEncoder(train, nonnum_columns_train)
test = labelEncoder(test, nonnum_columns_test)

#y = train['isDuplicate']
#x = train.drop('isDuplicate', axis=1)

#seed = 3
#test_size = 0.33
#x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=seed)

#model = RandomForestClassifier(n_estimators=200)

# GridSearchCV
#param_grid = {'n_estimators': [45, 100, 200]}
#model_cv = GridSearchCV(estimator=model, param_grid=param_grid, cv=7)
#model_cv.fit(x, y)
#print(model_cv.best_score_)
#print(model_cv.best_params_)

#model.fit(x,y)

#joblib.dump(model, 'rForest_model.pkl', compress=True)

model = joblib.load('rForest_model.pkl')
id = test['id']
test.drop('id', axis=1, inplace=True)

pred = model.predict_proba(test)
pred = [value[0] for value in pred]

result = pd.DataFrame({'id': id, 'probability': pred})
print(result.head)
result.to_csv('Submission_3.csv', encoding='utf-8', index=False)
"""                                                                         
