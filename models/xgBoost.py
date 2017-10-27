"""XGBoost model

"""

from __future__ import unicode_literals
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_selection import SelectFromModel


usecols_train = ['itemID', 'categoryID_x', 'title_x', 'price_x', 'locationID_x', 'itemID_2', 'lat_x', 'lon_y', 'isDuplicate', 'categoryID_y', 'title_y', 'price_y', 'locationID_y', 'lat_y', 'lon_y', 'parentCategoryID_y', 'parentCategoryID_x', 'regionID_x', 'regionID_y', 'lendiff_imagearray', 'priceDifference', 'latlonDifference', 'fuzz_ratio', 'lev_dist', 'jaro_dist', 'jarow_dist', 'description_x_clean', 'description_y_clean', 'intersect_BOW','sym_diff_BOW','clusters','desc_sim'] 
#
usecols_test = ['id','itemID', 'categoryID_x', 'title_x', 'price_x', 'locationID_x', 'itemID_2', 'lat_x', 'lon_y', 'categoryID_y', 'title_y', 'price_y', 'locationID_y', 'lat_y', 'lon_y', 'parentCategoryID_y', 'parentCategoryID_x', 'regionID_x', 'regionID_y', 'lendiff_imagearray', 'priceDifference', 'latlonDifference', 'fuzz_ratio', 'lev_dist', 'jaro_dist', 'jarow_dist', 'description_x_clean', 'description_y_clean', 'intersect_BOW','sym_diff_BOW','clusters','desc_sim'] 


#train = pd.read_csv('ItemInfo_trainfull.csv', encoding='utf-8', usecols=usecols_train)
test = pd.read_csv('ItemInfo_testfull.csv', encoding='utf-8', usecols=usecols_test)

#print(train.head())
print(test.head())
# Null values are numerical, impute with zero
#train = train.fillna(value=0,axis=1)
test = test.fillna(value=0, axis=1)

#nonnum_columns = [key for key in dict(train.dtypes) if dict(train.dtypes)[key] in ['object']]
nonnum_columns = [key for key in dict(test.dtypes) if dict(test.dtypes)[key] in ['object']]

def labelEncoder(data):
    le = LabelEncoder()

    for feature in nonnum_columns:
        #print(feature)
        #print(type(feature))
        #print(train[feature])
        data[feature] = le.fit_transform(data[feature])
    return data

#train = labelEncoder(train)
test = labelEncoder(test)

#print(train.head())
# Training of model
#model_1 = xgb.XGBClassifier(learning_rate=0.05, max_depth=6, min_child_weight=4, n_estimators=1000)
#model = xgb.XGBClassifier(learning_rate=0.1,gamma=0,max_depth=6,min_child_weight=1,max_delta_step=0,subsample=1,colsample_bytree=1,silent=1,seed=0,reg_lambda=1,reg_alpha=0)
#
#y = train['isDuplicate']
#x = train.drop('isDuplicate', axis=1)
#
#seed = 1
#test_size = 0.77
#x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = test_size, random_state = seed)
#
#gbm_train = model.fit(x_train,y_train)
#gbm_pred = model_1.fit(x, y)
from sklearn.externals import joblib

#print('Saving xgBoost model...')
#joblib.dump(gbm_pred, 'gbm_pred_model_1.pkl', compress=True)

id = test['id']
test.drop('id', axis=1, inplace=True)

print('Loading xgBoost model...')
gbm_pred = joblib.load('gbm_pred_model_1.pkl')

pred_label = gbm_pred.predict(test)

pred = gbm_pred.predict_proba(test)
pred = [value[1] for value in pred]

result = pd.DataFrame({'id': id, 'probability': pred})
print(result.head())
result.to_csv('Submission_1.csv', encoding='utf-8', index=False)
#pred = gbm_train.predict(x_test)
#pred_1 = [round(value) for value in pred]
#accuracy_1 = accuracy_score(y_test, pred_1)
#print('XGBoost Accuracy: {}'.format(accuracy_1 * 100.0))
#
#print('XGBoost ROC Accuracy: {}'.format(roc_auc_score(y_test, gbm_train.predict(x_test))))
#
#print('Feature importance: {}'.format(gbm_train.feature_importances_))
#
## ### Use Feature Importance for feature selction
#
#thresholds = np.sort(gbm_train.feature_importances_)
#
#
#lst =[]
#for i, thresh in enumerate(thresholds):
##     print('Threshold:{}, {}'.format(thresh, i))
#    # gbm is prefitted in the above 
#    selection = SelectFromModel(gbm_train, threshold=thresh, prefit=True)
#    # Output should be a matrix to input into model, no need to call fit
##     print(x_train.shape)
#    select_x_train = selection.transform(x_train)
#    
#    # Train model
#    selection_model = xgb.XGBClassifier(learning_rate=0.05, max_depth=6, min_child_weight=4, n_estimators=1000)
#    selection_model.fit(select_x_train, y_train)
#    
#    # Evaluate model
#    select_x_test = selection.transform(x_test)
#    y_pred = selection_model.predict(select_x_test)
#    predictions = [round(value) for value in y_pred]
#    accuracy = roc_auc_score(y_test,predictions)
#    print("Thresh=%.3f, n=%d, Accuracy: %.2f%%" % (thresh, select_x_train.shape[1], accuracy*100.0))
#
#
#    lst.append(roc_auc_score(y_test,predictions))    
#
#import pdb; pdb.set_trace()
#lst = np.array(lst) 
#feature = np.array(range(0,47))



#pred_test  = gbm.predict_proba(test[ 

#plt.figure(figsize=(8, 8))
#plt.plot(lst, color='darkorange', label='Accuracy curve')
#plt.xlim([0, 47])
#plt.ylim([0.6, 0.90])
#plt.grid(True)
#plt.show()


# After dropping  more than 15 features, accuracy score starts to drop at a more significant rate.

## Zoomed in view
#plt.figure(figsize=(8, 8))
#plt.plot(lst, color='darkorange', label='Accuracy curve')
#plt.xlim([0, 14])
#plt.ylim([0.824, 0.828])
#plt.grid(True)
#plt.show()


# Optimal number of features to drop: 5 

# ### GridSearchCV

# In[68]:

#from sklearn.grid_search import GridSearchCV


# In[66]:

#params = {'learning_rate': [0, 0.01, 0.05, 0.5]
#         ,'max_depth': [0, 2, 4, 6]
#         ,'min_child_weight':[4, 7, 11]
#         ,'n_estimators': [50, 100, 1000, 3000]}
#
#

#xgboost_gridsearch = GridSearchCV(gbm, params, n_jobs=5, scoring='roc_auc')


#xgboost_gridsearch.fit(x_train, y_train)


#print('XGBoost best gridsearch score: {}'.format(xgboost_gridsearch.best_score_))
#print('XGBoost best gridsearch params: {}'.format(xgboost_gridsearch.best_params_))


