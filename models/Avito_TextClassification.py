
# coding: utf-8

# #### Using intersection BOW and difference BOW as features for Naive Bayes, SGD.
# 
# #### Probabilities generated are then used as features for XGBoost later

# In[2]:

from __future__ import unicode_literals
import pandas as pd
from sklearn.model_selection import train_test_split


# In[ ]:

item_info = pd.read_csv('../ItemInfo_train.csv', encoding='utf-8', usecols=['itemID'])


# In[ ]:

desc_clean = pd.read_csv('./ItemInfo_train5.csv', encoding='utf-8',usecols=['description_x_clean','description_y_clean'], converters={'description_x_clean':lambda x: x.strip(u'[]').split(', '),
                                                                                                                                      'description_y_clean':lambda x: x.strip(u'[]').split(', ')})


# In[ ]:

item_pairs = pd.read_csv('../ItemPairs_train.csv', encoding='utf-8')
item_pairs.rename(columns={'itemID_1':'itemID'}, inplace=True)


# In[ ]:

df = pd.merge(item_info, item_pairs, on='itemID')
item_info.rename(columns={'itemID': 'itemID_2'}, inplace=True)
df = pd.merge(df, item_info, on='itemID_2')


# In[ ]:

df = pd.concat([df,desc_clean], axis=1)


# In[8]:

# Initial testing of 400 rows
#df = df.loc[:400,:]


# In[ ]:

set(df.description_x_clean[0])


# In[ ]:

set(df.description_x_clean[0]).intersection(set(df.description_y_clean[0]))


# In[28]:

# Feature engineering: intersecting bag of words
def intersection_BOW(df):
    x = df['description_x_clean']
    y = df['description_y_clean']

    return ' '.join(set(x).intersection(set(y)))


# In[29]:

# Feature engineering: Bag of words in either x or y but not both
def sym_diff_BOW(df):
    x = df['description_x_clean']
    y = df['description_y_clean']
    
    return ' '.join(set(x).symmetric_difference(set(y)))


# In[30]:

df['intersect_BOW'] = df[['description_x_clean','description_y_clean']].apply(intersection_BOW,axis=1)


# In[31]:

df['sym_diff_BOW'] = df[['description_x_clean','description_y_clean']].apply(sym_diff_BOW,axis=1)


# In[32]:

df.head()


# In[40]:

y = df.isDuplicate
x = df.intersect_BOW


# In[34]:

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
count_vect = CountVectorizer()
tfidf_vect = TfidfVectorizer()


# In[36]:

from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
nb = MultinomialNB()


# In[37]:

nb_count = make_pipeline(count_vect, nb)
nb_tfidf = make_pipeline(tfidf_vect, nb)


# In[ ]:

import pickle


#  TODO Run GridSearch to find best parameters for best model
#  Compute for both intersection and difference 
# 

# In[ ]:

nb_count.fit(x.loc[:50000,],y.loc[:50000,])
fn = 'nb_count.sav'
pickle.dump(nb_tfidf, open(fn), 'wb')


# In[ ]:

nb_tfidf.fit(x.loc[:50000,],y.loc[:50000,])
fn = 'nb_tfidf.sav'
pickle.dump(nb_tfidf, open(fn), 'wb')
# Using this for feature
#nb_tfidf.predict_proba(x.loc[350:,])


# In[49]:

# Scoring metrics
scores= ['accuracy', 'precision', 'recall', 'roc_auc']


# In[61]:

print('Score for CountVectorizer and NB:')
# Accuracy: ratio of correctly predicted observation to total observations
# TP+TN /TP+FP+TN+FN
print(cross_val_score(nb_count, x, y, cv=6, scoring='accuracy').mean())

# ROC_AUC: High TP, Low FP
print(cross_val_score(nb_count, x, y, cv=6, scoring='roc_auc').mean())

# Precision: TP/(TP+FP)
# Of all passengers that labeled as survived, how many actually survived?
print(cross_val_score(nb_count, x, y, cv=6, scoring='precision').mean())

# Recall: TP/TP+FN
#  Of all the passengers that truly survived, how many did we label?
print(cross_val_score(nb_count, x, y, cv=6, scoring='recall').mean())

# F1 score: weighted average of Precision and Recall
# 2*(Recall * Precision) / (Recall + Precision)


# F1 is usually more useful than accuracy, especially if you have an uneven class distribution. Accuracy works best if false positives and false negatives have similar cost. If the cost of false positives and false negatives are very different, it’s better to look at both Precision and Recall. 

# In[70]:

print('Score for tfidf and NB:')
print(cross_val_score(nb_tfidf, x, y, cv=6, scoring='accuracy').mean())
print(cross_val_score(nb_tfidf, x, y, cv=6, scoring='roc_auc').mean())


# In[43]:

# KNN
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()


# In[44]:

knn_count = make_pipeline(count_vect, knn)
knn_tfidf = make_pipeline(tfidf_vect, knn)


# In[71]:

print('Score for CountVectorizer and KNN:')
print(cross_val_score(knn_count, x, y, cv=6, scoring='accuracy').mean())
print(cross_val_score(knn_count, x, y, cv=6, scoring='roc_auc').mean())


# In[46]:

print('Score for tfidf and KNN:')
print(cross_val_score(knn_tfidf, x, y, cv=6, scoring='accuracy').mean())


# In[52]:

from sklearn.linear_model import SGDClassifier
sgd = SGDClassifier()


# In[53]:

sgd_count = make_pipeline(count_vect, sgd)
sgd_tfidf = make_pipeline(tfidf_vect, sgd)


# In[73]:

print('Score for CountVectorizer and SGD:')
print(cross_val_score(sgd_count, x, y, cv=6, scoring='accuracy').mean())
print(cross_val_score(sgd_count, x, y, cv=6, scoring='roc_auc').mean())
print(cross_val_score(sgd_count, x, y, cv=6, scoring='precision').mean())
print(cross_val_score(sgd_count, x, y, cv=6, scoring='recall').mean())


# In[55]:

print('Score for TfidfVectorizer and SGD:')
print(cross_val_score(sgd_tfidf, x, y, cv=6, scoring='accuracy').mean())


# In[77]:

from sklearn.model_selection import GridSearchCV


# In[75]:

# Parameters for GridSearch
param_grid_nb ={'multinomialnb__alpha':[0,0.025,0.05,0.1,0.3]}
param_grid_nb_tfidf = {'multinomialnb__alpha':[0,0.025,0.05,0.1,0.3]
                       ,'tfidfvectorizer__norm':['l1','l2']
                       ,'tfidfvectorizer__min_df':[0.0,0.03,0.1]
                       ,'tfidfvectorizer__use_idf':[True,False]}


# In[78]:

# Tfidf Naive Bayes
grid_nb_tfidf = GridSearchCV(nb_tfidf, param_grid_nb_tfidf, cv=5, scoring='accuracy')
grid_nb_tfidf.fit(x,y)

print(grid_nb_tfidf.best_score_)
print(grid_nb_tfidf.best_params_)


# In[80]:

grid_nb_count = GridSearchCV(nb_count, param_grid_nb, cv=5, scoring='accuracy')
get_ipython().magic(u'time grid_nb_count.fit(x,y)')

print(grid_nb_count.best_score_)
print(grid_nb_count.best_params_)


# In[81]:

from sklearn.model_selection import RandomizedSearchCV


# In[82]:

# Tfidf Naive Bayes
random_nb_tfidf = RandomizedSearchCV(nb_tfidf, param_grid_nb_tfidf, cv=5, scoring='accuracy', n_iter=5, random_state=1)
random_nb_tfidf.fit(x,y)
print(random_nb_tfidf.best_score_)
print(random_nb_tfidf.best_params_)

