import pandas as pd 
import timeit
from gensim import models, similarities

df = pd.read_csv('ItemInfo_test5.csv',encoding='utf-8'
                 ,converters={'description_x_clean': lambda x: x.strip(u'[]').split(', ')
                 ,'description_y_clean': lambda x: x.strip('[]').split(', ')})
##### FastText
print('Loading model...')
MODEL_FASTTEXT = models.wrappers.fasttext.FastText.load('description_fasttext')

def ft_sim(df):
    return MODEL_FASTTEXT.wv.n_similarity(df['description_x_clean'], df['description_y_clean'])

## Start Timing
start_time = timeit.default_timer()
df['ft_sim'] = df[['description_x_clean', 'description_y_clean']].apply(ft_sim, axis=1)

print('Deleting model and df...')
del MODEL_FASTTEXT
df.drop(['description_x_clean','description_y_clean'], axis=1, inplace=True)

print('No. of null:{}'.format(df.ft_sim.isnull().sum()))
print(df.head())
elapsed = timeit.default_timer() - start_time
print('Time elapsed: {}'.format(elapsed))

df.to_csv('description_similarity_ft_test.csv', encoding='utf-8', index=False)
print('Saved to description_similarity_ft_test.csv')

