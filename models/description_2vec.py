import pandas as pd
import timeit
from gensim import models, similarities

filepath = '~/Documents/Avito/'

df = pd.read_csv('ItemInfo_train3.2.csv', encoding='utf-8',usecols=['description_x_clean'], converters={'description_x_clean': lambda x: x.strip('[]').split(', ')})

#del df['Unnamed: 0']

print(df['description_x_clean'][0])
description = df['description_x_clean'].tolist()

print('No. of rows:{}'.format(len(description)))

start_time = timeit.default_timer()

model = models.word2vec.Word2Vec(description, sg=1, size=75, window =5,
                                         min_count=1, workers=3,seed=1)


model.save('description_2vecx')

elapsed = timeit.default_timer()-start_time
print('Time elapsed: {}'.format(elapsed))

