import pandas as pd
import timeit
from gensim import models

filepath = '~/Documents/Avito/'

df = pd.read_csv('ItemInfo_trainfull.csv', encoding='utf-8',usecols=['description_x_clean'], converters={'description_x_clean': lambda x: x.strip('[]').split(', ')})

#del df['Unnamed: 0']

print(df['description_x_clean'][0])
description = df['description_x_clean'].tolist()

print('No. of rows:{}'.format(len(description)))

start_time = timeit.default_timer()

model = models.wrappers.fasttext.FastText(sentences=description, sg=1, size=75, window =5, min_count=1, workers=3,seed=1)
#model.build_vocab(description)

model.save('description_fasttext')

elapsed = timeit.default_timer()-start_time
print('Time elapsed: {}'.format(elapsed))

