import pandas as pd
import timeit
from gensim import models, similarities

filepath = '~/Documents/Avito/'

df = pd.read_csv(filepath+'ItemInfo_train2.csv', encoding='utf-8', converters={"description_clean": lambda x: x.strip("[]").split(", ")})

del df['Unnamed: 0']

description = df['description_clean'].tolist()

print('No. of rows:{}'.format(len(description)))

start_time = timeit.default_timer()

model = models.word2vec.Word2Vec(description, sg=1, size=75, window =5,
                                         min_count=1, workers=3,seed=1)

model2 = model.wv
del model

model2.save('description_2vec.model')

elapsed = timeit.default_timer()-start_time
print('Time elapsed: {}'.format(elapsed))

