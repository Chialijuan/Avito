import pandas as pd
from gensim import models
import timeit

# Reading in cleaned data
filepath = '~/Documents/Avito/'
df = pd.read_csv(filepath+'ItemInfo_train2.csv', encoding='utf-8', converters={"description_clean": lambda x: x.strip("[]").split(", ")}, usecols=['description_clean'])

# Convert to TaggedDocument
def read_doc(df):
    lst =[]
    for i, row in enumerate(df['description_clean']):
        lst.append(models.doc2vec.TaggedDocument(row,[i]))
    return lst

# Training corpus
train_corpus = list(read_doc(df))
print(train_corpus[:2])

del df

start_time = timeit.default_timer()

# Doc2Vec object
model_doc = models.doc2vec.Doc2Vec(size=75, window=4,min_count=1, workers=4)

# Build Vocab
model_doc.build_vocab(train_corpus)

print('Training of model...')
# Train Doc2Vec model
model_doc.train(train_corpus, total_examples=model_doc.corpus_count, epochs=model_doc.iter)

del train_corpus

print('Saving model...')
# Save model
model_doc.save('description_doc')

elapsed = timeit.default_timer()-start_time
print('Time elapsed: {}'.format(elapsed))
