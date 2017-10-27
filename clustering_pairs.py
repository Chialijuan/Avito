"""
This module clusters rows together that contains the same items.
Clustering is done on ItemPairs.csv
"""

import pandas as pd
from pyclustering.cluster.rock import rock
from kmodes import kmodes

# Substitute for train/test
# Remove generationMethod for test data 
item_pairs = pd.read_csv('../ItemPairs_test.csv', encoding='utf-8', usecols=['itemID_1', 'itemID_2'])
print(item_pairs.head())
#rock_instance = rock(item_pairs, 1, 5, 0.5)

#rock_instance.process()

#clusters = rock_instance.get_clusters()

#visualizer = cluster_visualizer()
#visualizer.append_clusters(clusters,item_pairs)
#visualizer.show()


km = kmodes.KModes(n_clusters=4, init='Cao', n_init=5, verbose=1)
item_pairs['clusters'] = km.fit_predict(item_pairs)

# Save to train/test file
print('Saving to clustering_pairs_test.csv...')
item_pairs.to_csv('clustering_pairs_test.csv', encoding='utf-8', index=False)
