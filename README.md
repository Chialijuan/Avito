# Avito Duplicate Ads Detection

## Pre-processing
* Concat data fields for itemID_1 and itemID_2. 
* Map `categoryID` and `locationID` to the concatenated data.

### Data cleaning
Data cleaning is mainly done on the `description`. 
* Removed punctuations, russian stopwords, filled Nan with 'None' and tokenized.

## Feature Engineering
* Difference in length of image arrays (`diff_len_imagearray`)
* Difference in latitude and longtitude (`haversine`)
* Difference in price (`priceDifference`)

* Compute token_set_ratio[__fuzzy wuzzy__] , levenshtein distance, jaro distance, jaro-winkler distance [__jelly_fish__]. 

  _`token_set_ratio` tokenizes both strings and compare the intersecting and non-intersecting group of words seperately. The intuition here is that if the intersection component is always exaclty the same, the score increases when that (a) makes up a later portion of the complete string and (b) the string remainders are more similar._
  
  _hamming distance omitted as it requires the strings in comparison to be of the same length. However, the implementation in [__jelly_fish__] considers extra characters as differing._

* Intersecting bag-of-words between the descriptions. 
* Non-intersecting bag-of-words between the descriptions.
* Clusters based on itemID_1 and itemID_2.
* Reverse Geolocation: city, region, neighbourhood, street, postalcode. (Dropped due to API crawling issues for test dataset)
* Compute similarity between descriptions using Word2Vec models. [__gensim__]
## Models
* xgBoost
* RandomForest
* Neural Network [__keras__]

### Model Ensembling
* Rank average 
* Geomean



