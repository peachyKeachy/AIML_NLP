import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer

sns.set_style("whitegrid")
plt.style.use("fivethirtyeight")

# example text for model training(SMS messasges)
# (vocabulary)
simple_train = ['call you tonight', 'call me a cab', 'please call me... PLEASE!']

# fucntion from sklearn
vect = CountVectorizer()

# creates vocabulary
vect.fit(simple_train)

# prints unique words from vocabulary
print(vect.get_feature_names_out())

# transform training data/ vocabulary into a 'document-term matrix
# computer will understand this way
simple_train_dtm = vect.transform(simple_train)
print(simple_train_dtm)

# turn vector I created into a matrix
print(simple_train_dtm.toarray())

# viewing vocabulary and document-term matrix together
df = pd.DataFrame(simple_train_dtm.toarray(), columns=vect.get_feature_names_out())
print('TRAIN DATAFRAME: ')
print(df)

# example text for model testing
simple_test = ["please dont call me"]

# transform testing data into a document-term metrix using existing vocabulary
simple_test_dtm = vect.transform(simple_test)
simple_test_dtm.toarray()

test_df = pd.DataFrame(simple_test_dtm.toarray(), columns = vect.get_feature_names_out())

print('TEST DATAFRAME: ')
print(test_df)