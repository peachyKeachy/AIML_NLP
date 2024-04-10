import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer

sns.set_style("whitegrid")
plt.style.use("fivethirtyeight")

# example text for model training(SMS messasges)
simple_train = ['call you tonight', 'call me a cab', 'please call me... PLEASE!']

vect = CountVectorizer()
vect.fit(simple_train)
print(vect.get_feature_names_out())

