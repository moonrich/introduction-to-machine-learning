import pandas as pd
from sklearn import tree


data_url = "https://gist.githubusercontent.com/merxer/0c08bf233971e99b3f77fedce6511f8c/raw/ff3e34aad81426583fb03bbd68e8965403a71e5a/orange_and_apple.csv"
df = pd.read_csv(data_url, nrows=6)

# Prepare Data
features = df[['Weight', 'Texture']].values
labels = df['Label'].values

# Select Classifier and Training
clf = tree.DecisionTreeClassifier()
# fit = find patterns in data
clf.fit(features, labels)


# dumps
import _pickle as pickle
dir(pickle)

with open('model_apple_orange.pickle', 'wb') as filename:
    pickle.dump(clf, filename)
    
# loads
with open('model_apple_orange.pickle', 'rb') as filename:
    clf2 = pickle.load(filename)

# Make Predictions
clf2.predict([[165, 0], [142, 1]])
