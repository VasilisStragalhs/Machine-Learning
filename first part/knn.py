import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics


train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")
sample = pd.read_csv("sample_submission.csv")

sample.shape
sample.head(10)
train_data.shape
cols = train_data.columns
train_data.head(10)


print(test_data.shape)
test_data.head(5)

Y = train_data['type']
train_data = train_data.drop(['id'], axis=1)
train_data = train_data.drop(['type'], axis=1)
test_data = test_data.drop(['id'], axis=1)


Y.head(5)


test_data.head(5)


train_data.head(5)


train_data['color'] = train_data['color'].replace({'clear': str(1/6)})
train_data['color'] = train_data['color'].replace({'green': str(2/6)})
train_data['color'] = train_data['color'].replace({'black': str(3/6)})
train_data['color'] = train_data['color'].replace({'blue': str(4/6)})
train_data['color'] = train_data['color'].replace({'white': str(5/6)})
train_data['color'] = train_data['color'].replace({'blood': str(6/6)})



train_data.head(5)


test_data['color'] = test_data['color'].replace({'clear': str(1/6)})
test_data['color'] = test_data['color'].replace({'green': str(2/6)})
test_data['color'] = test_data['color'].replace({'black': str(3/6)})
test_data['color'] = test_data['color'].replace({'blue': str(4/6)})
test_data['color'] = test_data['color'].replace({'white': str(5/6)})
test_data['color'] = test_data['color'].replace({'blood': str(6/6)})



test_data.head(5)


Y = [0 if y == 'Ghoul' else 1 if y == 'Goblin' else 2 for y in Y]



X_train, X_test, y_train, y_test = train_test_split(train_data, Y, random_state=0, test_size=0.2)


classifier = KNeighborsClassifier(n_neighbors=10, metric='euclidean')
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

print("f1 score: ", metrics.f1_score(y_test, y_pred, average='weighted'))


############################################################################

knn = KNeighborsClassifier(n_neighbors = 10)

knn.fit(train_data, Y)

res = knn.predict(test_data)

type = ['Ghoul' if r == 0 else 'Goblin' if r == 1 else 'Ghost' for r in res]
type = pd.Series(type)

id = sample['id']

df = pd.DataFrame({'type': type})
df['id'] = id
#df.columns = ['ImageId', 'Label']
df = df[['id', 'type']]
df.columns
print(df.head(5))
#df['type'].value_counts()


df.to_csv('knn_submission.csv', index=False)