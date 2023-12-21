import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.model_selection import train_test_split


train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')
sample = pd.read_csv("sample_submission.csv")

Y = train_data['type']
train_data = train_data.drop(['id'], axis=1)
train_data = train_data.drop(['type'], axis=1)
test_data = test_data.drop(['id'], axis=1)

train_data['color'] = train_data['color'].replace({'clear': str(1/6)})
train_data['color'] = train_data['color'].replace({'green': str(2/6)})
train_data['color'] = train_data['color'].replace({'black': str(3/6)})
train_data['color'] = train_data['color'].replace({'blue': str(4/6)})
train_data['color'] = train_data['color'].replace({'white': str(5/6)})
train_data['color'] = train_data['color'].replace({'blood': str(6/6)})

test_data['color'] = test_data['color'].replace({'clear': str(1/6)})
test_data['color'] = test_data['color'].replace({'green': str(2/6)})
test_data['color'] = test_data['color'].replace({'black': str(3/6)})
test_data['color'] = test_data['color'].replace({'blue': str(4/6)})
test_data['color'] = test_data['color'].replace({'white': str(5/6)})
test_data['color'] = test_data['color'].replace({'blood': str(6/6)})



Y = [0 if y == 'Ghoul' else 1 if y == 'Goblin' else 2 for y in Y]


# print(Y)
# print(train_data.head(5))
# print(test_data.head(5))



X_train, X_test, y_train, y_test = train_test_split(train_data, Y, random_state=0, test_size=0.2)

#svm = SVC(kernel="linear")
svm = SVC(kernel="rbf") # gaussian

svm.fit(X_train, y_train)
y_pred = svm.predict(X_test)
print("f1 score: ", metrics.f1_score(y_test, y_pred, average='weighted'))

########################################################################################

#svm = SVC(kernel="linear")
svm = SVC(kernel="rbf") # gaussian
svm.fit(train_data, Y)
res = svm.predict(test_data)

type = ['Ghoul' if r == 0 else 'Goblin' if r == 1 else 'Ghost' for r in res]
type = pd.Series(type)


id = sample['id']

df = pd.DataFrame({'type': type})
df['id'] = id
df = df[['id', 'type']]
df.columns
print(df.head(5))

df.to_csv('svm_submission.csv', index=False)

