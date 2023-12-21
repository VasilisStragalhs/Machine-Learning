import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Activation, Dense
from sklearn import metrics
from sklearn.metrics import f1_score
########################################
from keras import backend as K
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score



train_data = pd.read_csv("train.csv")
print(train_data.head())

train_data.shape

train_data.info()

train_data['type'].value_counts()

test_data = pd.read_csv("test.csv")
print(test_data.head())

test_data.shape

sample = pd.read_csv("sample_submission.csv")


train_data = pd.concat([train_data, pd.get_dummies(train_data['color'])], axis=1)
train_data.head()


train_data = train_data.drop('color', axis=1)


test_data = pd.concat([test_data, pd.get_dummies(test_data['color'])], axis=1)
test_data = test_data.drop('color', axis=1)
test_data.head()


X = train_data.drop(['id', 'type'], axis=1)
y = pd.get_dummies(train_data['type'])

print(y.head())
print(X.head())

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, y, random_state=20, test_size=0.15)



model = Sequential()
model.add(Dense(64, input_shape=(X_train.shape[1],)))
model.add(Dense(200, activation='sigmoid',))
model.add(Dense(100, activation='sigmoid'))
model.add(Dense(3, activation='softmax'))
model.summary()

model.compile(optimizer='sgd',
             loss='categorical_crossentropy',
             metrics=['accuracy'])

model_data = model.fit(X_train, Y_train,
         validation_data=(X_test, Y_test),
         verbose=2,
         epochs=10,
         batch_size=16)

pred=model.predict(test_data.drop('id',axis=1))

# cm = confusion_matrix(X_test, Y_test)
# print(cm)



# f1_score = model.evaluate(X_test,  Y_test, verbose=2) 
# print("f1_score: ", f1_score)

pred_final=[np.argmax(i) for i in pred]
submission = pd.DataFrame({'id':test_data['id'], 'type':pred_final})
submission['type'].replace(to_replace=[0,1,2],value=['Ghost','Ghoul','Goblin'],inplace=True)
submission.to_csv('neural_submission.csv', index=False)

