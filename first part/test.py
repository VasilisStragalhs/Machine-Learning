import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf




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


model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(100, activation='sigmoid'), # 1o krimmeno epipedo me sigmoidi
    #tf.keras.layers.Dense(200, activation='sigmoid'), # 2o krimmeno epipedo me sigmoidi
    tf.keras.layers.Dense(10, activation='softmax') # 10 neurones eksodou
])

opt = tf.keras.optimizers.SGD(learning_rate=0.9) 

model.compile(optimizer=opt,loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),metrics=['accuracy'])

model.fit(train_data, Y, epochs=10) 

res = model.predict(test_data)



type = ['Ghoul' if r == 0 else 'Goblin' if r == 1 else 'Ghost' for r in res]
type = pd.Series(type)


id = sample['id']

df = pd.DataFrame({'type': type})
df['id'] = id
df = df[['id', 'type']]
df.columns
print(df.head(5))

df.to_csv('neural_submission.csv', index=False)