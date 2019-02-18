import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import keras
from keras.layers import Dense
from keras.models import Sequential
from keras.utils import to_categorical


#%%

df=pd.read_csv('hourly_wages.csv')

print(df.shape)

print(df.head())
data=df.describe().T

print(data)
#%%

# =============================================================================
#     Of the 9 predictor variables in the DataFrame, how many are binary indicators? 
#     The min and max values as shown by . describe() will be informative here.
# =============================================================================

dataMinMax=data[data[['min','max']].isin([0,1])][['min','max']].dropna().count()
print(dataMinMax)
#%%
predictors=df.drop('wage_per_hour',axis=1).values
target=df['wage_per_hour'].values
#%%
# Import necessary modules
import keras
from keras.layers import Dense
from keras.models import Sequential

# Save the number of columns in predictors: n_cols
n_cols = predictors.shape[1]

# Set up the model: model
model = Sequential()

# Add the first layer
model.add(Dense(50, activation='relu', input_shape=(n_cols,)))

# Add the second layer
model.add(Dense(32, activation='relu'))

# Add the output layer
model.add(Dense(1))

#%%

# Import necessary modules
import keras
from keras.layers import Dense
from keras.models import Sequential

# Specify the model
n_cols = predictors.shape[1]
model = Sequential()
model.add(Dense(50, activation='relu', input_shape = (n_cols,)))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Verify that model contains information from compiling
print("Loss function: " + model.loss)


#%%


# Import necessary modules
import keras
from keras.layers import Dense
from keras.models import Sequential

# Specify the model
n_cols = predictors.shape[1]
model = Sequential()
model.add(Dense(50, activation='relu', input_shape = (n_cols,)))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Fit the model
model.fit(predictors,target,epochs=10)

#%%

#df=pd.read_csv('https://assets.datacamp.com/production/course_1975/datasets/titanic_all_numeric.csv')
#df.to_csv('titanic_all_numeric.csv',index=False)
df=pd.read_csv('titanic_all_numeric.csv')
print(df.head())
print(df.shape)

predictors=df.drop('survived',axis=1).as_matrix()

n_cols=predictors.shape[1]
print(predictors)
#%%

df.describe()['age']

#%%


# Import necessary modules
import keras
from keras.layers import Dense
from keras.models import Sequential
from keras.utils import to_categorical

# Convert the target to categorical: target
target = to_categorical(df.survived)

# Set up the model
model = Sequential()

# Add the first layer
model.add(Dense(32,activation='relu',input_shape=(n_cols,)))

# Add the output layer
model.add(Dense(2,activation='softmax'))

# Compile the model
model.compile(optimizer='sgd', loss='categorical_crossentropy',metrics=['accuracy'] )

# Fit the model
model.fit(predictors,target,epochs=10)


#%%

pred_data=np.array([[2, 34.0, 0, 0, 13.0, 1, False, 0, 0, 1],       [2, 31.0, 1, 1, 26.25, 0, False, 0, 0, 1],       [1, 11.0, 1, 2, 120.0, 1, False, 0, 0, 1],
       [3, 0.42, 0, 1, 8.5167, 1, False, 1, 0, 0],       [3, 27.0, 0, 0, 6.975, 1, False, 0, 0, 1],       [3, 31.0, 0, 0, 7.775, 1, False, 0, 0, 1],
       [1, 39.0, 0, 0, 0.0, 1, False, 0, 0, 1],       [3, 18.0, 0, 0, 7.775, 0, False, 0, 0, 1],       [2, 39.0, 0, 0, 13.0, 1, False, 0, 0, 1],
       [1, 33.0, 1, 0, 53.1, 0, False, 0, 0, 1],       [3, 26.0, 0, 0, 7.8875, 1, False, 0, 0, 1],       [3, 39.0, 0, 0, 24.15, 1, False, 0, 0, 1],
       [2, 35.0, 0, 0, 10.5, 1, False, 0, 0, 1],       [3, 6.0, 4, 2, 31.275, 0, False, 0, 0, 1],       [3, 30.5, 0, 0, 8.05, 1, False, 0, 0, 1],
       [1, 29.69911764705882, 0, 0, 0.0, 1, True, 0, 0, 1],       [3, 23.0, 0, 0, 7.925, 0, False, 0, 0, 1],       [2, 31.0, 1, 1, 37.0042, 1, False, 1, 0, 0],
       [3, 43.0, 0, 0, 6.45, 1, False, 0, 0, 1],       [3, 10.0, 3, 2, 27.9, 1, False, 0, 0, 1],       [1, 52.0, 1, 1, 93.5, 0, False, 0, 0, 1],
       [3, 27.0, 0, 0, 8.6625, 1, False, 0, 0, 1],       [1, 38.0, 0, 0, 0.0, 1, False, 0, 0, 1],       [3, 27.0, 0, 1, 12.475, 0, False, 0, 0, 1],
       [3, 2.0, 4, 1, 39.6875, 1, False, 0, 0, 1],       [3, 29.69911764705882, 0, 0, 6.95, 1, True, 0, 1, 0],       [3, 29.69911764705882, 0, 0, 56.4958, 1, True, 0, 0, 1],
       [2, 1.0, 0, 2, 37.0042, 1, False, 1, 0, 0],       [3, 29.69911764705882, 0, 0, 7.75, 1, True, 0, 1, 0],       [1, 62.0, 0, 0, 80.0, 0, False, 0, 0, 0],
       [3, 15.0, 1, 0, 14.4542, 0, False, 1, 0, 0],       [2, 0.83, 1, 1, 18.75, 1, False, 0, 0, 1],       [3, 29.69911764705882, 0, 0, 7.2292, 1, True, 1, 0, 0],
       [3, 23.0, 0, 0, 7.8542, 1, False, 0, 0, 1],       [3, 18.0, 0, 0, 8.3, 1, False, 0, 0, 1],       [1, 39.0, 1, 1, 83.1583, 0, False, 1, 0, 0],
       [3, 21.0, 0, 0, 8.6625, 1, False, 0, 0, 1],       [3, 29.69911764705882, 0, 0, 8.05, 1, True, 0, 0, 1],       [3, 32.0, 0, 0, 56.4958, 1, False, 0, 0, 1],
       [1, 29.69911764705882, 0, 0, 29.7, 1, True, 1, 0, 0],       [3, 20.0, 0, 0, 7.925, 1, False, 0, 0, 1],       [2, 16.0, 0, 0, 10.5, 1, False, 0, 0, 1],
       [1, 30.0, 0, 0, 31.0, 0, False, 1, 0, 0],       [3, 34.5, 0, 0, 6.4375, 1, False, 1, 0, 0],       [3, 17.0, 0, 0, 8.6625, 1, False, 0, 0, 1],
       [3, 42.0, 0, 0, 7.55, 1, False, 0, 0, 1],       [3, 29.69911764705882, 8, 2, 69.55, 1, True, 0, 0, 1],       [3, 35.0, 0, 0, 7.8958, 1, False, 1, 0, 0],
       [2, 28.0, 0, 1, 33.0, 1, False, 0, 0, 1],       [1, 29.69911764705882, 1, 0, 89.1042, 0, True, 1, 0, 0],       [3, 4.0, 4, 2, 31.275, 1, False, 0, 0, 1],
       [3, 74.0, 0, 0, 7.775, 1, False, 0, 0, 1],       [3, 9.0, 1, 1, 15.2458, 0, False, 1, 0, 0],       [1, 16.0, 0, 1, 39.4, 0, False, 0, 0, 1],
       [2, 44.0, 1, 0, 26.0, 0, False, 0, 0, 1],       [3, 18.0, 0, 1, 9.35, 0, False, 0, 0, 1],       [1, 45.0, 1, 1, 164.8667, 0, False, 0, 0, 1],
       [1, 51.0, 0, 0, 26.55, 1, False, 0, 0, 1],       [3, 24.0, 0, 3, 19.2583, 0, False, 1, 0, 0],       [3, 29.69911764705882, 0, 0, 7.2292, 1, True, 1, 0, 0],
       [3, 41.0, 2, 0, 14.1083, 1, False, 0, 0, 1],       [2, 21.0, 1, 0, 11.5, 1, False, 0, 0, 1],       [1, 48.0, 0, 0, 25.9292, 0, False, 0, 0, 1],
       [3, 29.69911764705882, 8, 2, 69.55, 0, True, 0, 0, 1],       [2, 24.0, 0, 0, 13.0, 1, False, 0, 0, 1],       [2, 42.0, 0, 0, 13.0, 0, False, 0, 0, 1],
       [2, 27.0, 1, 0, 13.8583, 0, False, 1, 0, 0],       [1, 31.0, 0, 0, 50.4958, 1, False, 0, 0, 1],       [3, 29.69911764705882, 0, 0, 9.5, 1, True, 0, 0, 1],
       [3, 4.0, 1, 1, 11.1333, 1, False, 0, 0, 1],       [3, 26.0, 0, 0, 7.8958, 1, False, 0, 0, 1],
       [1, 47.0, 1, 1, 52.5542, 0, False, 0, 0, 1],       [1, 33.0, 0, 0, 5.0, 1, False, 0, 0, 1],       [3, 47.0, 0, 0, 9.0, 1, False, 0, 0, 1],
       [2, 28.0, 1, 0, 24.0, 0, False, 1, 0, 0],       [3, 15.0, 0, 0, 7.225, 0, False, 1, 0, 0],       [3, 20.0, 0, 0, 9.8458, 1, False, 0, 0, 1],
       [3, 19.0, 0, 0, 7.8958, 1, False, 0, 0, 1],       [3, 29.69911764705882, 0, 0, 7.8958, 1, True, 0, 0, 1],       [1, 56.0, 0, 1, 83.1583, 0, False, 1, 0, 0],
       [2, 25.0, 0, 1, 26.0, 0, False, 0, 0, 1],       [3, 33.0, 0, 0, 7.8958, 1, False, 0, 0, 1],       [3, 22.0, 0, 0, 10.5167, 0, False, 0, 0, 1],
       [2, 28.0, 0, 0, 10.5, 1, False, 0, 0, 1],       [3, 25.0, 0, 0, 7.05, 1, False, 0, 0, 1],       [3, 39.0, 0, 5, 29.125, 0, False, 0, 1, 0],
       [2, 27.0, 0, 0, 13.0, 1, False, 0, 0, 1],       [1, 19.0, 0, 0, 30.0, 0, False, 0, 0, 1],       [3, 29.69911764705882, 1, 2, 23.45, 0, True, 0, 0, 1],
       [1, 26.0, 0, 0, 30.0, 1, False, 1, 0, 0],       [3, 32.0, 0, 0, 7.75, 1, False, 0, 1, 0]], dtype=object)


#%%

# Specify, compile, and fit the model
model = Sequential()
model.add(Dense(32, activation='relu', input_shape = (n_cols,)))
model.add(Dense(2, activation='softmax'))
model.compile(optimizer='sgd', 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

his=model.fit(predictors, target,epochs=10)
print(pd.DataFrame(his.history['acc']).rename(columns={0:'accuracy'}).describe())

# Calculate predictions: predictions
predictions = model.predict(pred_data)

# Calculate predicted probability of survival: predicted_prob_true
predicted_prob_true = predictions[:,1]

# print predicted_prob_true
print(predicted_prob_true)






