import numpy as np
import pandas as pd
import math
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

df = pd.read_csv('CryptoCoinClose.csv')

#df.drop( ['BNB'], axis = 1)

features = ['BTC', 'XRP', 'XLM', 'LTC' ]
X = df[features]
y = df['ETH']

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

rf_model = RandomForestRegressor(random_state=1, n_estimators = 300, max_depth=15)

rf_model.fit(train_X, train_y)

rf_pred = rf_model.predict(val_X)

print(rf_model.predict(X.tail(10)))
df['ETH'].tail(10)


print(mean_absolute_error(val_y,rf_pred))
#print('Score=',r2_score(val_y,rf_pred))
