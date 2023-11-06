# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import mean_absolute_error
# from sklearn.metrics import accuracy_score
# import pandas as pd

# import joblib
    
# from sklearn.preprocessing import LabelEncoder
# file_path = '../../data/GlobalWeatherRepository.csv'
# data = pd.read_csv(file_path) 

# data = data.dropna(axis=0)

# le_y = LabelEncoder()
# data['condition_text'] = le_y.fit_transform(data['condition_text'])

# y = data['condition_text']  

# features = ['country', 'last_updated']
# X = data[features]

# le_X = LabelEncoder()
# X.country = le_X.fit_transform(X.country)
# X.last_updated = le_X.fit_transform(X.last_updated)

# print(X.head())
# train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0)

# forest_model = RandomForestClassifier()
# forest_model.fit(train_X, train_y)
# y_pred = forest_model.predict(train_X)

# # Use inverse_transform to get original 'condition_text'
# y_pred = le_y.inverse_transform(y_pred)
# print (y_pred)

# joblib.dump(le_y, 'le_y.joblib')
# joblib.dump(forest_model, 'randomforestweather.joblib')

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import accuracy_score
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

file_path = 'data/GlobalWeatherRepository.csv'
data = pd.read_csv(file_path) 

data = data.dropna(axis=0)

le_y = LabelEncoder()
data['condition_text'] = le_y.fit_transform(data['condition_text'])

y = data['condition_text']  

# Convert 'last_updated' to datetime
data['last_updated'] = pd.to_datetime(data['last_updated'])

# Extract components from 'last_updated'
data['year'] = data['last_updated'].dt.year
data['month'] = data['last_updated'].dt.month
data['day'] = data['last_updated'].dt.day
data['hour'] = data['last_updated'].dt.hour

features = ['country', 'year', 'month', 'day', 'hour']
X = data[features]
print(X.head())

le_X = LabelEncoder()
X.country = le_X.fit_transform(X.country)

print(X.head())
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0)

forest_model = RandomForestClassifier()
forest_model.fit(train_X, train_y)
y_pred = forest_model.predict(train_X)

# Use inverse_transform to get original 'condition_text'
y_pred = le_y.inverse_transform(y_pred)
print (y_pred)

joblib.dump(le_y, 'le_y.joblib')
joblib.dump(forest_model, 'randomforestweather.joblib')
