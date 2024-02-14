import pickle
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.metrics import accuracy_score

dataset = pd.read_csv("datasets/test-regression.csv")
dataset.fillna(method='ffill', inplace=True)

X_test = dataset.iloc[:, 0:-1]
Y_test = dataset['Average User Rating']

X_test.drop('Subtitle', axis=1, inplace=True)
X_test.drop('Description', axis=1, inplace=True)
X_test.drop('In-app Purchases', axis=1, inplace=True)

# X_test['Languages'] = X_test['Languages'].fillna(method='ffill')
X_test = X_test.iloc[:, 4:]
X_test['Age Rating'] = X_test['Age Rating'].str.replace(
    '+', '', regex=False).astype(int)

orginalD = X_test['Original Release Date']
orginalD = pd.to_datetime(orginalD, dayfirst=True)
year = orginalD.dt.year
X_test['Original Release Date'] = year

currentD = X_test['Current Version Release Date']
currentD = pd.to_datetime(currentD, dayfirst=True)
year = currentD.dt.year
X_test['Current Version Release Date'] = year

Y_test = Y_test.values.reshape(-1, 1)
Y_test = pd.DataFrame(Y_test, columns=['Rate'])

# ---------------------------------------------------------encoding

cols = (
    'Developer', 'Primary Genre'
)
# print(X_test['Developer'])

# Load the encoders from the pickle file
with open('models/encoders.pkl', 'rb') as file:
    encoders = pickle.load(file)

# Apply encoding on new data


for c in cols:
    X_test[c] = encoders[c].transform(list(X_test[c].values))


# --------------------------------------encoding lissts
encoders = {}
with open('models/encoder_lists.pkl', 'rb') as file:
    encoders = pickle.load(file)

list_cols = ('Languages', 'Genres')

for c in list_cols:
    X_test[c] = X_test[c].apply(lambda x: x.split(', '))
    names = encoders[c].classes_
    encoded_value = encoders[c].transform(names)
    encoded_dict = dict(zip(names, encoded_value))
    X_test[c] = X_test[c].apply(
        lambda x: [encoded_dict[lang] if lang in encoded_dict else 0 for lang in x])
    X_test[c] = X_test[c].apply(sum)
# --------------------------------------------------------scaling

with open('models/Scaling.pkl', 'rb') as file:
    savingScale = pickle.load(file)

with open('models/SCsize.pkl', 'rb') as file:
    SCsize = pickle.load(file)

sizeT = X_test.iloc[:, 5:6]
X_test.drop('Size', axis=1, inplace=True)
scaler = savingScale
X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
scaler = SCsize
sizeT = pd.DataFrame(scaler.transform(sizeT), columns=sizeT.columns)
X_test = pd.concat([X_test, sizeT], axis=1)

with open('models/normalizeY.pkl', 'rb') as file:
    saveNormalize = pickle.load(file)

Y_test = saveNormalize.transform(Y_test)
Y_test = pd.DataFrame(Y_test, columns=['Average User Rating'])

# -------------------------------------------------selection
with open('models/selection.pkl', 'rb') as file:
    selected_features = pickle.load(file)

X_test = X_test.iloc[:, selected_features]

# ------------------------------------------------models
# polynomial model

with open('models/polyFeatures.pkl', 'rb') as file:
    polyF = pickle.load(file)

X_test_poly = polyF.transform(X_test)
filename = 'models/polynomial_regression_model.pkl'
loaded_model = pickle.load(open(filename, 'rb'))
pred = loaded_model.predict(X_test_poly)
print('polynomial_regression')
print('Mean Square Error', metrics.mean_squared_error(Y_test, pred))
print("------------------------------------")

# multiple_regression
filename = 'models/multiple_regression_model.pkl'
loaded_model = pickle.load(open(filename, 'rb'))
pred = loaded_model.predict(X_test)
print('multiple_regression')
print('Mean Square Error', metrics.mean_squared_error(Y_test, pred))
print("------------------------------------")


# support_vector_regressor
filename = 'models/support_vector_regressor_model.pkl'
loaded_model = pickle.load(open(filename, 'rb'))
pred = loaded_model.predict(X_test)
print('support_vector_regressor')
print('Mean Square Error', metrics.mean_squared_error(Y_test, pred))
print("------------------------------------")
