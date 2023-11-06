import pickle

import pandas as pd
from matplotlib import pyplot as plt
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer, PolynomialFeatures
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
import operator
import numpy as np
from sklearn.svm import SVR


def featureScaling(X, xtest):
    size = X.iloc[:, 5:6]
    sizeT = xtest.iloc[:, 5:6]

    X.drop('Size', axis=1, inplace=True)
    xtest.drop('Size', axis=1, inplace=True)
    scaler = StandardScaler()
    scaler = scaler.fit(X)
    savingScale = scaler
    X = pd.DataFrame(scaler.transform(X), columns=X.columns)
    xtest = pd.DataFrame(scaler.transform(xtest), columns=X.columns)
    scaler = MinMaxScaler()
    scaler = scaler.fit(size)
    saveSizeSale = scaler
    size = pd.DataFrame(scaler.transform(size), columns=size.columns)
    sizeT = pd.DataFrame(scaler.transform(sizeT), columns=size.columns)

    X = pd.concat([X, size], axis=1)
    xtest = pd.concat([xtest, sizeT], axis=1)

    return X, xtest, savingScale, saveSizeSale


def normalizedY(y, ytest):
    y = y.values.reshape(-1, 1)
    ytest = ytest.values.reshape(-1, 1)
    scaler = MinMaxScaler()
    scaler= scaler.fit(y)
    saveNormalize = scaler
    y = scaler.transform(y)
    ytest = scaler.transform(ytest)
    y = pd.DataFrame(y, columns=['Average User Rating'])
    ytest = pd.DataFrame(ytest, columns=['Average User Rating'])
    return y, ytest,saveNormalize


# def feature_encoding(X, cols, xtest):
#     for c in cols:
#         xtest[c] = xtest[c].fillna(method='ffill')
#         X[c] = X[c].apply(lambda x: x.split(', '))
#         xtest[c] = xtest[c].apply(lambda x: x.split(', '))
#
#         mlb = MultiLabelBinarizer()
#         encoded = mlb.fit_transform(X[c])
#
#         encoded_xt = mlb.transform(xtest[c])
#
#         # create a new dataframe with the encoded values
#         encoded_df = pd.DataFrame(encoded, columns=mlb.classes_)
#         df_encoded = pd.concat([X[c], encoded_df], axis=1)
#
#         encoded_df_xt = pd.DataFrame(encoded_xt, columns=mlb.classes_)
#         df_encoded_xt = pd.concat([xtest[c], encoded_df_xt], axis=1)
#
#         # drop the original 'colors' column
#         df_encoded = df_encoded.drop(c, axis=1)
#         X = X.drop(c, axis=1)
#         X = pd.concat([X, df_encoded], axis=1)
#
#         # drop the original 'colors' column
#
#         df_encoded_xt = df_encoded_xt.drop(c, axis=1)
#         xtest = xtest.drop(c, axis=1)
#         xtest = pd.concat([xtest, df_encoded_xt], axis=1)
#
#     return X, xtest


def Feature_Encoder(X, cols, xtest):
    encoders = {}
    for c in cols:
        lbl = LabelEncoder()
        lbl.fit(list(X[c].values) + list(xtest[c].values))
        encoders[c] = lbl
        X[c] = lbl.transform(list(X[c].values))
        xtest[c] = lbl.transform(list(xtest[c].values))
    return X, xtest,encoders


def fun(X, cols, xtest):
    encoders = {}
    for c in cols:
        X[c] = X[c].apply(lambda x: x.split(', '))
        xtest[c] = xtest[c].apply(lambda x: x.split(', '))
        values = np.unique(X[c])
        unique_values = set()
        for lang in values:
            unique_values.update(lang)

        lbl = LabelEncoder()

        lbl.fit(list(unique_values))

        encoders[c] = lbl

        names = lbl.classes_
        encoded_value = lbl.transform(names)
        encoded_dict = dict(zip(names, encoded_value))
        X[c] = X[c].apply(lambda x: [encoded_dict[lang] for lang in x])
        xtest[c] = xtest[c].apply(lambda x: [encoded_dict[lang] if lang in encoded_dict else 0 for lang in x])

        X[c] = X[c].apply(sum)
        xtest[c] = xtest[c].apply(sum)

    return X, xtest,encoders


def feature_selection(X, Y):
    constant_features = []
    X = pd.DataFrame(X)
    for feature in X.columns:
        if X[feature].std() == 0:
            constant_features.append(feature)

    # remove constant features from dataset
    X.drop(labels=constant_features, axis=1, inplace=True)
    selector = SelectKBest(f_classif, k=2)
    Y = np.ravel(Y)
    selector.fit_transform(X, Y)
    selected_features = selector.get_support(indices=True)
    return selected_features


def pre_processing(X, y, xtest, y_test):
    X.drop('Subtitle', axis=1, inplace=True)
    X.drop('Description', axis=1, inplace=True)
    X.drop('In-app Purchases', axis=1, inplace=True)
    X['Languages'] = X['Languages'].fillna(method='ffill')
    X = X.iloc[:, 4:]
    X['Age Rating'] = X['Age Rating'].str.replace('+', '', regex=False).astype(int)
    orginalD = X['Original Release Date']
    orginalD = pd.to_datetime(orginalD, dayfirst=True)
    year = orginalD.dt.year
    X['Original Release Date'] = year

    currentD = X['Current Version Release Date']
    currentD = pd.to_datetime(currentD, dayfirst=True)
    year = currentD.dt.year
    X['Current Version Release Date'] = year

    xtest.drop('Subtitle', axis=1, inplace=True)
    xtest.drop('Description', axis=1, inplace=True)
    xtest.drop('In-app Purchases', axis=1, inplace=True)
    xtest['Languages'] = xtest['Languages'].fillna(method='ffill')
    xtest = xtest.iloc[:, 4:]
    xtest['Age Rating'] = xtest['Age Rating'].str.replace('+', '', regex=False).astype(int)

    orginalD = xtest['Original Release Date']
    orginalD = pd.to_datetime(orginalD, dayfirst=True)
    year = orginalD.dt.year
    xtest['Original Release Date'] = year

    currentD = xtest['Current Version Release Date']
    currentD = pd.to_datetime(currentD, dayfirst=True)
    year = currentD.dt.year
    xtest['Current Version Release Date'] = year

    # -------------Encoding----------

    cols = (
        'Developer', 'Primary Genre'
    )
    cols2 = ('Languages', 'Genres')



    X, xtest,encoders = Feature_Encoder(X, cols, xtest)

    # Save the encoders using pickle
    with open('encoders.pkl', 'wb') as file:
        pickle.dump(encoders, file)


    X, xtest,encoders = fun(X, cols2, xtest)

    with open('encoder_lists.pkl', 'wb') as file:
        pickle.dump(encoders, file)


    # -------------Scaling----------

    X, xtest,savingScale,saveSizeSale = featureScaling(X, xtest)

    with open('Scaling.pkl', 'wb') as file:
        pickle.dump(savingScale, file)

    with open('SCsize.pkl', 'wb') as file:
        pickle.dump(saveSizeSale, file)

    y, y_test,saveNormalize = normalizedY(y, y_test)

    with open('normalizeY.pkl', 'wb') as file:
        pickle.dump(saveNormalize, file)

    # -------------Selection----------

    selected_features = feature_selection(X, y)

    with open('selection.pkl', 'wb') as file:
        pickle.dump(selected_features, file)

    X = X.iloc[:, selected_features]
    xtest = xtest.iloc[:, selected_features]
    return X, y, xtest, y_test


def polynomial_regression(X_train, y_train, X_test, degree):
    poly = PolynomialFeatures(degree=degree)
    polyF = poly.fit(X_train)

    with open('polyFeatures.pkl', 'wb') as file:
        pickle.dump(polyF, file)

    X_train_poly = poly.transform(X_train)
    X_test_poly = poly.transform(X_test)
    model = LinearRegression()
    y_train = np.ravel(y_train)
    polymodel = model.fit(X_train_poly, y_train)

    with open('polynomial_regression_model.pkl', 'wb') as file:
        pickle.dump(polymodel, file)

    y_pred = model.predict(X_test_poly)

    sort_axis = operator.itemgetter(0)
    sorted_zip = sorted(zip(X_train, y_train, X_train_poly), key=sort_axis)
    X_train, y_train, X_train_poly = zip(*sorted_zip)

    plt.scatter(X_train, y_train, color='blue')
    plt.plot(X_train, model.predict(X_train_poly), color='red')
    plt.title('Polynomial Regression')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.show()

    return y_pred


def multiple_regression(X, y, X_test):
    reg = LinearRegression()
    y = np.ravel(y)
    multiReg = reg.fit(X, y)

    with open('multiple_regression_model.pkl', 'wb') as file:
        pickle.dump(multiReg, file)

    # Generate predictions for all combinations of x and y in the test set
    x_min, x_max = X_test.iloc[:, 0].min() - 1, X_test.iloc[:, 0].max() + 1
    y_min, y_max = X_test.iloc[:, 1].min() - 1, X_test.iloc[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
    X_mesh = np.column_stack([xx.ravel(), yy.ravel()])
    y_mesh = reg.predict(X_mesh)

    # Reshape the predictions to match the shape of xx and yy
    zz = y_mesh.reshape(xx.shape)

    # Plot the training data points, test data points, and the predicted surface
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(X.iloc[:, 0], X.iloc[:, 1], y_train, c='r', marker='o')
    ax.scatter(X_test.iloc[:, 0], X_test.iloc[:, 1], y_test, c='b', marker='o')
    ax.plot_surface(xx, yy, zz, cmap='coolwarm', alpha=0.5)
    ax.set_xlabel('Original Release Date')
    ax.set_ylabel('Current Version Release Date')
    ax.set_zlabel('Average User Rating')
    plt.show()

    y_pred = reg.predict(X_test)

    return y_pred


def svr(X_train, Y_train, X_test):
    # create SVR model with default hyperparameters
    svr_model = SVR()
    Y_train = np.ravel(Y_train)
    # train the model on training data
    Svrm = svr_model.fit(X_train, Y_train)

    with open('support_vector_regressor_model.pkl', 'wb') as file:
        pickle.dump(Svrm, file)

    # make predictions on test data
    y_pred = svr_model.predict(X_test)

    # plot the training data
    plt.scatter(X_train.iloc[:, 0], X_train.iloc[:, 1], c=Y_train, cmap='coolwarm')
    # plot the support vectors
    plt.scatter(svr_model.support_vectors_[:, 0], svr_model.support_vectors_[:, 1], s=100, facecolors='none',
                edgecolors='black')
    plt.xlabel('Original Release Date')
    plt.ylabel('Current Version Release Date')
    plt.title('SVR Model')
    plt.show()

    return y_pred


def test_train_data(X, y, X_test, y_test):
    pred = multiple_regression(X, y, X_test)
    print('multiple_regression')
    print('Mean Square Error', metrics.mean_squared_error(y_test, pred))
    print('Accuracy', metrics.r2_score(y_test, pred) * 100, '%')

    pred2 = polynomial_regression(X, y, X_test, 3)
    print('polynomial_regression')
    print('Mean Square Error', metrics.mean_squared_error(y_test, pred2))
    print('Accuracy', metrics.r2_score(y_test, pred2) * 100, '%')

    pred3 = svr(X, y, X_test)
    print('Support vector regressor')
    print('Mean Square Error', metrics.mean_squared_error(y_test, pred3))
    print('Accuracy', metrics.r2_score(y_test, pred3) * 100, '%')


"""def extract_feature(dataset):
        doc = nlp(dataset)
        # Extract named entities
        entities = [ent.text for ent in doc.ents]
        # Extract parts of speech
        pos_tags = [token.pos_ for token in doc]
        # Extract lemmas
        lemmas = [token.lemma_ for token in doc]
        # Return a dictionary of features
        return {'entities': entities, 'pos_tags': pos_tags, 'lemmas': lemmas}
"""

# read data
dataset = pd.read_csv("games-regression-dataset.csv")
X = dataset.iloc[:, 0:-1]
y = dataset['Average User Rating']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, shuffle=True, random_state=10)

# dataset = extract_feature(data)
xTrain, y_train, X_test, y_test = pre_processing(X_train, y_train, X_test, y_test)

test_train_data(xTrain, y_train, X_test, y_test)
# dataset['Description'] = dataset['Description'].apply(extract_feature)

