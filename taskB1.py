import numpy as np
from sklearn.metrics import classification_report, accuracy_score
from sklearn.svm import SVC
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNetCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier


# neural network class definition


def data_preprocessing(X, Y):
    # scaling of data

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
    X_train = X_train.reshape((6252, 68 * 2))
    X_test = X_test.reshape((1563, 68 * 2))
    scaler = StandardScaler()  # doctest: +SKIP
    # Don't cheat - fit only on training data
    scaler.fit(X_train)  # doctest: +SKIP
    X_train = scaler.transform(X_train)  # doctest: +SKIP
    # apply same transformation to test data
    X_test = scaler.transform(X_test)

    # PCA analysis

    """pca = PCA(n_components=68)
    pca.fit(X_train)
    X_train = pca.transform(X_train)
    X_test = pca.transform(X_test)

    # Feature selection using lasso and ridge regression

    ElasticNet = ElasticNetCV(cv=10, random_state=0)
    ElasticNet.fit(X_train, Y_train)
    all_features = ElasticNet.coef_
    not_important_features_indices = np.where(all_features == 0)[0]
    X_train = np.delete(X_train, not_important_features_indices, axis=1)
    X_test = np.delete(X_test, not_important_features_indices, axis=1)"""
    return X_train, X_test, Y_train, Y_test


def hypeparameters_determination(x_train, y_train):
    param_grid = {'C': np.logspace(-4, 1, num=100),
                  'kernel': ['linear', 'rbf', 'sigmoid', 'poly'], 'degree': np.arange(2, 3, 1), 'gamma': np.logspace(-4, 1, num=100)}
    grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=3, cv=10)
    grid.fit(x_train, y_train)
    grid_predictions = grid.predict(x_train)
    print("Best parameters ", grid.best_params_)
    # Train predictions accuracy is shown
    print("Train Accuracy after hypeparameters are determined: ", accuracy_score(y_train, grid_predictions))

    return grid.best_params_


training_images = np.load('Cartoon_Features_data.npy')
face_labels = np.load('All_face_types.npy')
tr_X, te_X, tr_Y, te_Y = data_preprocessing(training_images, face_labels)
"""model = GradientBoostingClassifier(random_state=0, max_depth=5, learning_rate=0.1)
model.fit(tr_X, tr_Y)
print("Accuracy on training set: {:.3f}".format(model.score(tr_X, tr_Y)))
print("Accuracy on test set: {:.3f}".format(model.score(te_X, te_Y)))"""

mlp = MLPClassifier(solver='lbfgs',hidden_layer_sizes=[300, 100], random_state=0, alpha=100, max_iter=2000).fit(tr_X, tr_Y)
print("Accuracy on training set: {:.3f}".format(mlp.score(tr_X, tr_Y)))
print("Accuracy on test set: {:.3f}".format(mlp.score(te_X, te_Y)))
