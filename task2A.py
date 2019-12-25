from sklearn.metrics import classification_report, accuracy_score
from sklearn.svm import SVC
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import ElasticNetCV


def img_SVM(training_data, training_labels, test_data, test_labels):
    # best parameters are given to SVM
    LinearClassifier = SVC(C=0.006579332246575682, kernel='linear', degree=2)

    # Model is trained on training data
    LinearClassifier.fit(training_data, training_labels)

    # Training predictions are given
    train_predictions = LinearClassifier.predict(training_data)

    # Test predictions are given
    test_predictions = LinearClassifier.predict(test_data)

    # Train predictions accuracy is shown
    print("Train Accuracy: ", accuracy_score(training_labels, train_predictions))

    # Cross validation accuracy and report is given on cv=10
    Validation_Score = cross_val_score(LinearClassifier, training_data, training_labels, cv=10)
    print("Separate Validation score  for each: ", Validation_Score)
    print("Validation score mean over cv=10: ", Validation_Score.mean())

    # Test final predictions accuracy is given
    print("Test Accuracy: ", accuracy_score(test_labels, test_predictions))
    print("Test report: ", classification_report(test_labels, test_predictions))
    return test_predictions, LinearClassifier


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


def data_preprocessing(X, Y):
    # scaling of data

    Y[Y == 0] = -1
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
    X_train = X_train.reshape((3840, 68 * 2))
    X_test = X_test.reshape((960, 68 * 2))
    scaler = StandardScaler()  # doctest: +SKIP
    # Don't cheat - fit only on training data
    scaler.fit(X_train)  # doctest: +SKIP
    X_train = scaler.transform(X_train)  # doctest: +SKIP
    # apply same transformation to test data
    X_test = scaler.transform(X_test)

    # PCA analysis

    pca = PCA(n_components=68)
    pca.fit(X_train)
    X_train = pca.transform(X_train)
    X_test = pca.transform(X_test)

    # Feature selection using lasso and ridge regression

    ElasticNet = ElasticNetCV(cv=10, random_state=0)
    ElasticNet.fit(X_train, Y_train)
    all_features = ElasticNet.coef_
    not_important_features_indices = np.where(all_features == 0)[0]
    X_train = np.delete(X_train, not_important_features_indices, axis=1)
    X_test = np.delete(X_test, not_important_features_indices, axis=1)
    return X_train, X_test, Y_train, Y_test


# Loading the data file
training_images = np.load('Features_data.npy')
smiling_labels = np.load('Smiling_labels.npy')
tr_X, te_X, tr_Y, te_Y = data_preprocessing(training_images, smiling_labels)
img_SVM(tr_X, tr_Y, te_X, te_Y)
