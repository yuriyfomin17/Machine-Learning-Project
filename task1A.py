
from sklearn.svm import SVC
from cvxopt import matrix as cvxopt_matrix
from cvxopt import solvers as cvxopt_solvers
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler  # doctest: +SKIP
import numpy as np
from sklearn.metrics import classification_report, accuracy_score
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import validation_curve
import itertools
import confusion_matrix
from sklearn.linear_model import ElasticNetCV


def img_SVM(training_data, training_labels, test_data, test_labels):
    # best parameters are given to SVM
    LinearClassifier = SVC(C=0.35111917342151344, kernel='linear')

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


def plot_learning_curve(x_train, y_train):
    LinearClassifier = SVC(C=0.35111917342151344, kernel='linear')
    y_train = np.array(y_train)
    y_train = y_train.astype(np.int8)
    indices = np.arange(y_train.shape[0])
    np.random.shuffle(indices)
    x_train, y_train = x_train[indices], y_train[indices]
    train_sizes = np.arange(0.05, 1.05, 0.05)
    x_range = train_sizes*3840
    train_sizes, train_scores, test_scores = learning_curve(LinearClassifier, x_train, y_train, cv=10, scoring='accuracy',
                                                            n_jobs=-1,
                                                            train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.figure(figsize=(10, 7))
    plt.ylim(0.8, 1.0)
    plt.yticks(np.arange(0.8, 1.001, 0.01))
    plt.ylabel("Accuracy")
    plt.xlabel("Training Examples")
    plt.title("Learning Curve (SVM, Linear Kernel, C=0.3511)")
    plt.xticks(x_range)
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1,
                     color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")
    plt.legend(loc="best")
    plt.grid()
    plt.show()
    return 0


def validation_curve_for_different_parameters_and_figures(tr_X, tr_Y):
    """# Validation curve for C parameter
    LinearClassifier = SVC(kernel='linear')
    C_range = np.logspace(-6, 2, num=100)
    train_scores, test_scores = validation_curve(
        LinearClassifier, tr_X, tr_Y, param_name="C", param_range=C_range,
        scoring="accuracy", n_jobs=-1, verbose=3, cv=10)
    train_scores_mean = np.mean(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    plt.figure(figsize=(8, 6))
    plt.title("Validation Curve with SVM")
    plt.xlabel("C")
    plt.ylabel("Score")
    plt.ylim(0.45, 1.0)
    plt.yticks(np.arange(0.45, 1.01, 0.05))
    plt.xlim(0.000001, 100)
    lw = 2
    plt.semilogx(C_range, train_scores_mean, label="Training score",
                 color="darkorange", lw=lw)
    plt.semilogx(C_range, test_scores_mean, label="Cross-validation score",
                 color="navy", lw=lw)
    plt.legend(loc="best")
    plt.grid()
    plt.show()"""

    # Validation curve for kernel parameter parameter
    """Kernel_parameter = ["linear", "poly", "rbf", "sigmoid"]
    LinearClassifier = SVC()
    train_scores, test_scores = validation_curve(
        LinearClassifier, tr_X, tr_Y, param_name="kernel", param_range=Kernel_parameter,
        scoring="accuracy", n_jobs=-1, verbose=3, cv=10)
    train_scores_mean = np.mean(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    plt.figure(figsize=(8, 6))
    plt.title("Validation Curve with SVM")
    plt.xlabel("Kernel Function")
    plt.ylabel("Score")
    plt.ylim(0.6, 1.05)
    plt.yticks(np.arange(0.6, 1.05, 0.05))
    lw = 2
    plt.plot(Kernel_parameter, train_scores_mean, marker="H", label="Training score",
             color="darkorange", lw=lw)
    plt.plot(Kernel_parameter, test_scores_mean, marker="s", label="Cross-validation score",
             color="navy", lw=lw)
    plt.legend(loc="best")
    plt.grid()
    plt.show()"""


def Classifier(X, Y, test_X):
    y_data = []
    for i in range(len(Y)):
        if Y[i] == 1.0:
            y_data.append(Y[i])
        else:
            y_data.append(-1)
    y = np.array(y_data)

    # Initializing values and computing H. Note the 1. to force to float type
    C = 0.09540954763499938
    m, n = X.shape
    y = y.reshape(-1, 1) * 1.
    X_dash = y * X
    H = np.dot(X_dash, X_dash.T) * 1.

    # Converting into cvxopt format - as previously
    P = cvxopt_matrix(H)
    q = cvxopt_matrix(-np.ones((m, 1)))
    G = cvxopt_matrix(np.vstack((np.eye(m) * -1, np.eye(m))))
    h = cvxopt_matrix(np.hstack((np.zeros(m), np.ones(m) * C)))
    A = cvxopt_matrix(y.reshape(1, -1))
    b = cvxopt_matrix(np.zeros(1))

    # Run solver
    sol = cvxopt_solvers.qp(P, q, G, h, A, b)
    alphas = np.array(sol['x'])

    # ==================Computing and printing parameters===============================#
    w = ((y * alphas).T @ X).reshape(-1, 1)
    S = (alphas > 0).flatten()
    b = y[S] - np.dot(X[S], w)
    w = w.reshape(1, 68)
    b = np.sum(b) / 3840
    y_pred = np.dot(w, test_X.T) + b
    return y_pred


def img_new_SVM(training_images, training_labels, test_images, test_labels):
    y_pred = Classifier(training_images, training_labels, test_images)
    Y_pred = []
    Test_label = []
    for i in range(960):
        Test_label.append(test_labels[i])
        if y_pred[0][i] < 0:
            Y_pred.append(0)
        else:
            Y_pred.append(1)
    print("Solver Linear Classifier Accuracy:", accuracy_score(Test_label, Y_pred))


def data_preprocessing(X, Y):
    # scaling of data

    Y[Y == 0] = -1
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
    X_train = X_train.reshape((3840, 68 * 2))
    X_test = X_test.reshape((960, 68 * 2))
    """scaler = StandardScaler()  # doctest: +SKIP
    # Don't cheat - fit only on training data
    scaler.fit(X_train)  # doctest: +SKIP
    X_train = scaler.transform(X_train)  # doctest: +SKIP
    # apply same transformation to test data
    X_test = scaler.transform(X_test)"""
    # compute the minimum value per feature on the training set
    min_on_training = X_train.min(axis=0)
    # compute the range of each feature (max - min) on the training set
    range_on_training = (X_train - min_on_training).max(axis=0)
    # subtract the min, and divide by range
    # afterward, min=0 and max=1 for each feature
    X_train = (X_train - min_on_training) / range_on_training
    X_test = (X_test - min_on_training) / range_on_training

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


def hypeparameters_determination(x_train, y_train):
    param_grid = {'C': np.logspace(-4, 1, num=100),
                  'kernel': ['linear']}
    grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=3, cv=10)
    grid.fit(x_train, y_train)
    grid_predictions = grid.predict(x_train)
    print("Best parameters ", grid.best_params_)
    return grid.best_params_


# Evaluation of Model - Confusion Matrix Plot
def plot_confusion_matrix(cm, target_names, title='Confusion matrix', cmap=None):
    tn, fp, fn, tp = cm.ravel()
    Accuracy = (tn + tp) * 100 / (tp + tn + fp + fn)
    Precision = tp / (tp + fp)
    Recall = tp / (tp + fn)
    f1 = (2 * Precision * Recall) / (Precision + Recall)

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(12, 10))
    plt.imshow(cm, interpolation='nearest', cmap=cmap, aspect='equal')
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(2)
        plt.xticks(tick_marks, target_names)
        plt.yticks(tick_marks, target_names)

    thresh = cm.max() / 2
    k = -0.25
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(x=j * 0.5 + 0.25 + k, y=i * 0.5 + 0.25, s=cm[i, j], horizontalalignment="center",
                 verticalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
        k = k * -1

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel(
        'Predicted label\n\n accuracy={:0.4f}\n Precision={:0.4f}\n Recall={:0.4f}\n F1={:0.4f} '.format(Accuracy,
                                                                                                         Precision,
                                                                                                         Recall, f1))
    plt.show()


training_images = np.load('Features_data.npy')
gender_labels = np.load('Gender_labels.npy')
tr_X, te_X, tr_Y, te_Y = data_preprocessing(training_images, gender_labels)
"param = hypeparameters_determination(tr_X, tr_Y)"
"validation_curve_for_different_parameters_and_figures(tr_X, tr_Y)"
"plot_learning_curve(tr_X, tr_Y)"
pred, model = img_SVM(tr_X, tr_Y, te_X, te_Y)
plot_confusion_matrix(confusion_matrix.confusion_matrix(te_Y, pred, [1, -1]),
                      target_names=['male', 'female'], title="Confusion Matrix")

img_SVM(tr_X, tr_Y, te_X, te_Y)
"img_new_SVM(tr_X, tr_Y, te_X, te_Y)"
