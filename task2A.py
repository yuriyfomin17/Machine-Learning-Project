from sklearn.metrics import classification_report, accuracy_score
from sklearn.svm import SVC
import numpy as np
from cvxopt import matrix as cvxopt_matrix
from cvxopt import solvers as cvxopt_solvers
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.datasets import load_digits


def img_SVM(training_images, training_labels, test_images, test_labels):
    LinearClassifier = SVC(C=0.0121, kernel='linear')
    LinearClassifier.fit(training_images, training_labels)
    print('w = ', LinearClassifier.coef_)
    print('b = ', LinearClassifier.intercept_)
    LinearPred = LinearClassifier.predict(test_images)
    print("Linear Classifier Accuracy:", accuracy_score(test_labels, LinearPred))


def plot_learning_curve(x, y):
    digits = load_digits()
    LinearClassifier = SVC(C=1, kernel='poly', degree=3)
    y = np.array(y)
    y = np.where(y == 0, -1, y)
    y = np.where(y == 1, 1, y)
    y = y.astype(np.int8)
    indices = np.arange(y.shape[0])
    np.random.shuffle(indices)
    x, y = x[indices], y[indices]
    train_sizes = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    train_sizes, train_scores, test_scores = learning_curve(LinearClassifier, x, y, cv=10, scoring='accuracy',
                                                            n_jobs=-1,
                                                            train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.figure(figsize=(20, 5))
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


def cross_validation(LinearClassifier, training_images, training_labels):
    C_value = np.arange(1, 20, 1)
    scores = []
    scoresSTD = []
    for C in C_value:
        LinearClassifier.C = C
        current_score = cross_val_score(LinearClassifier, training_images, training_labels, n_jobs=1)
        scores.append(np.mean(current_score))
        scoresSTD.append(np.std(current_score))
    pass
    plt.figure(figsize=(10, 10))
    plt.plot(C_value, scores)
    plt.plot(C_value, np.array(scores) + np.array(scoresSTD), 'b--')
    plt.plot(C_value, np.array(scores) - np.array(scoresSTD), 'b--')
    plt.xticks(np.arange(0, 20, 1))
    plt.xlabel('C parameter')
    plt.yticks(np.arange(0, 1, 0.1))
    plt.ylabel('Cross Validation accuracy')
    plt.show()
    scores.clear()
    scoresSTD.clear()
    Models = ['linear', 'poly', 'rbf', 'sigmoid']
    for i in range(len(Models)):
        LinearClassifier.kernel = Models[i]
        current_score = cross_val_score(LinearClassifier, training_images, training_labels, n_jobs=1)
        scores.append(np.mean(current_score))
        scoresSTD.append(np.std(current_score))
    pass
    plt.figure(figsize=(10, 10))
    plt.plot(Models, scores)
    plt.plot(Models, np.array(scores) + np.array(scoresSTD), 'b--')
    plt.plot(Models, np.array(scores) - np.array(scoresSTD), 'b--')
    plt.xlabel('C parameter')
    plt.yticks(np.arange(0, 1, 0.1))
    plt.ylabel('Cross Validation accuracy')
    plt.show()


def Classifier(X, Y, test_X):
    y_data = []
    for i in range(len(Y)):
        if Y[i] == 1.0:
            y_data.append(Y[i])
        else:
            y_data.append(-1)
    y = np.array(y_data)

    # Initializing values and computing H. Note the 1. to force to float type
    C = 0.0121
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
    w = w.reshape(1, 136)
    "print('Alphas = ', alphas[alphas > 1e-4])"
    print('w = ', w.flatten())
    b = np.sum(b) / 3840
    print(b)
    y_pred = np.dot(w, test_X.T) + b
    return y_pred


def img_new_SVM(training_images, training_labels, test_images, test_labels):
    y_pred = Classifier(training_images, training_labels, test_images)
    Y_pred = []
    Test_label = []
    for i in range(960):
        Test_label.append(test_labels[i])
        if y_pred[0][i] < 0:
            Y_pred.append(-1)
        else:
            Y_pred.append(1)
    print("My Linear Classifier Accuracy:", accuracy_score(Test_label, Y_pred))


# Loading the data file
training_images = np.load('Features_data.npy')
gender_labels = np.load('Smiling_labels.npy')
tr_X = training_images[:3840]
tr_Y = gender_labels[:3840]
te_X = training_images[3840:]
te_Y = gender_labels[3840:]
tr_X = tr_X.reshape((3840, 68 * 2))
te_X = te_X.reshape((960, 68 * 2))
img_SVM(tr_X, tr_Y, te_X, te_Y)
img_new_SVM(tr_X, tr_Y, te_X, te_Y)
