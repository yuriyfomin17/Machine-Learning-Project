import numpy as np
from sklearn.metrics import classification_report, accuracy_score
from sklearn.svm import SVC
from sklearn import svm
# neural network class definition

"""def predict_class(X, classifiers):
    predictions = np.arange(1563)
    for k in range(len(classifiers)):
        current_predict = classifiers[k].predict(X)
        indices = np.where(current_predict == 1)
        predictions[indices] = k
    pass
    return predictions"""


def predict_class(X, classifiers):
    predictions = np.zeros((X.shape[0], len(classifiers)))
    for idx, clf in enumerate(classifiers):
        predictions[:, idx] = clf.predict(X)
    # returns the class number if only one classifier predicted it # returns zero otherwise.
    return np.argmax(predictions, axis=1)


training_images = np.load('Cartoon_Features_data.npy')
face_labels = np.load('All_face_types.npy')

tr_X = training_images[:6252]
tr_Y = face_labels[:6252]
te_X = training_images[6252:]
te_Y = face_labels[6252:]

tr_X = tr_X.reshape((6252, 68 * 2))
te_X = te_X.reshape((1563, 68 * 2))

tr_Y0 = np.where(tr_Y == 0, 1, -1)
tr_Y1 = np.where(tr_Y == 1, 1, -1)
tr_Y2 = np.where(tr_Y == 2, 1, -1)
tr_Y3 = np.where(tr_Y == 3, 1, -1)
tr_Y4 = np.where(tr_Y == 4, 1, -1)
y_list = [tr_Y0, tr_Y1, tr_Y2, tr_Y3, tr_Y4]
classifiers_list = []
for i in range(5):
    LinearClassifier = SVC(C=0.001, kernel='linear')
    LinearClassifier.fit(tr_X, y_list[i])
    classifiers_list.append(LinearClassifier)

y_pred = predict_class(te_X, classifiers_list)
print("My Linear Classifier Accuracy:", accuracy_score(te_Y, y_pred))


LinearClassifier = svm.LinearSVC(C=0.001, multi_class='crammer_singer')
LinearClassifier.fit(tr_X, tr_Y)
pred = LinearClassifier.predict(te_X)
print("Crammer_singer:", accuracy_score(te_Y, pred))
