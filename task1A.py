
from sklearn.metrics import classification_report, accuracy_score
from sklearn.svm import SVC
import os.path
import numpy as np
from keras.preprocessing import image
import cv2
import dlib
from cvxopt import matrix as cvxopt_matrix
from cvxopt import solvers as cvxopt_solvers
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.datasets import load_digits
from tempfile import TemporaryFile

global basedir, image_paths, target_size
basedir = './dataset_AMLS_19-20'
images_dir = os.path.join(basedir, 'celeba/img')
labels_filename = 'labels.csv'

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')


def shape_to_np(shape, dtype="int"):
    # initialize the list of (x, y)-coordinates
    coords = np.zeros((shape.num_parts, 2), dtype=dtype)

    # loop over all facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, shape.num_parts):
        coords[i] = (shape.part(i).x, shape.part(i).y)

    # return the list of (x, y)-coordinates
    return coords


def rect_to_bb(rect):
    # take a bounding predicted by dlib and convert it
    # to the format (x, y, w, h) as we would normally do
    # with OpenCV
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y

    # return a tuple of (x, y, w, h)
    return (x, y, w, h)


def run_dlib_shape(image):
    # in this function we load the image, detect the landmarks of the face, and then return the image and the landmarks
    # load the input image, resize it, and convert it to grayscale
    resized_image = image.astype('uint8')

    gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    gray = gray.astype('uint8')

    # detect faces in the grayscale image
    rects = detector(gray, 1)
    num_faces = len(rects)

    if num_faces == 0:
        return None, resized_image

    face_areas = np.zeros((1, num_faces))
    face_shapes = np.zeros((136, num_faces), dtype=np.int64)

    # loop over the face detections
    for (i, rect) in enumerate(rects):
        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy
        # array
        temp_shape = predictor(gray, rect)
        temp_shape = shape_to_np(temp_shape)

        # convert dlib's rectangle to a OpenCV-style bounding box
        # [i.e., (x, y, w, h)],
        #   (x, y, w, h) = face_utils.rect_to_bb(rect)
        (x, y, w, h) = rect_to_bb(rect)
        face_shapes[:, i] = np.reshape(temp_shape, [136])
        face_areas[0, i] = w * h
    # find largest face and keep
    dlibout = np.reshape(np.transpose(face_shapes[:, np.argmax(face_areas)]), [68, 2])

    return dlibout, resized_image


def extract_features_labels():
    """
    This funtion extracts the landmarks features for all images in the folder 'dataset/celeba'.
    It also extracts the gender label for each image.
    :return:
        landmark_features:  an array containing 68 landmark points for each image in which a face was detected
        gender_labels:      an array containing the gender label (male=0 and female=1) for each image in
                            which a face was detected
    """
    image_paths = [os.path.join(images_dir, l) for l in os.listdir(images_dir)]
    target_size = None
    labels_file = open(os.path.join(basedir, labels_filename), 'r')
    lines = labels_file.readlines()
    gender_labels = {line.split()[0]: int(line.split()[2]) for line in lines[1:]}
    if os.path.isdir(images_dir):
        all_features = []
        all_labels = []
        for img_path in image_paths:
            file_name = img_path.split('.')[1].split('/')[-1]

            # load image
            img = image.img_to_array(
                image.load_img(img_path,
                               target_size=target_size,
                               interpolation='bicubic'))
            features, _ = run_dlib_shape(img)
            if features is not None:
                all_features.append(features)
                all_labels.append(gender_labels[file_name])

    landmark_features = np.array(all_features)
    gender_labels = (np.array(all_labels) + 1) / 2  # simply converts the -1 into 0, so male=0 and female=1
    return landmark_features, gender_labels


def get_data():
    X, y = extract_features_labels()
    Y = np.array([y, -(y - 1)]).T
    tr_X = X[:3840]
    tr_Y = Y[:3840]
    te_X = X[3840:]
    te_Y = Y[3840:]
    tr_X = tr_X.reshape((3840, 68 * 2))
    tr_Y = list(zip(*tr_Y))[0]
    te_X = te_X.reshape((960, 68 * 2))
    te_Y = list(zip(*te_Y))[0]
    np.save('A1_train_x_data.npy', tr_X)
    np.save('A1_train_y_data.npy', tr_Y)
    np.save('A1_test_x_data.npy', te_X)
    np.save('A1_test_y_data.npy', te_Y)
    return X, Y


def img_SVM(training_images, training_labels, test_images, test_labels):
    LinearClassifier = SVC(C=1, kernel='linear')
    LinearClassifier.fit(training_images, training_labels)
    LinearPred = LinearClassifier.predict(test_images)
    print("Linear Classifier Accuracy:", accuracy_score(test_labels, LinearPred))


def plot_learning_curve(x, y):
    digits = load_digits()
    LinearClassifier = SVC(C=1, kernel='linear')
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
    C = 1
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
    print('Alphas = ', alphas[alphas > 1e-4])
    print('w = ', w.flatten())
    b = np.sum(b) / 3840
    y_pred = np.dot(w, test_X.T) + b
    return y_pred


def Poly_data_mapping(x_train, y_train):
    copy = x_train.copy()
    for i in range(0, 136, 2):
        x_train[:, i] = x_train[:, i] ** 2
    for i in range(1, 136, 2):
        x_train[:, i] = np.sqrt(2) * copy[:, 0] * x_train[:, 1]
    for i in range(2, 204, 2):
        np.insert(x_train, i, copy[:, i - 1] ** 2, axis=1)

    return x_train


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
    print("My Linear Classifier Accuracy:", accuracy_score(Test_label, Y_pred))


# Loading the data file
training_images = np.load('A1_train_x_data.npy')
training_labels = np.load('A1_train_y_data.npy')
test_images = np.load('A1_test_x_data.npy')
test_labels = np.load('A1_test_y_data.npy')
img_SVM(training_images, training_labels, test_images, test_labels)
img_new_SVM(training_images, training_labels, test_images, test_labels)
