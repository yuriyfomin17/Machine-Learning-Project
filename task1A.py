
mport pandas
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score


# sklearn functions implementation => linearRegPredic function

def linearRegrPredict(xTrain, yTrain, xTest):
    # Create linear regression object

    regr = LinearRegression()

    # Train the model using the training sets

    regr.fit(xTrain, yTrain)
    # Make predictions using the testing set

    y_pred = regr.predict(xTest)
    beta = regr.coef_
    alpha = regr.intercept_

    # print("Accuracy Score:",regr.score(xTest,yTest))

    return y_pred


# My own simple Linear Regression Prediction Function

def sampleMean(Sample_data, sample_size):
    sum = 0
    for i in range(sample_size):
        sum = sum + Sample_data[i]
    sample_mean = sum / sample_size
    return sample_mean


def CalculatingBeta(XTrain1, YTrain1, x_train1_sample_mean, y_train1_sample_mean, sample_size):
    numerator_sum = 0
    for i in range(sample_size):
        numerator_sum = numerator_sum + XTrain1[i] * (YTrain1[i] - y_train1_sample_mean)
    denominator_sum = 0
    for i in range(sample_size):
        denominator_sum = denominator_sum + XTrain1[i] * (XTrain1[i] - x_train1_sample_mean)
    beta = numerator_sum / denominator_sum
    return beta


def MyRegFunction(XTrain1, YTrain1, XTest1):
    # total data 260 multiplied by training set proportion

    sample_size = 195
    x_train1_sample_mean = sampleMean(XTrain1, sample_size)
    y_train1_sample_mean = sampleMean(YTrain1, sample_size)
    beta = CalculatingBeta(XTrain1, YTrain1, x_train1_sample_mean, y_train1_sample_mean, sample_size)
    alpha = y_train1_sample_mean - beta * x_train1_sample_mean
    beta = round(beta, 6)
    alpha = round(alpha, 6)

    y_pred = alpha + beta * xTest1
    # rounding y_pred to 6 digits
    # total data 260 multiplied by test set proportion
    for i in range(260 - sample_size):
        y_pred[i] = round(y_pred[i], 6)

    print("My regression alpha = ", alpha)
    print("My regression beta  = ", beta)
    return y_pred


# Solution Regression function from Jupyter Notebook

def paramEstimates(xTrain, yTrain):
    beta = np.sum(np.multiply(xTrain, (np.add(yTrain, -np.mean(yTrain))))) / np.sum(
        np.multiply(xTrain, (np.add(xTrain, -np.mean(xTrain)))))
    xTrain1 = np.array(np.mean(xTrain)).flatten()
    yTrain1 = np.array(np.mean(yTrain)).flatten()
    alpha = yTrain1 - beta * xTrain1
    print("Jupyter alpha", alpha)
    print("Jupyter beta", beta)
    return alpha, beta


def linearRegrNEWPredict(xTrain, yTrain, xTest):
    alpha, beta = paramEstimates(xTrain, yTrain)
    y_pred = alpha + beta * xTest
    return y_pred


# My SSR function
def Small_Res_Square_Value(yTest, yPrediction, TestSample_size):
    sum = 0.00000000000
    for i in range(TestSample_size):
        sum = sum + (yTest[i] - yPrediction[i]) * (yTest[i] - yPrediction[i])

    return sum


def SSR(yTest, y_pred):
    ssr = np.sum(np.multiply((np.add(yTest, -y_pred)), (np.add(yTest, -y_pred))))
    return ssr


# Gradient Descent

def GDparamEstimates(xTrain, yTrain, xTest):
    alpha = 0
    beta = 0
    n = len(xTrain)
    iterations = 100000
    AlphaLearning_rate = 0.001
    BetaLearning_rate = 0.000000000001
    for i in range(iterations):
        y_predicted = alpha + beta * xTrain
        SumAlphaGrad = -(2 / n) * np.sum(yTrain - y_predicted)
        SumBetaGrad = -(2 / n) * np.sum(xTrain * (yTrain - y_predicted))
        alpha = alpha - AlphaLearning_rate * SumAlphaGrad
        beta = beta - BetaLearning_rate * SumBetaGrad
    y_predicted = alpha + beta * xTest
    print(" GD alpha = ", alpha)
    print("GD beta = ", beta)
    return y_predicted


# Loading the CSV file

houseprice = pandas.read_csv('regression_data.csv')
houseprice = houseprice[['Price (Older)', 'Price (New)']]  # Choose 2 columns

# Split the data

X = houseprice[['Price (Older)']]
Y = houseprice[['Price (New)']]

# Split the data into training and testing(75% training and 25% testing data)

xTrain, xTest, yTrain, yTest = train_test_split(X, Y)

# Transform dataframes to numpy arrays
xTrain1 = np.array(xTrain.values).flatten()
xTest1 = np.array(xTest.values).flatten()
yTrain1 = np.array(yTrain.values).flatten()
yTest1 = np.array(yTest.values).flatten()
Array = np.array(xTest.values).flatten()
ypredGD = GDparamEstimates(xTrain1, yTrain1, xTest1)

y_pred1 = MyRegFunction(xTrain1, yTrain1, xTest1)
y_predSklearn = linearRegrPredict(xTrain, yTrain, xTest)
y_predJupyter = linearRegrNEWPredict(xTrain, yTrain, xTest)
test_sample_size = 65
SSR_of_y_pred1 = SSR(yTest1, y_pred1)
print("SSR of my Simple Regression function: ", SSR_of_y_pred1)

SklearnSSR = SSR(yTest, y_predSklearn)
print("Scikit-learn linear regression SSR: %.4f" % SklearnSSR)

Jupyter = SSR(yTest, y_predJupyter)
print("Jupyter SSR: %.4f" % Jupyter)

SSRGD = SSR(yTest1, ypredGD)
print(" Gradient descent: ", SSRGD)

# y_pred = linearRegrPredict(xTrain, yTrain, xTest)
# print(y_pred)

# Plot testing set predictions
plt.title('My Regression')
plt.scatter(xTest, yTest)
plt.plot(xTest1, y_pred1, 'r-')
plt.show()

plt.title('Sklearn Regression')
plt.scatter(xTest, yTest)
plt.plot(xTest, y_predSklearn, 'r-')
plt.show()

plt.title('Jupiter Solution Regression')
plt.scatter(xTest, yTest)
plt.plot(xTest1, y_predJupyter, 'r-')
plt.show()
print("Hello")

# EXample of my Neural Network
import numpy
import scipy.special
import matplotlib.pyplot as plt


# neural network class definition

class neuralNetwork:

    # initialise the neural network
    def __init__(self, inputN, hiddenN, outputN, lr):
        # set number of nodes in each input, hidden, output layer
        self.inodes = inputN
        self.hnodes = hiddenN
        self.onodes = outputN

        # link weight matrices, wih and who
        # weights inside the arrays are w_i_j, where link is from node i to node j in the next layer
        # w11 w21
        # w12 w22 etc
        self.wih = numpy.random.normal(0.0, pow(self.inodes, -0.5), (self.hnodes, self.inodes))
        self.who = numpy.random.normal(0.0, pow(self.hnodes, -0.5), (self.onodes, self.hnodes))

        # learning rate
        self.lr = lr

        # activation function is the sigmoid function
        self.activation_function = lambda x: scipy.special.expit(x)

        pass

    # train the neural network
    def train(self, inputs_list, targets_list):
        # convert inputs list to 2d array
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T

        # calculate signals into hidden layer
        hidden_inputs = numpy.dot(self.wih, inputs)
        # calculate the signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)

        # calculate signals into final output layer
        final_inputs = numpy.dot(self.who, hidden_outputs)
        # calculate the signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs)

        # output layer error is the (target - actual)
        output_errors = targets - final_outputs
        # hidden layer error is the output_errors, split by weights, recombined at hidden nodes
        hidden_errors = numpy.dot(self.who.T, output_errors)

        # update the weights for the links between the hidden and output layers
        self.who += self.lr * numpy.dot((output_errors * final_outputs * (1.0 - final_outputs)),
                                        numpy.transpose(hidden_outputs))

        # update the weights for the links between the input and hidden layers
        self.wih += self.lr * numpy.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)),
                                        numpy.transpose(inputs))

        pass

    # query the neural network
    def query(self, inputs_list):
        # convert inputs list to 2d array
        inputs = numpy.array(inputs_list, ndmin=2).T

        # calculate signals into hidden layer
        hidden_inputs = numpy.dot(self.wih, inputs)
        # calculate the signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)

        # calculate signals into final output layer
        final_inputs = numpy.dot(self.who, hidden_outputs)
        # calculate the signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs)

        return final_outputs


# number of input, hidden and output nodes
input_nodes = 784
hidden_nodes = 100
output_nodes = 10

# learning rate
learning_rate = 0.1

# create instance of neural network
n = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)
training_data_file = open("DataSet/mnist_train.csv", 'r')
training_data_list = training_data_file.readlines()
training_data_file.close()

# train the neural network

# epochs is the number of times the training data set is used for training
epochs = 5

for e in range(epochs):
    # go through all records in the training data set
    for record in training_data_list:
        # split the record by the ',' commas
        all_values = record.split(',')
        # scale and shift the inputs
        inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
        # create the target output values (all 0.01, except the desired label which is 0.99)
        targets = numpy.zeros(output_nodes) + 0.01
        # all_values[0] is the target label for this record
        targets[int(all_values[0])] = 0.99
        n.train(inputs, targets)
        pass
    pass

# load the mnist test data CSV file into a list
test_data_file = open("DataSet/mnist_test.csv", 'r')
test_data_list = test_data_file.readlines()
test_data_file.close()

# test the neural network

# scorecard for how well the network performs, initially empty
scorecard = []

# go through all the records in the test data set
for record in test_data_list:
    # split the record by the ',' commas
    all_values = record.split(',')
    # correct answer is first value
    correct_label = int(all_values[0])
    # scale and shift the inputs
    inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
    # query the network
    outputs = n.query(inputs)
    # the index of the highest value corresponds to the label
    label = numpy.argmax(outputs)
    # append correct or incorrect to list
    if label == correct_label:
        # network's answer matches correct answer, add 1 to scorecard
        scorecard.append(1)
    else:
        # network's answer doesn't match correct answer, add 0 to scorecard
        scorecard.append(0)
        pass

    pass

# calculate the performance score, the fraction of correct answers
scorecard_array = numpy.asarray(scorecard)
print("performance = ", scorecard_array.sum() / scorecard_array.size)

# test the neural network

# scorecard for how well the network performs, initially empty
scorecard = []

# go through all the records in the test data set
for record in test_data_list:
    # split the record by the ',' commas
    all_values = record.split(',')
    # correct answer is first value
    correct_label = int(all_values[0])
    # scale and shift the inputs
    inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
    # query the network
    outputs = n.query(inputs)
    # the index of the highest value corresponds to the label
    label = numpy.argmax(outputs)
    # append correct or incorrect to list
    if label == correct_label:
        # network's answer matches correct answer, add 1 to scorecard
        scorecard.append(1)
    else:
        # network's answer doesn't match correct answer, add 0 to scorecard
        scorecard.append(0)
        pass

    pass

# calculate the performance score, the fraction of correct answers
scorecard_array = numpy.asarray(scorecard)
print("performance = ", scorecard_array.sum() / scorecard_array.size)
