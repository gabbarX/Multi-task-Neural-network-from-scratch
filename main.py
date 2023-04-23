import numpy as np
import pandas as pd


class MultiTaskNeuralNetwork:
    def __init__(self, nInput, nHidden, nOutput):
        self.nInput = nInput
        self.nHidden = nHidden
        self.nOutput = nOutput

        self.weight1 = np.random.randn(self.nInput, self.nHidden[0])
        self.weight2 = np.random.randn(self.nHidden[0], self.nHidden[1])
        self.weight3 = np.random.randn(self.nHidden[1], self.nOutput[0])
        self.weight4 = np.random.randn(self.nHidden[1], self.nOutput[1])

        self.bias1 = np.zeros((1, self.nHidden[0]))
        self.bias2 = np.zeros((1, self.nHidden[1]))
        self.bias3 = np.zeros((1, self.nOutput[0]))
        self.bias4 = np.zeros((1, self.nOutput[1]))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def tanh(self, z):
        return np.tanh(z)

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def tanh_derivative(self, x):
        return 1 - np.tanh(x) ** 2

    def forward(self, X):
        self.z1 = np.dot(X, self.weight1) + self.bias1
        self.a1 = self.sigmoid(self.z1)

        self.z2 = np.dot(self.a1, self.weight2) + self.bias2
        self.a2 = self.tanh(self.z2)

        # These both making use of A2
        self.z3 = np.dot(self.a2, self.weight3) + self.bias3
        self.regression_output = self.z3

        self.z4 = np.dot(self.a2, self.weight4) + self.bias4
        self.classification_output = self.sigmoid(self.z4)
        # print(self.classification_output)
        return self.regression_output, self.classification_output

    def backward(self, X, y_regression, y_classification, learning_rate):
        regression_output, classification_output = self.forward(X)

        regression_delta = (regression_output - y_regression) / len(X)
        regression_gradient = np.dot(self.a2.T, regression_delta)
        self.weight3 = self.weight3 - learning_rate * regression_gradient
        self.bias3 = self.bias3 - learning_rate * np.sum(regression_delta, axis=0)

        classification_delta = (classification_output - y_classification) / len(X)
        classification_gradient = np.dot(self.a2.T, classification_delta)
        self.weight4 = self.weight4 - learning_rate * classification_gradient
        self.bias4 = self.bias4 - learning_rate * np.sum(classification_delta, axis=0)

        hidden2_delta = np.dot(
            classification_delta, self.weight4.T
        ) * self.tanh_derivative(self.z2)
        hidden2_gradient = np.dot(self.a1.T, hidden2_delta)
        self.weight2 = self.weight2 - learning_rate * hidden2_gradient
        self.bias2 = self.bias2 - learning_rate * np.sum(hidden2_delta, axis=0)

        hidden1_delta = np.dot(hidden2_delta, self.weight2.T) * self.sigmoid_derivative(
            self.z1
        )
        hidden1_gradient = np.dot(X.T, hidden1_delta)
        self.weight1 = self.weight1 - learning_rate * hidden1_gradient
        self.bias1 = self.bias1 - learning_rate * np.sum(hidden1_delta, axis=0)

    def train(self, X, y_regression, y_classification, learning_rate, iter):
        for _ in range(iter):
            regression_output, classification_output = self.forward(X)
            self.backward(X, y_regression, y_classification, learning_rate)

    def predict(self, X):
        regression_output, classification_output = self.forward(X)
        return regression_output, classification_output

    def calculateStuff(self, yr_pred, yc_pred, y_regression, y_classification):
        print(
            "=================|VALIDATION (20%) (30 datapoints)| ====================="
        )
        print("Regression Loss:", np.mean(np.abs(yr_pred - y_regression)))
        print(
            "Classification Accuracy:",
            np.mean(yc_pred == y_classification),
        )


splitRatio = 0.8
learning_rate = 0.1
iterations = 1000


network = MultiTaskNeuralNetwork(nInput=2, nHidden=[4, 4], nOutput=[1, 1])
data = pd.read_csv("mt.csv")
# print(data.head())
inputs = data[["F1", "F2"]].to_numpy()
y_binary = data["T1"].to_numpy()
y_regression = data["T2"].to_numpy()
# print(inputs)
splitIdx = int(splitRatio * len(inputs))

# Split the data into training and validation sets
X_train = inputs[:splitIdx]
y_train_clf = y_binary[:splitIdx]
y_train_reg = y_regression[:splitIdx]

X_test = inputs[splitIdx:]
X_test_clf = y_binary[splitIdx:]
X_test_reg = y_regression[splitIdx:]
network.train(X_train, y_train_reg, y_train_clf, learning_rate, iterations)

regression_output, classification_output = network.predict(X_test)
regression_output = np.mean(regression_output, axis=1)
classification_output = np.mean(classification_output, axis=1)
classification_output = np.where(classification_output >= 0.5, 2, 1)
network.calculateStuff(regression_output, classification_output, X_test_reg, X_test_clf)

# print(X_test_clf)
