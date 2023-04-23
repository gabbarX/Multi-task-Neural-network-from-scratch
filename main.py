import numpy as np
import pandas as pd


class NeuralNetwork:
    def __init__(self, input_dim, hidden_dim, output_dim):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.W1 = np.random.randn(self.input_dim, self.hidden_dim[0])
        self.b1 = np.zeros((1, self.hidden_dim[0]))
        self.W2 = np.random.randn(self.hidden_dim[0], self.hidden_dim[1])
        self.b2 = np.zeros((1, self.hidden_dim[1]))

        self.W3 = np.random.randn(self.hidden_dim[1], self.output_dim[0])
        self.b3 = np.zeros((1, self.output_dim[0]))
        self.W4 = np.random.randn(self.hidden_dim[1], self.output_dim[1])
        self.b4 = np.zeros((1, self.output_dim[1]))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def tanh(self, z):
        return np.tanh(z)

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def tanh_derivative(self, x):
        return 1 - np.tanh(x) ** 2

    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.sigmoid(self.z1)

        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.tanh(self.z2)

        self.z3 = np.dot(self.a2, self.W3) + self.b3
        self.regression_output = self.z3

        self.z4 = np.dot(self.a2, self.W4) + self.b4
        self.classification_output = self.sigmoid(self.z4)

        # print(self.classification_output)

        return self.regression_output, self.classification_output

    def backward(self, X, y_regression, y_classification, learning_rate):
        regression_output, classification_output = self.forward(X)

        regression_delta = (regression_output - y_regression) / len(X)
        regression_gradient = np.dot(self.a2.T, regression_delta)
        self.W3 = self.W3 - learning_rate * regression_gradient
        self.b3 = self.b3 - learning_rate * np.sum(regression_delta, axis=0)

        classification_delta = (classification_output - y_classification) / len(X)
        classification_gradient = np.dot(self.a2.T, classification_delta)
        self.W4 = self.W4 - learning_rate * classification_gradient
        self.b4 = self.b4 - learning_rate * np.sum(classification_delta, axis=0)

        hidden2_delta = np.dot(classification_delta, self.W4.T) * self.tanh_derivative(
            self.z2
        )
        hidden2_gradient = np.dot(self.a1.T, hidden2_delta)
        self.W2 = self.W2 - learning_rate * hidden2_gradient
        self.b2 = self.b2 - learning_rate * np.sum(hidden2_delta, axis=0)

        hidden1_delta = np.dot(hidden2_delta, self.W2.T) * self.sigmoid_derivative(
            self.z1
        )
        hidden1_gradient = np.dot(X.T, hidden1_delta)
        self.W1 = self.W1 - learning_rate * hidden1_gradient
        self.b1 = self.b1 - learning_rate * np.sum(hidden1_delta, axis=0)

    def train(self, X, y_regression, y_classification, learning_rate, epochs):
        for i in range(epochs):
            regression_output, classification_output = self.forward(X)
            self.backward(X, y_regression, y_classification, learning_rate)

            if i % 1000 == 0:
                print("Epoch:", i)
                print(
                    "Regression Loss:",
                    np.mean(np.abs(regression_output - y_regression)),
                )
                print(
                    "Classification Accuracy:",
                    np.mean((classification_output > 0.5) == y_classification),
                )

    def predict(self, X):
        regression_output, classification_output = self.forward(X)
        return regression_output, classification_output


nn = NeuralNetwork(input_dim=2, hidden_dim=[4, 4], output_dim=[1, 1])


data = pd.read_csv("mt.csv")
# print(data.head())
inputs = data[["F1", "F2"]].to_numpy()
y_binary = data["T1"].to_numpy()
y_regression = data["T2"].to_numpy()
# print(inputs)

split_ratio = 0.8
split_index = int(split_ratio * len(inputs))

# Split the data into training and validation sets
train_inputs = inputs[:split_index]
train_classification = y_binary[:split_index]
train_regression = y_regression[:split_index]

val_inputs = inputs[split_index:]
val_classification = y_binary[split_index:]
val_regression = y_regression[split_index:]

learning_rate = 0.1
epochs = 100

# print(train_inputs.shape)
# print(train_classification.shape)
# print(train_classification.shape)

nn.train(train_inputs, train_regression, train_classification, learning_rate, epochs)

regression_output, classification_output = nn.predict(val_inputs)
print("Regression Output:", regression_output)
print("Classification Output:", classification_output)
