import numpy as np
import json
from activations import Activations


def MSE_loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def MSE_loss_derivative(y_true, y_pred):
    return 2 * (y_pred - y_true.T) / y_true.size

class Neural(Activations):
    def __init__(self):
        self.weights = []
        self.biases = []
        self.loss = 'MSE'
        self.layers = []
    
    def add(self, neurons, activation):
        self.layers.append({'neurons': neurons, 'activation': activation})
    
    def random_params(self):
        for i in range(1, len(self.layers)):
            self.weights.append(2 * np.random.random((self.layers[i-1]['neurons'], self.layers[i]['neurons'])) - 1)
            self.biases.append(np.zeros((1, self.layers[i]['neurons'])))
    
    def fit(self, X, y, epochs=1000, learning_rate=0.01):
        self.random_params()

        for epoch in range(epochs):
            # forward
            activations = [X]

            for i in range(len(self.layers)-1):
                if self.layers[i]['activation'] == 'relu':
                    activations.append(self.ReLU(np.dot(activations[-1], self.weights[i])+self.biases[i]))
                elif self.layers[i]['activation'] == 'sigmoid':
                    activations.append(self.sigmoid(np.dot(activations[-1], self.weights[i])+self.biases[i]))
                elif self.layers[i]['activation'] == 'softmax':
                    activations.append(self.softmax(np.dot(activations[-1], self.weights[i])+self.biases[i]))
            
            #backward
            if self.loss == 'MSE':
                self.deltas = MSE_loss_derivative(y, activations[-1])       
            elif self.loss == 'cross-entropy':
                self.deltas = cross_entropy_loss_derivative(y, activations[-1])
            
            for i in range(len(activations)-2, -1, -1):
                if self.layers[i]['activation'] == 'relu':
                    self.deltas = self.deltas.dot(self.weights[i].T) * self.ReLU_derivative(activations[i])
                elif self.layers[i]['activation'] == 'sigmoid':
                    self.deltas = self.deltas.dot(self.weights[i].T) * self.sigmoid_derivative(activations[i])
                elif self.layers[i]['activation'] == 'softmax':
                    self.deltas = self.deltas.dot(self.weights[i].T) * self.softmax_derivative(activations[i])
                
                self.weights[i] = self.weights[i] - learning_rate * activations[i].T.dot(self.deltas)
                self.biases[i] = self.biases[i-1] - learning_rate * np.sum(self.deltas, axis=0, keepdims=True)
        print(self.weights)
            
    def save_model(self, model_path):
        model_data = {}
        model_data['layers'] = self.layers
        model_data['weights'] = [w.tolist() for w in self.weights]
        model_data['bias'] = [b.tolist() for b in self.biases]
        model_data['loss'] = self.loss
        with open(model_path, 'w') as outfile:
            json.dump(model_data, outfile)

    @classmethod
    def load_model(cls, model_path):
        with open(model_path) as json_file:
            model_data = json.load(json_file)
        model = cls()
        model.layers = model_data['layers']
        model.weights = [np.array(w) for w in model_data['weights']]
        model.bias = [np.array(b) for b in model_data['bias']]
        model.loss = model_data['loss']
        return model

def cross_entropy_loss_derivative(y_true, y_pred):
    return y_pred - y_true